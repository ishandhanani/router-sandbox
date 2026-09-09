// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicyError,
};
use lru::LruCache;

use crate::SessionAffinityConfig;

/// Cache-aware fallback cost, in tokens, for a request that has no usable remembered
/// worker: the prefill the worker still has to run for this prompt plus its in-flight
/// prefill and decode work.
pub(crate) struct SessionAffinityScorer;

impl WorkerScorer for SessionAffinityScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD | WorkerInputs::CACHE
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let overlap_blocks = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?
            .device_overlap_blocks();
        if !overlap_blocks.is_finite() || overlap_blocks < 0.0 {
            return Err(WorkerSelectionPolicyError::failed(
                "device overlap must be finite and non-negative",
            ));
        }
        let cost = fallback_cost(
            context.request_blocks(),
            context.block_size(),
            overlap_blocks,
            load.active_prefill_tokens(),
            load.decode_cost_blocks(),
        );
        if cost.is_finite() {
            Ok(cost)
        } else {
            Err(WorkerSelectionPolicyError::failed(
                "worker cost must be finite",
            ))
        }
    }
}

pub(crate) fn fallback_cost(
    request_blocks: u64,
    block_size: u32,
    overlap_blocks: f64,
    active_prefill_tokens: usize,
    decode_cost_blocks: f64,
) -> f64 {
    let block_size = f64::from(block_size);
    (request_blocks as f64 - overlap_blocks).max(0.0) * block_size
        + active_prefill_tokens as f64
        + decode_cost_blocks * block_size
}

/// The worker that last served each session, most recently routed first.
pub(crate) struct SessionPlacements {
    entries: LruCache<String, WorkerWithDpRank>,
}

impl SessionPlacements {
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            entries: LruCache::new(capacity),
        }
    }

    /// Look up a session and mark it most recently used.
    pub(crate) fn get(&mut self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.entries.get(session_id).copied()
    }

    pub(crate) fn record(&mut self, session_id: &str, worker: WorkerWithDpRank) {
        match self.entries.get_mut(session_id) {
            Some(existing) => *existing = worker,
            None => {
                self.entries.put(session_id.to_owned(), worker);
            }
        }
    }

    pub(crate) fn forget(&mut self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.entries.pop(session_id)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Choice {
    /// The remembered worker, kept because it passed the load gate.
    Sticky(usize),
    /// The lowest-cost worker, used when there is no remembered worker, it is not
    /// eligible, or it failed the load gate.
    Fallback(usize),
}

/// Decide in one pass over `(worker, cost, active_requests)` rows.
///
/// The remembered worker wins while its in-flight request count is at most
/// `slack_requests` above the least-loaded row. Otherwise the lowest cost wins,
/// with exact ties broken by Dynamo's stable worker key.
pub(crate) fn decide<I>(
    rows: I,
    remembered: Option<WorkerWithDpRank>,
    slack_requests: usize,
) -> Result<Option<Choice>, WorkerSelectionPolicyError>
where
    I: IntoIterator<Item = (WorkerWithDpRank, f64, usize)>,
{
    let mut min_active = usize::MAX;
    let mut remembered_row: Option<(usize, usize)> = None;
    let mut best: Option<(usize, f64, WorkerWithDpRank)> = None;
    for (row, (worker, cost, active_requests)) in rows.into_iter().enumerate() {
        if !cost.is_finite() {
            return Err(WorkerSelectionPolicyError::failed(
                "worker cost must be finite",
            ));
        }
        min_active = min_active.min(active_requests);
        if remembered == Some(worker) {
            remembered_row = Some((row, active_requests));
        }
        let replace = best.is_none_or(|(_, best_cost, best_worker)| {
            cost.total_cmp(&best_cost).is_lt()
                || (cost.total_cmp(&best_cost).is_eq() && worker < best_worker)
        });
        if replace {
            best = Some((row, cost, worker));
        }
    }
    Ok(match (remembered_row, best) {
        (Some((row, active)), _) if active <= min_active.saturating_add(slack_requests) => {
            Some(Choice::Sticky(row))
        }
        (_, Some((row, _, _))) => Some(Choice::Fallback(row)),
        (_, None) => None,
    })
}

/// Keeps a session on the worker that served its previous turn while that worker
/// is within `slack_requests` of the least-loaded eligible worker.
pub(crate) struct SessionAffinityPicker {
    slack_requests: usize,
    placements: SessionPlacements,
}

impl SessionAffinityPicker {
    pub(crate) fn new(config: SessionAffinityConfig) -> Self {
        Self {
            slack_requests: config.slack_requests,
            placements: SessionPlacements::new(config.max_sessions()),
        }
    }
}

impl WorkerPicker for SessionAffinityPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if loads.len() != candidates.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "load input does not match the candidate table",
            ));
        }

        let session = context.session_context();
        let remembered = session.and_then(|session| self.placements.get(session.session_id()));
        let rows = candidates.iter().zip(loads).map(|(candidate, load)| {
            (candidate.worker(), candidate.cost(), load.active_requests())
        });
        let row = match decide(rows, remembered, self.slack_requests)? {
            Some(Choice::Sticky(row) | Choice::Fallback(row)) => row,
            None => return Err(WorkerSelectionPolicyError::failed("no eligible worker")),
        };

        if let Some(session) = session {
            if session.session_final() == Some(true) {
                self.placements.forget(session.session_id());
            } else {
                self.placements
                    .record(session.session_id(), candidates[row].worker());
            }
        }
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    fn rows(
        table: &[(u64, f64, usize)],
    ) -> impl Iterator<Item = (WorkerWithDpRank, f64, usize)> + '_ {
        table
            .iter()
            .map(|&(id, cost, active)| (worker(id), cost, active))
    }

    #[test]
    fn no_memory_selects_lowest_cost_with_stable_tie_break() {
        let table = [(9, 5.0, 0), (1, 5.0, 0), (2, 7.0, 0)];
        assert_eq!(
            decide(rows(&table), None, 4).unwrap(),
            Some(Choice::Fallback(1))
        );
    }

    #[test]
    fn remembered_worker_wins_within_slack_even_when_costlier() {
        let table = [(1, 100.0, 4), (2, 1.0, 0)];
        assert_eq!(
            decide(rows(&table), Some(worker(1)), 4).unwrap(),
            Some(Choice::Sticky(0))
        );
    }

    #[test]
    fn remembered_worker_loses_when_gate_closes() {
        let table = [(1, 100.0, 5), (2, 1.0, 0)];
        assert_eq!(
            decide(rows(&table), Some(worker(1)), 4).unwrap(),
            Some(Choice::Fallback(1))
        );
    }

    #[test]
    fn zero_slack_keeps_session_only_on_a_least_loaded_worker() {
        let table = [(1, 100.0, 3), (2, 1.0, 3), (3, 1.0, 4)];
        assert_eq!(
            decide(rows(&table), Some(worker(1)), 0).unwrap(),
            Some(Choice::Sticky(0))
        );
        let table = [(1, 100.0, 4), (2, 1.0, 3)];
        assert_eq!(
            decide(rows(&table), Some(worker(1)), 0).unwrap(),
            Some(Choice::Fallback(1))
        );
    }

    #[test]
    fn remembered_worker_missing_from_candidates_falls_back() {
        let table = [(2, 3.0, 0), (3, 2.0, 0)];
        assert_eq!(
            decide(rows(&table), Some(worker(1)), 4).unwrap(),
            Some(Choice::Fallback(1))
        );
    }

    #[test]
    fn empty_candidate_table_yields_no_choice() {
        assert_eq!(decide(rows(&[]), Some(worker(1)), 4).unwrap(), None);
    }

    #[test]
    fn non_finite_cost_is_an_error() {
        let table = [(1, f64::NAN, 0)];
        assert!(decide(rows(&table), None, 4).is_err());
    }

    #[test]
    fn placements_record_update_forget_and_evict_least_recent() {
        let mut placements = SessionPlacements::new(NonZeroUsize::new(2).unwrap());
        placements.record("a", worker(1));
        placements.record("b", worker(2));
        assert_eq!(placements.get("a"), Some(worker(1)));

        placements.record("a", worker(3));
        assert_eq!(placements.get("a"), Some(worker(3)));
        assert_eq!(placements.len(), 2);

        // "a" was touched most recently, so "b" is evicted when "c" arrives.
        placements.record("c", worker(4));
        assert_eq!(placements.get("b"), None);
        assert_eq!(placements.get("a"), Some(worker(3)));

        assert_eq!(placements.forget("a"), Some(worker(3)));
        assert_eq!(placements.get("a"), None);
        assert_eq!(placements.forget("a"), None);
    }

    #[test]
    fn fallback_cost_counts_missing_prefill_and_in_flight_work() {
        // 10 request blocks, 6 resident, block size 16: 64 missing prefill tokens,
        // plus 100 in-flight prefill tokens and 2 decode blocks (32 tokens).
        assert_eq!(fallback_cost(10, 16, 6.0, 100, 2.0), 64.0 + 100.0 + 32.0);
        // Overlap beyond the request never produces a negative prefill term.
        assert_eq!(fallback_cost(4, 16, 9.0, 0, 0.0), 0.0);
    }
}
