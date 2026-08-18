// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

use crate::SglangCacheAwareConfig;

#[derive(Clone, Copy, Debug)]
struct CandidateMetrics {
    device_overlap_blocks: f64,
    active_requests: usize,
}

impl CandidateMetrics {
    fn from_picker_input(
        input: WorkerInputView<'_>,
        row: usize,
    ) -> Result<Self, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .and_then(|inputs| inputs.get(row))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .and_then(|inputs| inputs.get(row))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let device_overlap_blocks = cache.device_overlap_blocks();
        if !device_overlap_blocks.is_finite() || device_overlap_blocks < 0.0 {
            return Err(WorkerSelectionPolicyError::failed(
                "device overlap must be finite and non-negative",
            ));
        }
        Ok(Self {
            device_overlap_blocks,
            active_requests: load.active_requests(),
        })
    }
}

/// Reproduces SGLang's global load gate before considering cache affinity.
///
/// Dynamo supplies exact device-KV overlap instead of SGLang's approximate text tree.
pub(crate) struct SglangCacheAwarePicker {
    config: SglangCacheAwareConfig,
}

impl SglangCacheAwarePicker {
    pub(crate) fn new(config: SglangCacheAwareConfig) -> Self {
        Self { config }
    }
}

impl WorkerPicker for SglangCacheAwarePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let mut minimum_load = usize::MAX;
        let mut maximum_load = 0;
        for row in 0..candidates.len() {
            let metrics = CandidateMetrics::from_picker_input(input, row)?;
            minimum_load = minimum_load.min(metrics.active_requests);
            maximum_load = maximum_load.max(metrics.active_requests);
        }
        if minimum_load == usize::MAX {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        if is_imbalanced(minimum_load, maximum_load, &self.config) {
            return lowest_load_row(input, minimum_load);
        }

        let request_blocks = context.request_blocks();
        let mut best_cache_match: Option<(usize, f64)> = None;
        for (row, candidate) in candidates.iter().enumerate() {
            let metrics = CandidateMetrics::from_picker_input(input, row)?;
            let cache_fraction = if request_blocks > 0 {
                metrics.device_overlap_blocks / request_blocks as f64
            } else {
                0.0
            };
            if cache_fraction <= self.config.cache_threshold {
                continue;
            }
            let replace = best_cache_match.is_none_or(|(best_row, best_overlap)| {
                metrics
                    .device_overlap_blocks
                    .total_cmp(&best_overlap)
                    .is_gt()
                    || (metrics
                        .device_overlap_blocks
                        .total_cmp(&best_overlap)
                        .is_eq()
                        && candidate.worker() < candidates[best_row].worker())
            });
            if replace {
                best_cache_match = Some((row, metrics.device_overlap_blocks));
            }
        }

        best_cache_match
            .map(|(row, _)| row)
            .map(Ok)
            .unwrap_or_else(|| lowest_load_row(input, minimum_load))
    }
}

fn is_imbalanced(
    minimum_load: usize,
    maximum_load: usize,
    config: &SglangCacheAwareConfig,
) -> bool {
    maximum_load.saturating_sub(minimum_load) > config.balance_abs_threshold
        && maximum_load as f64 > minimum_load as f64 * config.balance_rel_threshold
}

fn lowest_load_row(
    input: WorkerInputView<'_>,
    minimum_load: usize,
) -> Result<usize, WorkerSelectionPolicyError> {
    let mut selected: Option<usize> = None;
    for (row, candidate) in input.candidates().iter().enumerate() {
        let metrics = CandidateMetrics::from_picker_input(input, row)?;
        if metrics.active_requests != minimum_load {
            continue;
        }
        if selected
            .is_none_or(|best_row| candidate.worker() < input.candidates()[best_row].worker())
        {
            selected = Some(row);
        }
    }
    selected.ok_or_else(|| WorkerSelectionPolicyError::failed("no least-loaded worker"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestCandidate {
        key: u8,
        device_overlap_blocks: f64,
        active_requests: usize,
    }

    fn candidate(key: u8, device_overlap_blocks: f64, active_requests: usize) -> TestCandidate {
        TestCandidate {
            key,
            device_overlap_blocks,
            active_requests,
        }
    }

    fn choose(
        config: &SglangCacheAwareConfig,
        request_blocks: u64,
        candidates: &[TestCandidate],
    ) -> Option<usize> {
        let minimum_load = candidates
            .iter()
            .map(|candidate| candidate.active_requests)
            .min()?;
        let maximum_load = candidates
            .iter()
            .map(|candidate| candidate.active_requests)
            .max()?;
        if is_imbalanced(minimum_load, maximum_load, config) {
            return candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.active_requests == minimum_load)
                .min_by_key(|(_, candidate)| candidate.key)
                .map(|(row, _)| row);
        }
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                request_blocks > 0
                    && candidate.device_overlap_blocks / request_blocks as f64
                        > config.cache_threshold
            })
            .max_by(|(_, left), (_, right)| {
                left.device_overlap_blocks
                    .total_cmp(&right.device_overlap_blocks)
                    .then_with(|| right.key.cmp(&left.key))
            })
            .or_else(|| {
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| candidate.active_requests == minimum_load)
                    .min_by_key(|(_, candidate)| candidate.key)
            })
            .map(|(row, _)| row)
    }

    fn config(
        cache_threshold: f64,
        balance_abs_threshold: usize,
        balance_rel_threshold: f64,
    ) -> SglangCacheAwareConfig {
        SglangCacheAwareConfig {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
        }
    }

    #[test]
    fn balanced_pool_prefers_the_largest_qualifying_device_overlap() {
        let candidates = [candidate(1, 8.0, 9), candidate(2, 6.0, 0)];
        assert_eq!(choose(&config(0.5, 16, 2.0), 10, &candidates), Some(0));
    }

    #[test]
    fn cache_threshold_is_strict() {
        let candidates = [candidate(1, 5.0, 9), candidate(2, 0.0, 0)];
        assert_eq!(choose(&config(0.5, 16, 2.0), 10, &candidates), Some(1));
    }

    #[test]
    fn cache_miss_chooses_the_least_loaded_worker() {
        let candidates = [candidate(1, 2.0, 4), candidate(2, 0.0, 1)];
        assert_eq!(choose(&config(0.5, 16, 2.0), 10, &candidates), Some(1));
    }

    #[test]
    fn imbalance_requires_both_absolute_and_relative_conditions() {
        let candidates = [candidate(1, 8.0, 6), candidate(2, 0.0, 1)];
        assert_eq!(choose(&config(0.5, 4, 10.0), 10, &candidates), Some(0));
        assert_eq!(choose(&config(0.5, 8, 1.0), 10, &candidates), Some(0));
    }

    #[test]
    fn imbalance_overrides_cache_affinity() {
        let candidates = [candidate(1, 10.0, 3), candidate(2, 0.0, 0)];
        assert_eq!(choose(&config(0.5, 2, 1.5), 10, &candidates), Some(1));
    }

    #[test]
    fn ties_use_worker_identity_not_candidate_order() {
        let candidates = [candidate(9, 10.0, 0), candidate(1, 10.0, 0)];
        assert_eq!(choose(&config(0.5, 1, 2.0), 10, &candidates), Some(1));

        let candidates = [candidate(9, 0.0, 0), candidate(1, 0.0, 0)];
        assert_eq!(choose(&config(0.5, 1, 2.0), 10, &candidates), Some(1));
    }

    #[test]
    fn empty_candidate_set_has_no_selection() {
        assert_eq!(choose(&SglangCacheAwareConfig::default(), 10, &[]), None);
    }
}
