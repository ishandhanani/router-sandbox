// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicyError,
};

use crate::StickyUntilSaturatedConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrimaryLoad {
    ProjectedPrefill,
    ActiveRequests,
}

impl PrimaryLoad {
    fn requires_prefill_tracking(self) -> bool {
        self == Self::ProjectedPrefill
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateMetrics {
    device_overlap_blocks: f64,
    active_prefill_tokens: usize,
    active_requests: usize,
}

impl CandidateMetrics {
    fn from_candidate(candidate: &WorkerCandidate) -> Result<Self, WorkerSelectionPolicyError> {
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let device_overlap_blocks = cache.device_overlap_blocks();
        if !device_overlap_blocks.is_finite() || device_overlap_blocks < 0.0 {
            return Err(WorkerSelectionPolicyError::failed(
                "device overlap must be finite and non-negative",
            ));
        }
        Ok(Self {
            device_overlap_blocks,
            active_prefill_tokens: load.active_prefill_tokens(),
            active_requests: load.active_requests(),
        })
    }

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
            active_prefill_tokens: load.active_prefill_tokens(),
            active_requests: load.active_requests(),
        })
    }
}

/// Logs the primary, lower-is-better signal while the picker applies cache affinity globally.
pub(crate) struct StickyUntilSaturatedScorer {
    primary_load: PrimaryLoad,
}

impl StickyUntilSaturatedScorer {
    pub(crate) fn new(primary_load: PrimaryLoad) -> Self {
        Self { primary_load }
    }
}

impl WorkerScorer for StickyUntilSaturatedScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        require_prefill_tracking(context, self.primary_load)?;
        let metrics = CandidateMetrics::from_candidate(candidate)?;
        primary_cost(context, metrics, self.primary_load)
    }
}

/// Chooses a warm worker until a cold worker clears the token-based saturation margin.
pub(crate) struct StickyUntilSaturatedPicker {
    config: StickyUntilSaturatedConfig,
    primary_load: PrimaryLoad,
}

impl StickyUntilSaturatedPicker {
    pub(crate) fn new(config: StickyUntilSaturatedConfig, primary_load: PrimaryLoad) -> Self {
        Self {
            config,
            primary_load,
        }
    }
}

impl WorkerPicker for StickyUntilSaturatedPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        require_prefill_tracking(context, self.primary_load)?;
        let candidates = input.candidates();
        let mut best_warm_prefill: Option<f64> = None;
        for row in 0..candidates.len() {
            let metrics = CandidateMetrics::from_picker_input(input, row)?;
            if is_warm(context, metrics, self.config.affinity_threshold) {
                let projected_prefill = projected_prefill_tokens(context, metrics);
                if best_warm_prefill.is_none_or(|best| projected_prefill < best) {
                    best_warm_prefill = Some(projected_prefill);
                }
            }
        }

        let mut selected: Option<(usize, f64)> = None;
        for (row, candidate) in candidates.iter().enumerate() {
            let metrics = CandidateMetrics::from_picker_input(input, row)?;
            let warm = is_warm(context, metrics, self.config.affinity_threshold);
            let projected_prefill = projected_prefill_tokens(context, metrics);
            let eligible = best_warm_prefill.is_none_or(|best_warm| {
                warm || projected_prefill < best_warm - self.config.saturation_tokens()
            });
            if !eligible {
                continue;
            }

            let cost = primary_cost(context, metrics, self.primary_load)?;
            let replace = selected.is_none_or(|(selected_row, selected_cost)| {
                cost.total_cmp(&selected_cost).is_lt()
                    || (cost.total_cmp(&selected_cost).is_eq()
                        && candidate.worker() < candidates[selected_row].worker())
            });
            if replace {
                selected = Some((row, cost));
            }
        }

        selected
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn require_prefill_tracking(
    context: &WorkerSelectionContext<'_>,
    primary_load: PrimaryLoad,
) -> Result<(), WorkerSelectionPolicyError> {
    if primary_load.requires_prefill_tracking() && !context.tracks_prefill_tokens() {
        return Err(WorkerSelectionPolicyError::failed(
            "sticky-until-saturated requires prefill-token tracking",
        ));
    }
    Ok(())
}

fn is_warm(
    context: &WorkerSelectionContext<'_>,
    metrics: CandidateMetrics,
    affinity_threshold: f64,
) -> bool {
    let request_blocks = context.request_blocks();
    request_blocks > 0
        && metrics.device_overlap_blocks / request_blocks as f64 >= affinity_threshold
}

fn projected_prefill_tokens(
    context: &WorkerSelectionContext<'_>,
    metrics: CandidateMetrics,
) -> f64 {
    metrics.active_prefill_tokens as f64
        + (context.request_blocks() as f64 - metrics.device_overlap_blocks).max(0.0)
            * context.block_size() as f64
}

fn primary_cost(
    context: &WorkerSelectionContext<'_>,
    metrics: CandidateMetrics,
    primary_load: PrimaryLoad,
) -> Result<f64, WorkerSelectionPolicyError> {
    let cost = match primary_load {
        PrimaryLoad::ProjectedPrefill => projected_prefill_tokens(context, metrics),
        PrimaryLoad::ActiveRequests => metrics.active_requests as f64,
    };
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(WorkerSelectionPolicyError::failed(
            "worker primary load must be finite",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestCandidate {
        key: u8,
        metrics: CandidateMetrics,
    }

    fn test_config(saturation_tokens: f64) -> StickyUntilSaturatedConfig {
        StickyUntilSaturatedConfig {
            affinity_threshold: 0.8,
            peak_prefill_tokens_per_second: saturation_tokens,
            max_ttft_penalty_ms: 1_000,
        }
    }

    fn choose(
        config: &StickyUntilSaturatedConfig,
        primary_load: PrimaryLoad,
        request_blocks: u64,
        block_size: u32,
        tracks_prefill_tokens: bool,
        candidates: &[TestCandidate],
    ) -> Result<usize, &'static str> {
        if primary_load.requires_prefill_tracking() && !tracks_prefill_tokens {
            return Err("prefill tracking disabled");
        }
        let is_warm = |metrics: CandidateMetrics| {
            request_blocks > 0
                && metrics.device_overlap_blocks / request_blocks as f64
                    >= config.affinity_threshold
        };
        let projected_prefill = |metrics: CandidateMetrics| {
            metrics.active_prefill_tokens as f64
                + (request_blocks as f64 - metrics.device_overlap_blocks).max(0.0)
                    * block_size as f64
        };
        let best_warm_prefill = candidates
            .iter()
            .filter(|candidate| is_warm(candidate.metrics))
            .map(|candidate| projected_prefill(candidate.metrics))
            .min_by(f64::total_cmp);
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                best_warm_prefill.is_none_or(|best_warm| {
                    is_warm(candidate.metrics)
                        || projected_prefill(candidate.metrics)
                            < best_warm - config.saturation_tokens()
                })
            })
            .min_by(|(_, left), (_, right)| {
                let left_cost = match primary_load {
                    PrimaryLoad::ProjectedPrefill => projected_prefill(left.metrics),
                    PrimaryLoad::ActiveRequests => left.metrics.active_requests as f64,
                };
                let right_cost = match primary_load {
                    PrimaryLoad::ProjectedPrefill => projected_prefill(right.metrics),
                    PrimaryLoad::ActiveRequests => right.metrics.active_requests as f64,
                };
                left_cost
                    .total_cmp(&right_cost)
                    .then_with(|| left.key.cmp(&right.key))
            })
            .map(|(row, _)| row)
            .ok_or("no eligible worker")
    }

    fn candidate(
        key: u8,
        device_overlap_blocks: f64,
        active_prefill_tokens: usize,
        active_requests: usize,
    ) -> TestCandidate {
        TestCandidate {
            key,
            metrics: CandidateMetrics {
                device_overlap_blocks,
                active_prefill_tokens,
                active_requests,
            },
        }
    }

    #[test]
    fn cold_start_selects_from_all_workers() {
        let candidates = [candidate(9, 1.0, 30, 5), candidate(1, 2.0, 0, 1)];
        assert_eq!(
            choose(
                &test_config(10.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(1)
        );
    }

    #[test]
    fn warm_worker_is_sticky_when_cold_worker_does_not_clear_margin() {
        let candidates = [candidate(1, 8.0, 12, 1), candidate(2, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(5.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(0)
        );
    }

    #[test]
    fn saturation_margin_is_strict() {
        let candidates = [candidate(1, 8.0, 12, 1), candidate(2, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(4.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(0)
        );
    }

    #[test]
    fn cold_worker_rejoins_after_crossing_saturation_margin() {
        let candidates = [candidate(1, 8.0, 20, 1), candidate(2, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(2.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(1)
        );
    }

    #[test]
    fn only_cold_workers_below_best_warm_margin_rejoin() {
        let candidates = [
            candidate(1, 8.0, 20, 4),
            candidate(2, 0.0, 0, 0),
            candidate(3, 0.0, 18, 0),
        ];
        assert_eq!(
            choose(
                &test_config(2.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(1)
        );
    }

    #[test]
    fn affinity_threshold_is_inclusive() {
        let candidates = [candidate(1, 8.0, 0, 1), candidate(2, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(100.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(0)
        );
    }

    #[test]
    fn ties_use_worker_identity_not_candidate_order() {
        let candidates = [candidate(9, 0.0, 0, 0), candidate(1, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(1.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                true,
                &candidates
            ),
            Ok(1)
        );
    }

    #[test]
    fn decode_pool_ranks_eligible_workers_by_active_requests() {
        let candidates = [candidate(1, 0.0, 0, 3), candidate(2, 0.0, 100, 1)];
        assert_eq!(
            choose(
                &test_config(1_000.0),
                PrimaryLoad::ActiveRequests,
                10,
                1,
                false,
                &candidates
            ),
            Ok(1)
        );
    }

    #[test]
    fn prefill_pool_requires_tracking() {
        let candidates = [candidate(1, 0.0, 0, 0)];
        assert_eq!(
            choose(
                &test_config(1.0),
                PrimaryLoad::ProjectedPrefill,
                10,
                1,
                false,
                &candidates
            ),
            Err("prefill tracking disabled")
        );
    }
}
