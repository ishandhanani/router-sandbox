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

    fn uses_cache_affinity(self) -> bool {
        self == Self::ProjectedPrefill
    }

    fn required_worker_inputs(self) -> WorkerInputs {
        if self.uses_cache_affinity() {
            WorkerInputs::LOAD | WorkerInputs::CACHE
        } else {
            WorkerInputs::LOAD
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateMetrics {
    device_overlap_blocks: Option<f64>,
    active_prefill_tokens: usize,
    active_requests: usize,
}

impl CandidateMetrics {
    fn from_candidate(
        candidate: &WorkerCandidate,
        uses_cache_affinity: bool,
    ) -> Result<Self, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let device_overlap_blocks = if uses_cache_affinity {
            Some(cache_overlap(
                candidate
                    .cache()
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?
                    .device_overlap_blocks(),
            )?)
        } else {
            None
        };
        Ok(Self {
            device_overlap_blocks,
            active_prefill_tokens: load.active_prefill_tokens(),
            active_requests: load.active_requests(),
        })
    }

    fn from_picker_input(
        input: WorkerInputView<'_>,
        row: usize,
        uses_cache_affinity: bool,
    ) -> Result<Self, WorkerSelectionPolicyError> {
        let load = input
            .load()
            .and_then(|inputs| inputs.get(row))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let device_overlap_blocks = if uses_cache_affinity {
            Some(cache_overlap(
                input
                    .cache()
                    .and_then(|inputs| inputs.get(row))
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?
                    .device_overlap_blocks(),
            )?)
        } else {
            None
        };
        Ok(Self {
            device_overlap_blocks,
            active_prefill_tokens: load.active_prefill_tokens(),
            active_requests: load.active_requests(),
        })
    }
}

fn cache_overlap(device_overlap_blocks: f64) -> Result<f64, WorkerSelectionPolicyError> {
    if !device_overlap_blocks.is_finite() || device_overlap_blocks < 0.0 {
        return Err(WorkerSelectionPolicyError::failed(
            "device overlap must be finite and non-negative",
        ));
    }
    Ok(device_overlap_blocks)
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
        self.primary_load.required_worker_inputs()
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        require_prefill_tracking(context, self.primary_load)?;
        let metrics =
            CandidateMetrics::from_candidate(candidate, self.primary_load.uses_cache_affinity())?;
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
        self.primary_load.required_worker_inputs()
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        require_prefill_tracking(context, self.primary_load)?;
        let candidates = input.candidates();
        let keep_sticky_set = if self.primary_load.uses_cache_affinity() {
            sticky_set_is_eligible(context, input, &self.config)?
        } else {
            false
        };

        let mut selected: Option<(usize, f64)> = None;
        for (row, candidate) in candidates.iter().enumerate() {
            let metrics = CandidateMetrics::from_picker_input(
                input,
                row,
                self.primary_load.uses_cache_affinity(),
            )?;
            let warm = is_warm(context, metrics, self.config.affinity_threshold);
            if keep_sticky_set && !warm {
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

/// Implements LLM-D's affinity filter before the token-load scorer runs:
/// keep the warm set unless the *best* cold worker's current in-flight load
/// beats the best warm worker by more than the configured TTFT margin. When
/// the gate opens, LLM-D restores the entire candidate set, not only that
/// particular cold worker.
fn sticky_set_is_eligible(
    context: &WorkerSelectionContext<'_>,
    input: WorkerInputView<'_>,
    config: &StickyUntilSaturatedConfig,
) -> Result<bool, WorkerSelectionPolicyError> {
    let mut best_warm_in_flight: Option<f64> = None;
    let mut best_cold_in_flight: Option<f64> = None;
    for row in 0..input.candidates().len() {
        let metrics = CandidateMetrics::from_picker_input(input, row, true)?;
        let in_flight = metrics.active_prefill_tokens as f64;
        let best = if is_warm(context, metrics, config.affinity_threshold) {
            &mut best_warm_in_flight
        } else {
            &mut best_cold_in_flight
        };
        if best.is_none_or(|current| in_flight < current) {
            *best = Some(in_flight);
        }
    }

    let Some(best_warm) = best_warm_in_flight else {
        return Ok(false);
    };
    if config.max_ttft_penalty_ms == 0 {
        return Ok(true);
    }
    let Some(best_cold) = best_cold_in_flight else {
        return Ok(true);
    };
    Ok(best_warm - best_cold <= config.saturation_tokens())
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
        && metrics
            .device_overlap_blocks
            .is_some_and(|overlap| overlap / request_blocks as f64 >= affinity_threshold)
}

fn projected_prefill_tokens(
    context: &WorkerSelectionContext<'_>,
    metrics: CandidateMetrics,
) -> f64 {
    metrics.active_prefill_tokens as f64
        + (context.request_blocks() as f64 - metrics.device_overlap_blocks.unwrap_or(0.0)).max(0.0)
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

/// Input row for the policy-only Criterion benchmark.
///
/// This is intentionally separate from Dynamo's host-owned candidate table. It exercises the
/// picker's two scans and arithmetic without benchmarking discovery, eligibility, or transport.
#[cfg(feature = "bench")]
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct BenchmarkCandidate {
    pub worker_key: u64,
    pub device_overlap_blocks: f64,
    pub active_prefill_tokens: usize,
}

/// Run the prefill decision over synthetic, already host-eligible candidate rows.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub fn benchmark_prefill_pick_index(
    config: &StickyUntilSaturatedConfig,
    request_blocks: u64,
    block_size: u32,
    candidates: &[BenchmarkCandidate],
) -> Option<usize> {
    let is_warm = |candidate: &BenchmarkCandidate| {
        request_blocks > 0
            && candidate.device_overlap_blocks / request_blocks as f64 >= config.affinity_threshold
    };
    let projected_prefill = |candidate: &BenchmarkCandidate| {
        candidate.active_prefill_tokens as f64
            + (request_blocks as f64 - candidate.device_overlap_blocks).max(0.0) * block_size as f64
    };
    let best_warm_in_flight = candidates
        .iter()
        .filter(|candidate| is_warm(candidate))
        .map(|candidate| candidate.active_prefill_tokens as f64)
        .min_by(f64::total_cmp);
    let best_cold_in_flight = candidates
        .iter()
        .filter(|candidate| !is_warm(candidate))
        .map(|candidate| candidate.active_prefill_tokens as f64)
        .min_by(f64::total_cmp);
    let keep_sticky_set = match best_warm_in_flight {
        None => false,
        Some(_) if config.max_ttft_penalty_ms == 0 => true,
        Some(best_warm) => best_cold_in_flight
            .is_none_or(|best_cold| best_warm - best_cold <= config.saturation_tokens()),
    };

    candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !keep_sticky_set || is_warm(candidate))
        .min_by(|(_, left), (_, right)| {
            projected_prefill(left)
                .total_cmp(&projected_prefill(right))
                .then_with(|| left.worker_key.cmp(&right.worker_key))
        })
        .map(|(row, _)| row)
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
                && metrics.device_overlap_blocks.is_some_and(|overlap| {
                    overlap / request_blocks as f64 >= config.affinity_threshold
                })
        };
        let projected_prefill = |metrics: CandidateMetrics| {
            metrics.active_prefill_tokens as f64
                + (request_blocks as f64 - metrics.device_overlap_blocks.unwrap_or(0.0)).max(0.0)
                    * block_size as f64
        };
        let keep_sticky_set = if primary_load.uses_cache_affinity() {
            let best_warm_in_flight = candidates
                .iter()
                .filter(|candidate| is_warm(candidate.metrics))
                .map(|candidate| candidate.metrics.active_prefill_tokens as f64)
                .min_by(f64::total_cmp);
            let best_cold_in_flight = candidates
                .iter()
                .filter(|candidate| !is_warm(candidate.metrics))
                .map(|candidate| candidate.metrics.active_prefill_tokens as f64)
                .min_by(f64::total_cmp);
            match best_warm_in_flight {
                None => false,
                Some(_) if config.max_ttft_penalty_ms == 0 => true,
                Some(best_warm) => best_cold_in_flight
                    .is_none_or(|best_cold| best_warm - best_cold <= config.saturation_tokens()),
            }
        } else {
            false
        };
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| !keep_sticky_set || is_warm(candidate.metrics))
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
                device_overlap_blocks: Some(device_overlap_blocks),
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
    fn gate_keeps_warm_workers_when_ttft_penalty_is_not_exceeded() {
        let candidates = [candidate(1, 8.0, 5, 1), candidate(2, 0.0, 0, 0)];
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
        let candidates = [candidate(1, 8.0, 5, 1), candidate(2, 0.0, 0, 0)];
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
    fn load_escape_restores_full_candidate_set_before_scoring() {
        let candidates = [
            candidate(1, 1_600.0, 1_000, 4),
            candidate(2, 0.0, 0, 0),
            candidate(3, 1_599.0, 950, 0),
        ];
        assert_eq!(
            choose(
                &test_config(100.0),
                PrimaryLoad::ProjectedPrefill,
                2_000,
                1,
                true,
                &candidates
            ),
            Ok(2)
        );
    }

    #[test]
    fn zero_ttft_penalty_always_keeps_the_warm_set() {
        let candidates = [candidate(1, 8.0, 100, 1), candidate(2, 0.0, 0, 0)];
        let config = StickyUntilSaturatedConfig {
            max_ttft_penalty_ms: 0,
            ..test_config(1.0)
        };
        assert_eq!(
            choose(
                &config,
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
