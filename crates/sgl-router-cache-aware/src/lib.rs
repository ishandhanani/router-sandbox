// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental SGL Router cache-aware policy using Dynamo KV-router inputs.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerCacheInput, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults.
const CACHE_AFFINITY_THRESHOLD: f64 = 0.5;
const IMBALANCE_DELTA_REQUESTS: usize = 32;
const IMBALANCE_RATIO: f64 = 1.1;

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

fn select_row(
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
) -> Option<usize> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }

    let min_load = load.iter().map(|item| item.active_requests()).min()?;
    let max_load = load.iter().map(|item| item.active_requests()).max()?;
    if max_load.saturating_sub(min_load) > IMBALANCE_DELTA_REQUESTS
        && (max_load as f64) > IMBALANCE_RATIO * (min_load as f64)
    {
        return least_loaded(load, 0..load.len());
    }

    let max_overlap = cache
        .iter()
        .map(|item| item.device_overlap_blocks())
        .max_by(f64::total_cmp)?;
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    if cache_ratio > CACHE_AFFINITY_THRESHOLD {
        return least_loaded(
            load,
            cache.iter().enumerate().filter_map(|(row, item)| {
                (item.device_overlap_blocks() == max_overlap).then_some(row)
            }),
        );
    }

    least_loaded(load, 0..load.len())
}

struct SglRouterCacheAwarePicker;

impl WorkerPicker for SglRouterCacheAwarePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        select_row(cache, load, context.request_blocks())
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn provider(
    _parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SglRouterCacheAwarePicker),
            )
        },
    ))
}

/// Register the `sgl-router-cache-aware` worker-selection policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("sgl-router-cache-aware", Arc::new(provider))
}
