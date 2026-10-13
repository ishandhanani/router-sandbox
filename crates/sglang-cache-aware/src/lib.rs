// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang's cache-aware, load-gated selection policy using Dynamo worker signals.

mod config;
mod kv_fetch;
mod selection;

pub use config::{ConfigError, SglangCacheAwareConfig};

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::experimental::kv_hints::with_kv_transfer_policy;
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy, WorkerType};
use kv_fetch::MinKvFetch;
use selection::SglangCacheAwarePicker;

/// Build one SGLang cache-aware policy for a Dynamo routing partition.
pub fn worker_selection_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: SglangCacheAwareConfig,
) -> Result<WorkerSelectionPolicy, ConfigError> {
    config.validate()?;
    Ok(validated_policy(kv_router_config, worker_type, config))
}

fn validated_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: SglangCacheAwareConfig,
) -> WorkerSelectionPolicy {
    let kv_fetch_min_blocks = config.kv_fetch_min_blocks;
    let policy = WorkerSelectionPolicy::new(
        kv_router_config,
        worker_type.as_str(),
        Vec::new(),
        Box::new(SglangCacheAwarePicker::new(config)),
    );
    match kv_fetch_min_blocks {
        Some(min_blocks) => with_kv_transfer_policy(policy, Box::new(MinKvFetch { min_blocks })),
        None => policy,
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: SglangCacheAwareConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| WorkerSelectionPolicyProviderError::new(error.to_string()))?;
    Ok(Arc::new(move |router, worker_type, _partition| {
        validated_policy(router.clone(), worker_type, config.clone())
    }))
}

/// Register the `sglang-cache-aware` worker-selection policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("sglang-cache-aware", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_once() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();
        assert!(matches!(
            register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "sglang-cache-aware"
        ));
    }
}
