// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-aware cache-affinity routing for Dynamo.

mod config;
mod selection;

pub use config::{ConfigError, StickyUntilSaturatedConfig};

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy, WorkerType};
use selection::{PrimaryLoad, StickyUntilSaturatedPicker, StickyUntilSaturatedScorer};

/// Build one sticky-until-saturated policy for a Dynamo routing partition.
pub fn worker_selection_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: StickyUntilSaturatedConfig,
) -> Result<WorkerSelectionPolicy, ConfigError> {
    config.validate()?;
    Ok(validated_policy(kv_router_config, worker_type, config))
}

fn validated_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: StickyUntilSaturatedConfig,
) -> WorkerSelectionPolicy {
    let primary_load = match worker_type {
        WorkerType::Prefill | WorkerType::Aggregated => PrimaryLoad::ProjectedPrefill,
        WorkerType::Decode | WorkerType::Encode => PrimaryLoad::ActiveRequests,
    };
    WorkerSelectionPolicy::new(
        kv_router_config,
        worker_type.as_str(),
        vec![Box::new(StickyUntilSaturatedScorer::new(primary_load))],
        Box::new(StickyUntilSaturatedPicker::new(config, primary_load)),
    )
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: StickyUntilSaturatedConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| WorkerSelectionPolicyProviderError::new(error.to_string()))?;
    Ok(Arc::new(move |router, worker_type, _partition| {
        validated_policy(router.clone(), worker_type, config.clone())
    }))
}

/// Register the `sticky-until-saturated` worker-selection policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("sticky-until-saturated", Arc::new(provider))
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
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "sticky-until-saturated"
        ));
    }
}
