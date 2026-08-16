// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod policy;
mod selection;

pub use config::{ConfigError, ThunderAgentConfig};

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy, WorkerType};

use selection::SessionAssignments;
use selection::{ThunderAgentPicker, ThunderAgentScorer};

use policy::ThunderAgentPolicy;

/// Build one complete ThunderAgent policy for a Dynamo routing partition.
pub fn worker_selection_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: ThunderAgentConfig,
) -> Result<WorkerSelectionPolicy, ConfigError> {
    config.validate()?;
    Ok(validated_policy(
        kv_router_config,
        worker_type,
        config,
        false,
    ))
}

fn validated_policy(
    kv_router_config: KvRouterConfig,
    worker_type: WorkerType,
    config: ThunderAgentConfig,
    storage_handoff_experiment: bool,
) -> WorkerSelectionPolicy {
    let assignments = Arc::new(SessionAssignments::default());
    let admission = ThunderAgentPolicy::new(config, Arc::clone(&assignments));
    let scorer = ThunderAgentScorer;
    let picker = ThunderAgentPicker::new(assignments, storage_handoff_experiment);
    WorkerSelectionPolicy::new(
        kv_router_config,
        worker_type.as_str(),
        vec![Box::new(scorer)],
        Box::new(picker),
    )
    .with_admission_policy(Box::new(admission))
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    provider_with_storage_handoff(parameters, false)
}

fn storage_handoff_provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    provider_with_storage_handoff(parameters, true)
}

fn provider_with_storage_handoff(
    parameters: &WorkerSelectionPolicyParameters,
    storage_handoff_experiment: bool,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: ThunderAgentConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| WorkerSelectionPolicyProviderError::new(error.to_string()))?;
    Ok(Arc::new(move |router, worker_type, _partition| {
        validated_policy(
            router.clone(),
            worker_type,
            config.clone(),
            storage_handoff_experiment,
        )
    }))
}

/// Register ThunderAgent under the `thunderagent` worker-selection policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("thunderagent", Arc::new(provider))?;
    registry.register(
        "thunderagent-storage-handoff-experiment",
        Arc::new(storage_handoff_provider),
    )
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
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "thunderagent"
        ));
    }
}
