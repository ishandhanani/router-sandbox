// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod request_classifier;
mod worker_selection;

pub use config::{ConfigError, ThunderAgentConfig};
pub use request_classifier::{
    ThunderAgentClassifier, WorkerCapacityProvider, WorkerCapacitySnapshot,
};

use dynamo_kv_router::scheduling::{RequestClassifierRegistry, RequestClassifierRegistryError};
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

pub const THUNDERAGENT_CLASSIFIER_TYPE: &str = "thunderagent";

/// Register ThunderAgent's stateless worker-target selector.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    worker_selection::register(registry)
}

/// Register ThunderAgent as a statically linked request-classifier plugin.
pub fn register_request_classifiers(
    registry: &mut RequestClassifierRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    request_classifier::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_classifier_and_worker_selector() {
        let mut worker_registry = WorkerSelectionPolicyRegistry::default();
        register(&mut worker_registry).unwrap();
        assert!(!worker_registry.is_empty());

        let mut classifier_registry = RequestClassifierRegistry::default();
        register_request_classifiers(&mut classifier_registry).unwrap();
        assert!(!classifier_registry.is_empty());
    }
}
