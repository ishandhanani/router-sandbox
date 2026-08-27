// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod capacity;
mod config;
mod policy;
mod scheduler;
mod selection;

pub use capacity::{WorkerCapacityProvider, WorkerCapacitySnapshot};
pub use config::{ConfigError, ThunderAgentConfig};
pub use policy::ThunderAgentClassifier;

use std::sync::Arc;

use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy};

use selection::{SessionAssignments, ThunderAgentPicker, ThunderAgentScorer};

/// Classifier and worker-selection components sharing ThunderAgent's session assignments.
pub struct ThunderAgentComponents {
    pub classifier: ThunderAgentClassifier,
    pub worker_selection_policy: WorkerSelectionPolicy,
}

impl ThunderAgentComponents {
    pub fn new(
        kv_router_config: KvRouterConfig,
        worker_label: &'static str,
        config: ThunderAgentConfig,
        capacity_provider: Arc<dyn WorkerCapacityProvider>,
    ) -> Result<Self, ConfigError> {
        let assignments = Arc::new(SessionAssignments::default());
        let classifier = ThunderAgentClassifier::with_assignments(
            config,
            capacity_provider,
            Arc::clone(&assignments),
        )?;
        let worker_selection_policy = WorkerSelectionPolicy::new(
            kv_router_config,
            worker_label,
            vec![Box::new(ThunderAgentScorer)],
            Box::new(ThunderAgentPicker::new(assignments)),
        );
        Ok(Self {
            classifier,
            worker_selection_policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::scheduling::RequestClassifier;

    use super::*;

    #[test]
    fn constructs_classifier_and_worker_selection_together() {
        let snapshot = Arc::new(WorkerCapacitySnapshot::default());
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let components = ThunderAgentComponents::new(
            KvRouterConfig::default(),
            "generate",
            ThunderAgentConfig::default(),
            provider,
        )
        .unwrap();

        fn assert_classifier<T: RequestClassifier>(_classifier: &T) {}
        assert_classifier(&components.classifier);
        let _ = components.worker_selection_policy;
    }
}
