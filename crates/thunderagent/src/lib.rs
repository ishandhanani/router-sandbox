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

use dynamo_kv_router::scheduling::{
    RequestClassifierContext, RequestClassifierFactory, RequestClassifierParameters,
    RequestClassifierProviderError, RequestClassifierRegistry, RequestClassifierRegistryError,
};
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{WorkerSelectionPolicy, WorkerType};
use selection::{ThunderAgentPicker, ThunderAgentScorer};

pub const THUNDERAGENT_CLASSIFIER_TYPE: &str = "thunderagent";

fn capacity_provider(context: RequestClassifierContext) -> Arc<dyn WorkerCapacityProvider> {
    let block_size = u64::from(context.block_size());
    Arc::new(move || {
        let workers = context.workers();
        let live_workers = workers.iter().map(|worker| worker.worker());
        let capacities = workers.iter().filter_map(|worker| {
            let total_kv_blocks = worker.total_kv_blocks()?;
            let capacity = total_kv_blocks.saturating_mul(block_size);
            Some((
                worker.worker(),
                usize::try_from(capacity).unwrap_or(usize::MAX),
            ))
        });
        Arc::new(WorkerCapacitySnapshot::new(capacities).with_live_workers(live_workers))
    })
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let config: ThunderAgentConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| RequestClassifierProviderError::new(error.to_string()))?;

    Ok(Arc::new(move |context| {
        let capacity_provider = capacity_provider(context);
        Box::new(
            ThunderAgentClassifier::new(config.clone(), capacity_provider)
                .expect("ThunderAgent configuration was validated during catalog resolution"),
        )
    }))
}

fn worker_selection_provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: ThunderAgentConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| WorkerSelectionPolicyProviderError::new(error.to_string()))?;
    Ok(Arc::new(
        move |router, worker_type: WorkerType, _partition| {
            WorkerSelectionPolicy::new(
                router.clone(),
                worker_type.as_str(),
                vec![Box::new(ThunderAgentScorer)],
                Box::new(ThunderAgentPicker),
            )
        },
    ))
}

/// Register ThunderAgent's stateless worker-target selector.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(
        THUNDERAGENT_CLASSIFIER_TYPE,
        Arc::new(worker_selection_provider),
    )
}

/// Register ThunderAgent as a statically linked request-classifier plugin.
pub fn register_request_classifiers(
    registry: &mut RequestClassifierRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register(THUNDERAGENT_CLASSIFIER_TYPE, Arc::new(classifier_provider))
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_kv_router::scheduling::RequestClassifierWorker;

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

    #[test]
    fn derives_capacity_and_liveness_from_the_host_context() {
        let worker_with_capacity = WorkerWithDpRank::new(1, 0);
        let worker_without_capacity = WorkerWithDpRank::new(2, 0);
        let context = RequestClassifierContext::new(16, move || {
            vec![
                RequestClassifierWorker::new(worker_with_capacity, Some(10)),
                RequestClassifierWorker::new(worker_without_capacity, None),
            ]
        });

        let snapshot = capacity_provider(context).snapshot();
        assert_eq!(
            snapshot.iter().collect::<Vec<_>>(),
            [(worker_with_capacity, 160)]
        );
        assert!(snapshot.is_live(worker_with_capacity));
        assert!(snapshot.is_live(worker_without_capacity));
    }
}
