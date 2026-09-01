// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicyError,
};
use parking_lot::RwLock;

#[derive(Default)]
pub(crate) struct SessionAssignments {
    workers: RwLock<HashMap<String, WorkerWithDpRank>>,
}

impl SessionAssignments {
    pub(crate) fn set(&self, session_id: &str, worker: Option<WorkerWithDpRank>) {
        let mut workers = self.workers.write();
        match worker {
            Some(worker) => {
                workers.insert(session_id.to_owned(), worker);
            }
            None => {
                workers.remove(session_id);
            }
        }
    }

    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.workers.read().get(session_id).copied()
    }
}

pub(crate) struct ThunderAgentScorer;

impl WorkerScorer for ThunderAgentScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("worker load unavailable"))?;
        Ok(load.active_prefill_tokens() as f64
            + load.decode_cost_blocks() * f64::from(context.block_size()))
    }
}

pub(crate) struct ThunderAgentPicker {
    assignments: Arc<SessionAssignments>,
}

impl ThunderAgentPicker {
    pub(crate) fn new(assignments: Arc<SessionAssignments>) -> Self {
        Self { assignments }
    }
}

impl WorkerPicker for ThunderAgentPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if let Some(worker) = context
            .session_context()
            .and_then(|session| self.assignments.get(session.session_id()))
            && let Some(row) = candidates
                .iter()
                .position(|candidate| candidate.worker() == worker)
        {
            return Ok(row);
        }

        candidates
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.cost()
                    .total_cmp(&right.cost())
                    .then_with(|| left.worker().cmp(&right.worker()))
            })
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
    use dynamo_kv_router::{
        KvRouterConfig, SessionContext, WorkerSelectionInput, WorkerSelectionPolicy, WorkerSelector,
    };

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }
    }

    fn request(session_id: &str) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("request".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: Some(SessionContext::new(
                session_id.into(),
                None,
                Some(false),
                None,
                None,
            )),
            affinity_target: None,
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    #[test]
    fn picker_honors_the_classifier_assignment_when_eligible() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);
        let workers = HashMap::from([(1, TestWorker), (2, TestWorker)]);
        let assignments = Arc::new(SessionAssignments::default());
        assignments.set("session-a", Some(worker_2));
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            vec![Box::new(ThunderAgentScorer)],
            Box::new(ThunderAgentPicker::new(assignments)),
        );
        let mut request = request("session-a");

        let selected = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, worker_2);

        request.pinned_worker = Some(worker_1);
        let selected = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, worker_1);
    }
}
