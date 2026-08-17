// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ThunderAgentConfig;
use crate::selection::SessionAssignments;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    QueueAdmissionDecision, QueueAdmissionEvent, QueueAdmissionId, QueueAdmissionPolicy,
    QueueAdmissionRequest, QueueAdmissionWorker, RequestProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramStatus {
    Reasoning,
    Acting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramLifecycle {
    Active,
    Paused,
}

#[derive(Clone)]
struct Program {
    status: ProgramStatus,
    lifecycle: ProgramLifecycle,
    assigned_worker: Option<WorkerWithDpRank>,
    token_total: usize,
    request_progress: Option<RequestProgress>,
    step_count: usize,
    marked_for_pause: bool,
    acting_since: Option<Instant>,
    deferred_since: Option<Instant>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            status: ProgramStatus::Reasoning,
            lifecycle: ProgramLifecycle::Active,
            assigned_worker: None,
            token_total: 0,
            request_progress: None,
            step_count: 0,
            marked_for_pause: false,
            acting_since: None,
            deferred_since: None,
        }
    }
}

struct RequestState {
    request_id: String,
    session_id: String,
    context_tokens: usize,
    progress: RequestProgress,
    eligible_workers: HashSet<WorkerWithDpRank>,
    dispatched: bool,
    prior: Option<Program>,
}

#[derive(Default)]
struct SessionRequests {
    current: Option<QueueAdmissionId>,
    waiting: VecDeque<QueueAdmissionId>,
}

#[derive(Clone, Copy)]
struct WorkerState {
    capacity_tokens: Option<usize>,
    available: bool,
}

#[derive(Default, Clone, Copy)]
struct WorkerUsage {
    used: usize,
    decayed: usize,
}

impl WorkerUsage {
    fn add_program(&mut self, normal: usize, decayed: usize, buffer: usize) {
        self.used = self.used.saturating_add(normal).saturating_add(buffer);
        self.decayed = self.decayed.saturating_add(decayed).saturating_add(buffer);
    }

    fn remove_program(&mut self, normal: usize, decayed: usize, buffer: usize) {
        self.used = self.used.saturating_sub(normal).saturating_sub(buffer);
        self.decayed = self.decayed.saturating_sub(decayed).saturating_sub(buffer);
    }
}

/// Session-aware admission state owned by one Dynamo policy queue.
pub(crate) struct ThunderAgentPolicy {
    config: ThunderAgentConfig,
    programs: HashMap<String, Program>,
    paused: HashSet<String>,
    requests: HashMap<QueueAdmissionId, RequestState>,
    request_ids: HashMap<String, QueueAdmissionId>,
    sessions: HashMap<String, SessionRequests>,
    workers: HashMap<WorkerWithDpRank, WorkerState>,
    assignments: Arc<SessionAssignments>,
    next_tick: Instant,
}

impl ThunderAgentPolicy {
    pub(crate) fn new(config: ThunderAgentConfig, assignments: Arc<SessionAssignments>) -> Self {
        let next_tick = Instant::now() + Duration::from_secs_f64(config.scheduler_interval_seconds);
        Self {
            config,
            programs: HashMap::new(),
            paused: HashSet::new(),
            requests: HashMap::new(),
            request_ids: HashMap::new(),
            sessions: HashMap::new(),
            workers: HashMap::new(),
            assignments,
            next_tick,
        }
    }

    fn admit_request(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
        let Some(session_id) = request
            .session_context()
            .map(|context| context.session_id())
        else {
            return QueueAdmissionDecision::Bypass;
        };
        if self.request_ids.contains_key(request.request_id()) {
            tracing::warn!(
                request_id = request.request_id(),
                "Duplicate active request ID"
            );
            return QueueAdmissionDecision::Bypass;
        }

        self.update_workers(request.workers());
        let id = request.id();
        let session_is_busy = self
            .sessions
            .get(session_id)
            .is_some_and(|requests| requests.current.is_some());
        self.requests.insert(
            id,
            RequestState {
                request_id: request.request_id().to_owned(),
                session_id: session_id.to_owned(),
                context_tokens: request.context_tokens(),
                progress: request.progress().clone(),
                eligible_workers: {
                    let mut eligible_workers = HashSet::new();
                    request.for_each_eligible_worker(|worker| {
                        eligible_workers.insert(worker.worker());
                    });
                    eligible_workers
                },
                dispatched: false,
                prior: None,
            },
        );
        self.request_ids.insert(request.request_id().to_owned(), id);
        if session_is_busy {
            self.sessions
                .get_mut(session_id)
                .expect("busy session exists")
                .waiting
                .push_back(id);
            return QueueAdmissionDecision::Defer;
        }

        self.begin_request(id, Instant::now())
    }

    fn begin_request(&mut self, id: QueueAdmissionId, now: Instant) -> QueueAdmissionDecision {
        let Some(request) = self.requests.get(&id) else {
            return QueueAdmissionDecision::Defer;
        };
        let session_id = request.session_id.clone();
        let context_tokens = request.context_tokens;
        let request_progress = request.progress.clone();
        let eligible_workers = request.eligible_workers.clone();
        let prior = self.programs.get(&session_id).cloned();
        let was_new = prior.is_none();
        self.requests.get_mut(&id).expect("request exists").prior = prior;
        self.sessions.entry(session_id.clone()).or_default().current = Some(id);

        if let Some(program) = self.programs.get_mut(&session_id) {
            program.step_count = program.step_count.saturating_add(1);
            if context_tokens > 0 {
                program.token_total = context_tokens;
            }
            program.request_progress = Some(request_progress);
            program.status = ProgramStatus::Reasoning;
            program.acting_since = None;
        } else {
            self.programs.insert(
                session_id.clone(),
                Program {
                    step_count: 1,
                    token_total: context_tokens,
                    request_progress: Some(request_progress),
                    ..Default::default()
                },
            );
        }

        let program = &self.programs[&session_id];
        if program.lifecycle == ProgramLifecycle::Paused {
            self.defer_request(&session_id, now, false);
            return QueueAdmissionDecision::Defer;
        }
        if let Some(worker) = program.assigned_worker
            && eligible_workers.contains(&worker)
        {
            match self.workers.get(&worker) {
                Some(worker_state) if worker_state.available => {
                    return QueueAdmissionDecision::Ready;
                }
                Some(_) => {
                    self.defer_request(&session_id, now, true);
                    return QueueAdmissionDecision::Defer;
                }
                None => self.set_assignment(&session_id, None),
            }
        } else if program.assigned_worker.is_some() {
            self.set_assignment(&session_id, None);
        }

        if was_new && !self.paused.is_empty() {
            self.defer_request(&session_id, now, false);
            return QueueAdmissionDecision::Defer;
        }

        let capacities: Vec<_> = self
            .available_capacities()
            .into_iter()
            .filter(|(worker, _)| eligible_workers.contains(worker))
            .collect();
        if capacities.is_empty() {
            return QueueAdmissionDecision::Ready;
        }
        let required = context_tokens.saturating_add(self.config.buffer_per_program);
        let usage = self.worker_usage(now);
        let selected = capacities
            .into_iter()
            .filter_map(|(worker, capacity)| {
                let used = usage.get(&worker).map_or(0, |usage| usage.used);
                capacity
                    .checked_sub(used)
                    .is_some_and(|remaining| remaining >= required)
                    .then_some((worker, used))
            })
            .min_by_key(|(worker, used)| (*used, *worker))
            .map(|(worker, _)| worker);
        match selected {
            Some(worker) => {
                self.set_assignment(&session_id, Some(worker));
                QueueAdmissionDecision::Ready
            }
            None => {
                self.defer_request(&session_id, now, false);
                QueueAdmissionDecision::Defer
            }
        }
    }

    fn defer_request(&mut self, session_id: &str, now: Instant, preserve_assignment: bool) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        program.lifecycle = ProgramLifecycle::Paused;
        program.deferred_since = Some(now);
        if !preserve_assignment {
            self.set_assignment(session_id, None);
        }
        self.paused.insert(session_id.to_owned());
    }

    fn dispatched(&mut self, id: QueueAdmissionId, worker: WorkerWithDpRank) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if self
            .sessions
            .get(&request.session_id)
            .and_then(|requests| requests.current)
            != Some(id)
        {
            return;
        }
        request.dispatched = true;
        let session_id = request.session_id.clone();
        self.set_assignment(&session_id, Some(worker));
    }

    fn finish_by_request_id(
        &mut self,
        request_id: &str,
        completed: bool,
        context_tokens: Option<usize>,
    ) -> Vec<QueueAdmissionId> {
        let Some(id) = self.request_ids.remove(request_id) else {
            return Vec::new();
        };
        self.finish_request(id, completed, context_tokens)
    }

    fn finish_request(
        &mut self,
        id: QueueAdmissionId,
        completed: bool,
        context_tokens: Option<usize>,
    ) -> Vec<QueueAdmissionId> {
        let Some(request) = self.requests.remove(&id) else {
            return Vec::new();
        };
        debug_assert_eq!(self.request_ids.remove(&request.request_id), None);
        let is_current = self
            .sessions
            .get(&request.session_id)
            .and_then(|requests| requests.current)
            == Some(id);
        if !is_current {
            if let Some(requests) = self.sessions.get_mut(&request.session_id)
                && let Some(index) = requests.waiting.iter().position(|waiting| *waiting == id)
            {
                requests.waiting.remove(index);
            }
            return Vec::new();
        }

        if let Some(requests) = self.sessions.get_mut(&request.session_id) {
            requests.current = None;
        }
        if let Some(program) = self.programs.get_mut(&request.session_id) {
            program.deferred_since = None;
        }
        if !request.dispatched || !completed {
            match request.prior {
                Some(prior) => {
                    let paused = prior.lifecycle == ProgramLifecycle::Paused;
                    let assignment = prior.assigned_worker;
                    self.programs.insert(request.session_id.clone(), prior);
                    self.assignments.set(&request.session_id, assignment);
                    if paused {
                        self.paused.insert(request.session_id.clone());
                    } else {
                        self.paused.remove(&request.session_id);
                    }
                }
                None => {
                    self.programs.remove(&request.session_id);
                    self.paused.remove(&request.session_id);
                    self.assignments.set(&request.session_id, None);
                }
            }
        } else if let Some(program) = self.programs.get_mut(&request.session_id) {
            program.token_total =
                context_tokens.unwrap_or_else(|| request.progress.context_tokens());
            program.request_progress = None;
            program.status = ProgramStatus::Acting;
            program.acting_since = Some(Instant::now());
            if std::mem::take(&mut program.marked_for_pause) {
                self.pause_acting(&request.session_id);
            }
        }

        self.promote_next(&request.session_id)
    }

    fn promote_next(&mut self, session_id: &str) -> Vec<QueueAdmissionId> {
        let next = self
            .sessions
            .get_mut(session_id)
            .and_then(|requests| requests.waiting.pop_front());
        let Some(id) = next else {
            self.sessions.remove(session_id);
            return Vec::new();
        };
        match self.begin_request(id, Instant::now()) {
            QueueAdmissionDecision::Ready => vec![id],
            _ => Vec::new(),
        }
    }

    fn reconcile(&mut self, workers: &[QueueAdmissionWorker]) -> Vec<QueueAdmissionId> {
        self.update_workers(workers);
        let now = Instant::now();
        if now < self.next_tick {
            return Vec::new();
        }
        self.next_tick = now + Duration::from_secs_f64(self.config.scheduler_interval_seconds);
        self.expire_retained_programs(now);
        let mut usage = self.worker_usage(now);
        let mut ready = self.greedy_resume(&mut usage, now);
        ready.extend(self.force_timed_out(&mut usage, now));
        self.pause_until_safe(&mut usage, now);
        ready
    }

    fn update_workers(&mut self, workers: &[QueueAdmissionWorker]) {
        self.workers.clear();
        self.workers.extend(workers.iter().map(|worker| {
            (
                worker.worker(),
                WorkerState {
                    capacity_tokens: worker.capacity_tokens(),
                    available: worker.is_available(),
                },
            )
        }));
        let removed: Vec<String> = self
            .programs
            .iter()
            .filter(|(_, program)| {
                program
                    .assigned_worker
                    .is_some_and(|worker| !self.workers.contains_key(&worker))
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();
        for session_id in removed {
            self.set_assignment(&session_id, None);
        }
    }

    fn available_capacities(&self) -> Vec<(WorkerWithDpRank, usize)> {
        self.workers
            .iter()
            .filter_map(|(&worker, state)| {
                (state.available)
                    .then_some(state.capacity_tokens)
                    .flatten()
                    .map(|capacity| (worker, capacity))
            })
            .collect()
    }

    fn set_assignment(&mut self, session_id: &str, worker: Option<WorkerWithDpRank>) {
        if let Some(program) = self.programs.get_mut(session_id) {
            program.assigned_worker = worker;
        }
        self.assignments.set(session_id, worker);
    }

    fn expire_retained_programs(&mut self, now: Instant) {
        let retention = Duration::from_secs_f64(self.config.session_retention_seconds);
        let expired: Vec<String> = self
            .programs
            .iter()
            .filter(|(session_id, program)| {
                program.status == ProgramStatus::Acting
                    && !self.sessions.contains_key(session_id.as_str())
                    && program
                        .acting_since
                        .is_some_and(|since| now.saturating_duration_since(since) >= retention)
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();
        for session_id in expired {
            self.programs.remove(&session_id);
            self.paused.remove(&session_id);
            self.assignments.set(&session_id, None);
        }
    }

    fn program_tokens(&self, program: &Program, decayed: bool, now: Instant) -> usize {
        let token_total = current_token_total(program);
        if program.status != ProgramStatus::Acting {
            return token_total;
        }
        let weight = if decayed {
            let idle = program
                .acting_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
            2.0_f64.powf(-idle.as_secs_f64() / self.config.acting_decay_tau_seconds)
        } else {
            self.config.acting_token_weight
        };
        scale_tokens(token_total, weight)
    }

    fn worker_usage(&self, now: Instant) -> HashMap<WorkerWithDpRank, WorkerUsage> {
        let mut usage = HashMap::<WorkerWithDpRank, WorkerUsage>::new();
        for program in self.programs.values() {
            if program.lifecycle == ProgramLifecycle::Active
                && let Some(worker) = program.assigned_worker
            {
                usage.entry(worker).or_default().add_program(
                    self.program_tokens(program, false, now),
                    self.program_tokens(program, true, now),
                    self.config.buffer_per_program,
                );
            }
        }
        usage
    }

    fn greedy_resume(
        &mut self,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) -> Vec<QueueAdmissionId> {
        let ceiling = (self.config.pause_threshold - self.config.resume_hysteresis).max(0.0);
        let mut capacities: Vec<(WorkerWithDpRank, usize)> = self
            .available_capacities()
            .into_iter()
            .filter_map(|(worker, capacity)| {
                let limit = scale_tokens(capacity, ceiling);
                let remaining = limit.saturating_sub(usage.get(&worker).map_or(0, |u| u.used));
                (remaining > self.config.buffer_per_program).then_some((worker, remaining))
            })
            .collect();
        sort_capacities(&mut capacities);

        let mut paused: Vec<String> = self.paused.iter().cloned().collect();
        paused.sort_by_key(|session_id| {
            let program = &self.programs[session_id];
            let group = if program.step_count <= 1 {
                1
            } else if program.status == ProgramStatus::Reasoning {
                0
            } else {
                2
            };
            (group, current_token_total(program))
        });

        let mut ready = Vec::new();
        for session_id in paused {
            let required = current_token_total(&self.programs[&session_id])
                .saturating_add(self.config.buffer_per_program);
            let assigned = self.programs[&session_id].assigned_worker;
            let eligible = self
                .sessions
                .get(&session_id)
                .and_then(|requests| requests.current)
                .and_then(|id| self.requests.get(&id))
                .map(|request| &request.eligible_workers);
            let Some(position) = capacities.iter().position(|(worker, remaining)| {
                eligible.is_none_or(|eligible| eligible.contains(worker))
                    && assigned.is_none_or(|assigned| assigned == *worker)
                    && required <= *remaining
            }) else {
                continue;
            };
            let (worker, remaining) = capacities[position];
            ready.extend(self.resume_program(&session_id, Some(worker)));
            let program = &self.programs[&session_id];
            usage.entry(worker).or_default().add_program(
                self.program_tokens(program, false, now),
                self.program_tokens(program, true, now),
                self.config.buffer_per_program,
            );
            capacities[position].1 = remaining - required;
            sort_capacities(&mut capacities);
        }
        ready
    }

    fn force_timed_out(
        &mut self,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) -> Vec<QueueAdmissionId> {
        let timeout = Duration::from_secs_f64(self.config.resume_timeout_seconds);
        let timed_out: Vec<String> = self
            .paused
            .iter()
            .filter(|session_id| {
                self.programs
                    .get(*session_id)
                    .and_then(|program| program.deferred_since)
                    .is_some_and(|since| now.saturating_duration_since(since) >= timeout)
            })
            .cloned()
            .collect();
        let capacities = self.available_capacities();
        let mut ready = Vec::new();
        for session_id in timed_out {
            let assigned = self.programs[&session_id].assigned_worker;
            let eligible = self
                .sessions
                .get(&session_id)
                .and_then(|requests| requests.current)
                .and_then(|id| self.requests.get(&id))
                .map(|request| &request.eligible_workers);
            let target = capacities
                .iter()
                .filter(|(worker, _)| {
                    eligible.is_none_or(|eligible| eligible.contains(worker))
                        && assigned.is_none_or(|assigned| assigned == *worker)
                })
                .max_by_key(|(worker, capacity)| {
                    (
                        *capacity as i128
                            - usage.get(worker).map_or(0, |usage| usage.decayed) as i128,
                        Reverse(*worker),
                    )
                })
                .map(|(worker, _)| *worker);
            let any_available = self.workers.iter().any(|(worker, state)| {
                state.available && eligible.is_none_or(|eligible| eligible.contains(worker))
            });
            if target.is_none() && !any_available {
                continue;
            }
            ready.extend(self.resume_program(&session_id, target));
            if let Some(worker) = target {
                let program = &self.programs[&session_id];
                usage.entry(worker).or_default().add_program(
                    self.program_tokens(program, false, now),
                    self.program_tokens(program, true, now),
                    self.config.buffer_per_program,
                );
            }
        }
        ready
    }

    fn pause_until_safe(
        &mut self,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) {
        let mut acting = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        let mut reasoning = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        for (session_id, program) in &self.programs {
            if program.lifecycle != ProgramLifecycle::Active || program.marked_for_pause {
                continue;
            }
            let Some(worker) = program.assigned_worker else {
                continue;
            };
            match program.status {
                ProgramStatus::Acting => &mut acting,
                ProgramStatus::Reasoning => &mut reasoning,
            }
            .entry(worker)
            .or_default()
            .push((current_token_total(program), session_id.clone()));
        }
        for programs in acting.values_mut().chain(reasoning.values_mut()) {
            programs.sort_by_key(|(tokens, _)| *tokens);
        }

        for (worker, capacity) in self.available_capacities() {
            let threshold = scale_tokens(capacity, self.config.pause_threshold);
            if usage.get(&worker).map_or(0, |usage| usage.used) <= threshold {
                continue;
            }
            let target = scale_tokens(capacity, self.config.pause_target);
            if let Some(programs) = acting.get(&worker) {
                for (_, session_id) in programs {
                    if usage.get(&worker).map_or(0, |usage| usage.used) <= target {
                        break;
                    }
                    let program = &self.programs[session_id];
                    let normal = self.program_tokens(program, false, now);
                    let decayed = self.program_tokens(program, true, now);
                    self.pause_acting(session_id);
                    usage.entry(worker).or_default().remove_program(
                        normal,
                        decayed,
                        self.config.buffer_per_program,
                    );
                }
            }
            if usage.get(&worker).map_or(0, |usage| usage.used) > target
                && let Some(programs) = reasoning.get(&worker)
            {
                for (_, session_id) in programs {
                    if let Some(program) = self.programs.get_mut(session_id) {
                        program.marked_for_pause = true;
                    }
                }
            }
        }
    }

    fn pause_acting(&mut self, session_id: &str) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        if program.lifecycle != ProgramLifecycle::Active || program.status != ProgramStatus::Acting
        {
            return;
        }
        program.lifecycle = ProgramLifecycle::Paused;
        self.set_assignment(session_id, None);
        self.paused.insert(session_id.to_owned());
    }

    fn resume_program(
        &mut self,
        session_id: &str,
        worker: Option<WorkerWithDpRank>,
    ) -> Vec<QueueAdmissionId> {
        let deferred_id = self
            .sessions
            .get(session_id)
            .and_then(|requests| requests.current);
        let Some(program) = self.programs.get_mut(session_id) else {
            return Vec::new();
        };
        if program.lifecycle != ProgramLifecycle::Paused {
            return Vec::new();
        }
        program.lifecycle = ProgramLifecycle::Active;
        let was_deferred = program.deferred_since.take().is_some();
        self.set_assignment(session_id, worker);
        self.paused.remove(session_id);
        match (was_deferred, deferred_id) {
            (true, Some(id)) => vec![id],
            _ => Vec::new(),
        }
    }
}

impl QueueAdmissionPolicy for ThunderAgentPolicy {
    fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
        self.admit_request(request)
    }

    fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
        match event {
            QueueAdmissionEvent::Dispatched { id, worker } => self.dispatched(id, worker),
            QueueAdmissionEvent::Completed {
                request_id,
                context_tokens,
            } => {
                ready.extend(self.finish_by_request_id(request_id, true, context_tokens));
            }
            QueueAdmissionEvent::Aborted { request_id } => {
                ready.extend(self.finish_by_request_id(request_id, false, None));
            }
            QueueAdmissionEvent::Reconcile { snapshot } => {
                ready.extend(self.reconcile(snapshot.workers()));
            }
            _ => {}
        }
    }

    fn reconcile_interval(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.config.scheduler_interval_seconds,
        ))
    }
}

fn scale_tokens(tokens: usize, factor: f64) -> usize {
    ((tokens as f64) * factor).clamp(0.0, usize::MAX as f64) as usize
}

fn current_token_total(program: &Program) -> usize {
    program
        .request_progress
        .as_ref()
        .map_or(program.token_total, RequestProgress::context_tokens)
}

fn sort_capacities(capacities: &mut [(WorkerWithDpRank, usize)]) {
    capacities.sort_unstable_by_key(|(worker, remaining)| (Reverse(*remaining), *worker));
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::{QueueAdmissionWorkerSnapshot, SessionContext};

    use super::*;

    fn worker(id: u64, capacity: usize) -> QueueAdmissionWorker {
        QueueAdmissionWorker::new(WorkerWithDpRank::new(id, 0), Some(capacity), true)
    }

    fn snapshot(workers: impl Into<Vec<QueueAdmissionWorker>>) -> QueueAdmissionWorkerSnapshot {
        QueueAdmissionWorkerSnapshot::new(0, workers.into())
    }

    fn context(session_id: &str) -> SessionContext {
        SessionContext::new(session_id.to_owned(), None, None, None, None)
    }

    fn request<'a>(
        id: u64,
        request_id: &'a str,
        session: Option<&'a SessionContext>,
        workers: &'a QueueAdmissionWorkerSnapshot,
        tokens: usize,
    ) -> QueueAdmissionRequest<'a> {
        QueueAdmissionRequest::new(
            QueueAdmissionId::new(id),
            request_id,
            tokens,
            session,
            workers,
        )
    }

    fn policy(config: ThunderAgentConfig) -> ThunderAgentPolicy {
        ThunderAgentPolicy::new(config, Arc::new(SessionAssignments::default()))
    }

    #[test]
    fn serializes_requests_for_one_session() {
        let workers = [worker(1, 1_000)];
        let workers = snapshot(workers);
        let session = context("session-a");
        let mut policy = policy(Default::default());
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&session), &workers, 100)),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.admit(request(2, "request-2", Some(&session), &workers, 100)),
            QueueAdmissionDecision::Defer
        );
        policy.on_event(
            QueueAdmissionEvent::Dispatched {
                id: QueueAdmissionId::new(1),
                worker: WorkerWithDpRank::new(1, 0),
            },
            &mut Vec::new(),
        );
        let mut ready = Vec::new();
        policy.on_event(
            QueueAdmissionEvent::Completed {
                request_id: "request-1",
                context_tokens: Some(150),
            },
            &mut ready,
        );
        assert_eq!(ready, [QueueAdmissionId::new(2)]);
    }

    #[test]
    fn defers_when_program_capacity_is_exhausted() {
        let workers = [worker(1, 250)];
        let workers = snapshot(workers);
        let first = context("session-a");
        let second = context("session-b");
        let mut policy = policy(ThunderAgentConfig {
            buffer_per_program: 100,
            ..Default::default()
        });
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&first), &workers, 100)),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.admit(request(2, "request-2", Some(&second), &workers, 100)),
            QueueAdmissionDecision::Defer
        );
    }

    #[test]
    fn admission_ignores_request_ineligible_workers() {
        let workers = snapshot([worker(1, 1_000), worker(2, 1_000)]);
        let eligible = HashSet::from([WorkerWithDpRank::new(2, 0)]);
        let session = context("session-a");
        let mut policy = policy(Default::default());

        assert_eq!(
            policy.admit(
                request(1, "request-1", Some(&session), &workers, 100)
                    .with_eligible_workers(&eligible)
            ),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.programs["session-a"].assigned_worker,
            Some(WorkerWithDpRank::new(2, 0))
        );
    }

    #[test]
    fn aborted_request_rolls_back_new_program() {
        let workers = [worker(1, 1_000)];
        let workers = snapshot(workers);
        let session = context("session-a");
        let mut policy = policy(Default::default());
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&session), &workers, 100)),
            QueueAdmissionDecision::Ready
        );
        policy.on_event(
            QueueAdmissionEvent::Aborted {
                request_id: "request-1",
            },
            &mut Vec::new(),
        );
        assert!(!policy.programs.contains_key("session-a"));
    }

    #[test]
    fn sessionless_request_bypasses_without_state() {
        let workers = [worker(1, 1_000)];
        let workers = snapshot(workers);
        let mut policy = policy(Default::default());
        assert_eq!(
            policy.admit(request(1, "request-1", None, &workers, 100)),
            QueueAdmissionDecision::Bypass
        );
        assert!(policy.programs.is_empty());
        assert!(policy.requests.is_empty());
    }

    #[test]
    fn reconcile_wakes_program_after_capacity_grows() {
        let small = snapshot([worker(1, 250)]);
        let large = snapshot([worker(1, 500)]);
        let first = context("session-a");
        let second = context("session-b");
        let mut policy = policy(ThunderAgentConfig {
            buffer_per_program: 100,
            ..Default::default()
        });
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&first), &small, 100)),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.admit(request(2, "request-2", Some(&second), &small, 100)),
            QueueAdmissionDecision::Defer
        );

        policy.next_tick = Instant::now();
        let mut ready = Vec::new();
        policy.on_event(
            QueueAdmissionEvent::Reconcile { snapshot: &large },
            &mut ready,
        );
        assert_eq!(ready, [QueueAdmissionId::new(2)]);
        assert_eq!(
            policy.programs["session-b"].assigned_worker,
            Some(WorkerWithDpRank::new(1, 0))
        );
    }

    #[test]
    fn dispatch_records_actual_worker_for_next_turn() {
        let workers = [worker(1, 1_000), worker(2, 1_000)];
        let workers = snapshot(workers);
        let session = context("session-a");
        let mut policy = policy(Default::default());
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&session), &workers, 100)),
            QueueAdmissionDecision::Ready
        );
        policy.on_event(
            QueueAdmissionEvent::Dispatched {
                id: QueueAdmissionId::new(1),
                worker: WorkerWithDpRank::new(2, 0),
            },
            &mut Vec::new(),
        );
        policy.on_event(
            QueueAdmissionEvent::Completed {
                request_id: "request-1",
                context_tokens: Some(150),
            },
            &mut Vec::new(),
        );
        assert_eq!(policy.programs["session-a"].token_total, 150);
        assert_eq!(
            policy.admit(request(2, "request-2", Some(&session), &workers, 100)),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.programs["session-a"].assigned_worker,
            Some(WorkerWithDpRank::new(2, 0))
        );
    }

    #[test]
    fn live_progress_updates_reasoning_worker_usage() {
        let assigned = WorkerWithDpRank::new(1, 0);
        let (progress, updater) = RequestProgress::new(100);
        let mut policy = policy(ThunderAgentConfig {
            buffer_per_program: 100,
            ..Default::default()
        });
        policy.programs.insert(
            "session-a".to_owned(),
            Program {
                assigned_worker: Some(assigned),
                token_total: 100,
                request_progress: Some(progress),
                ..Default::default()
            },
        );

        updater.update_context_tokens(400);

        assert_eq!(policy.worker_usage(Instant::now())[&assigned].used, 500);
    }

    #[test]
    fn pressure_pauses_acting_program_and_capacity_resumes_it() {
        let constrained = snapshot([worker(1, 500)]);
        let expanded = snapshot([worker(1, 1_000)]);
        let session = context("session-a");
        let mut policy = policy(ThunderAgentConfig {
            buffer_per_program: 100,
            ..Default::default()
        });
        assert_eq!(
            policy.admit(request(1, "request-1", Some(&session), &constrained, 400,)),
            QueueAdmissionDecision::Ready
        );
        policy.on_event(
            QueueAdmissionEvent::Dispatched {
                id: QueueAdmissionId::new(1),
                worker: WorkerWithDpRank::new(1, 0),
            },
            &mut Vec::new(),
        );
        policy.on_event(
            QueueAdmissionEvent::Completed {
                request_id: "request-1",
                context_tokens: Some(400),
            },
            &mut Vec::new(),
        );

        policy.next_tick = Instant::now();
        policy.on_event(
            QueueAdmissionEvent::Reconcile {
                snapshot: &constrained,
            },
            &mut Vec::new(),
        );
        assert_eq!(
            policy.programs["session-a"].lifecycle,
            ProgramLifecycle::Paused
        );

        policy.next_tick = Instant::now();
        policy.on_event(
            QueueAdmissionEvent::Reconcile {
                snapshot: &expanded,
            },
            &mut Vec::new(),
        );
        assert_eq!(
            policy.programs["session-a"].lifecycle,
            ProgramLifecycle::Active
        );
    }
}
