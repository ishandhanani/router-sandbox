// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::ClassifyEvent;
use tokio::sync::Notify;

use crate::ThunderAgentConfig;
use crate::capacity::WorkerCapacitySnapshot;
use crate::policy::ThunderAgentError;
use crate::selection::SessionAssignments;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramStatus {
    Reasoning,
    Acting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramLifecycle {
    Active,
    Paused,
}

#[derive(Debug, Clone)]
pub(crate) struct Program {
    status: ProgramStatus,
    pub(crate) lifecycle: ProgramLifecycle,
    pub(crate) assigned_worker: Option<WorkerWithDpRank>,
    token_total: usize,
    pub(crate) step_count: usize,
    marked_for_pause: bool,
    acting_since: Option<Instant>,
    deferred_since: Option<Instant>,
}

impl Program {
    pub(crate) fn new(input_tokens: usize) -> Self {
        Self {
            status: ProgramStatus::Reasoning,
            lifecycle: ProgramLifecycle::Active,
            assigned_worker: None,
            token_total: input_tokens,
            step_count: 1,
            marked_for_pause: false,
            acting_since: None,
            deferred_since: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestPhase {
    Waiting,
    Released,
}

pub(crate) struct RequestState {
    session_id: String,
    input_tokens: usize,
    session_final: bool,
    phase: RequestPhase,
    prior_program: Option<Program>,
    began_program: bool,
    pub(crate) notify: Arc<Notify>,
}

#[derive(Default)]
struct SessionRequests {
    current: Option<String>,
    waiting: VecDeque<String>,
    stale_waiting: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaitStatus {
    Waiting,
    Released,
    Missing,
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

pub(crate) struct State {
    config: ThunderAgentConfig,
    assignments: Arc<SessionAssignments>,
    pub(crate) programs: HashMap<String, Program>,
    paused_programs: HashSet<String>,
    normal_usage: HashMap<WorkerWithDpRank, usize>,
    pub(crate) requests: HashMap<String, RequestState>,
    sessions: HashMap<String, SessionRequests>,
    pub(crate) arrival_order: VecDeque<String>,
    waiting_arrivals: HashSet<String>,
    pub(crate) timer_owner: Option<String>,
    next_reconcile_at: Instant,
    next_retention_expiry: Option<Instant>,
    capacity_snapshot_id: Option<u64>,
}

impl State {
    pub(crate) fn new(config: ThunderAgentConfig, assignments: Arc<SessionAssignments>) -> Self {
        let next_reconcile_at = Instant::now() + config.scheduler_interval();
        Self {
            config,
            assignments,
            programs: HashMap::new(),
            paused_programs: HashSet::new(),
            normal_usage: HashMap::new(),
            requests: HashMap::new(),
            sessions: HashMap::new(),
            arrival_order: VecDeque::new(),
            waiting_arrivals: HashSet::new(),
            timer_owner: None,
            next_reconcile_at,
            next_retention_expiry: None,
            capacity_snapshot_id: None,
        }
    }

    pub(crate) fn register(
        &mut self,
        request_id: String,
        session_id: String,
        input_tokens: usize,
        session_final: bool,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> Result<Arc<Notify>, ThunderAgentError> {
        if self.requests.contains_key(&request_id) {
            return Err(ThunderAgentError::DuplicateRequestId(request_id));
        }
        if self.requests.len() >= self.config.max_tracked_requests {
            return Err(ThunderAgentError::RequestLimitExceeded {
                limit: self.config.max_tracked_requests,
            });
        }
        self.expire_retained_programs(now);
        self.ensure_program_capacity(&session_id)?;
        self.clear_removed_workers(capacities);
        let mut usage = self.normal_worker_usage();
        self.pause_until_safe(capacities, &mut usage, now);
        let notify = Arc::new(Notify::new());

        self.sessions
            .entry(session_id.clone())
            .or_default()
            .waiting
            .push_back(request_id.clone());
        self.arrival_order.push_back(request_id.clone());
        self.waiting_arrivals.insert(request_id.clone());
        if self.timer_owner.is_none() {
            self.timer_owner = Some(request_id.clone());
        }
        self.requests.insert(
            request_id.clone(),
            RequestState {
                session_id,
                input_tokens,
                session_final,
                phase: RequestPhase::Waiting,
                prior_program: None,
                began_program: false,
                notify: Arc::clone(&notify),
            },
        );

        self.admit_request(&request_id, capacities, now);
        self.compact_arrival_order();
        Ok(notify)
    }

    pub(crate) fn wait_status(&self, request_id: &str) -> WaitStatus {
        match self.requests.get(request_id).map(|request| request.phase) {
            Some(RequestPhase::Waiting) => WaitStatus::Waiting,
            Some(RequestPhase::Released) => WaitStatus::Released,
            None => WaitStatus::Missing,
        }
    }

    fn ensure_program_capacity(&mut self, session_id: &str) -> Result<(), ThunderAgentError> {
        if self.programs.contains_key(session_id)
            || self.programs.len() < self.config.max_tracked_requests
        {
            return Ok(());
        }

        let evicted = self
            .programs
            .iter()
            .filter(|(candidate, _)| !self.sessions.contains_key(candidate.as_str()))
            .min_by_key(|(_, program)| program.acting_since)
            .map(|(candidate, _)| candidate.clone());
        let Some(evicted) = evicted else {
            return Err(ThunderAgentError::ProgramLimitExceeded {
                limit: self.config.max_tracked_requests,
            });
        };
        self.remove_program(&evicted);
        Ok(())
    }

    fn program_charge(&self, program: &Program) -> Option<(WorkerWithDpRank, usize)> {
        if program.lifecycle != ProgramLifecycle::Active {
            return None;
        }
        let worker = program.assigned_worker?;
        let tokens = if program.status == ProgramStatus::Acting {
            scale_tokens(program.token_total, self.config.acting_token_weight)
        } else {
            program.token_total
        };
        Some((
            worker,
            tokens.saturating_add(self.config.buffer_per_program),
        ))
    }

    fn subtract_charge(&mut self, charge: Option<(WorkerWithDpRank, usize)>) {
        let Some((worker, tokens)) = charge else {
            return;
        };
        let Some(used) = self.normal_usage.get_mut(&worker) else {
            return;
        };
        *used = used.saturating_sub(tokens);
        if *used == 0 {
            self.normal_usage.remove(&worker);
        }
    }

    fn add_charge(&mut self, charge: Option<(WorkerWithDpRank, usize)>) {
        if let Some((worker, tokens)) = charge {
            let used = self.normal_usage.entry(worker).or_default();
            *used = used.saturating_add(tokens);
        }
    }

    fn update_program<R>(
        &mut self,
        session_id: &str,
        update: impl FnOnce(&mut Program) -> R,
    ) -> Option<R> {
        let before = self
            .programs
            .get(session_id)
            .and_then(|program| self.program_charge(program));
        self.subtract_charge(before);
        let result = update(self.programs.get_mut(session_id)?);
        let (after, paused, assignment) = self.programs.get(session_id).map(|program| {
            (
                self.program_charge(program),
                program.lifecycle == ProgramLifecycle::Paused,
                program.assigned_worker,
            )
        })?;
        self.add_charge(after);
        if paused {
            self.paused_programs.insert(session_id.to_owned());
        } else {
            self.paused_programs.remove(session_id);
        }
        self.assignments.set(session_id, assignment);
        Some(result)
    }

    fn insert_program(&mut self, session_id: String, program: Program) {
        let charge = self.program_charge(&program);
        let paused = program.lifecycle == ProgramLifecycle::Paused;
        let assignment = program.assigned_worker;
        debug_assert!(!self.programs.contains_key(&session_id));
        self.programs.insert(session_id.clone(), program);
        self.add_charge(charge);
        if paused {
            self.paused_programs.insert(session_id.clone());
        }
        self.assignments.set(&session_id, assignment);
    }

    fn remove_program(&mut self, session_id: &str) -> Option<Program> {
        let program = self.programs.remove(session_id)?;
        let charge = self.program_charge(&program);
        self.subtract_charge(charge);
        self.paused_programs.remove(session_id);
        self.assignments.set(session_id, None);
        Some(program)
    }

    fn replace_program(&mut self, session_id: String, program: Program) {
        self.remove_program(&session_id);
        self.insert_program(session_id, program);
    }

    pub(crate) fn reconcile(&mut self, capacities: &WorkerCapacitySnapshot, now: Instant) -> bool {
        self.next_reconcile_at = now + self.config.scheduler_interval();
        let mut changed = self.expire_retained_programs(now);
        changed |= self.clear_removed_workers(capacities);
        changed |= self.admit_front_requests(capacities, now);
        let mut usage = self.worker_usage(now);
        changed |= self.greedy_resume(capacities, &mut usage, now);
        changed |= self.force_timed_out(capacities, &mut usage, now);
        changed |= self.pause_until_safe(capacities, &mut usage, now);
        self.compact_arrival_order();
        changed
    }

    pub(crate) fn reconcile_if_due(
        &mut self,
        request_id: &str,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> bool {
        if self.timer_owner.as_deref() != Some(request_id) || now < self.next_reconcile_at {
            return false;
        }
        self.reconcile(capacities, now)
    }

    pub(crate) fn reconcile_delay(&self, request_id: &str, now: Instant) -> Option<Duration> {
        (self.timer_owner.as_deref() == Some(request_id))
            .then(|| self.next_reconcile_at.saturating_duration_since(now))
    }

    fn admit_front_requests(&mut self, capacities: &WorkerCapacitySnapshot, now: Instant) -> bool {
        let candidates: Vec<String> = self
            .arrival_order
            .iter()
            .filter(|request_id| self.waiting_arrivals.contains(request_id.as_str()))
            .filter_map(|request_id| {
                let request = self.requests.get(request_id)?;
                (request.phase == RequestPhase::Waiting
                    && self.is_front_and_idle(request_id, &request.session_id))
                .then(|| request_id.clone())
            })
            .collect();

        let mut changed = false;
        for request_id in candidates {
            changed |= self.admit_request(&request_id, capacities, now);
        }
        changed
    }

    fn admit_request(
        &mut self,
        request_id: &str,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> bool {
        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        if request.phase != RequestPhase::Waiting
            || !self.is_front_and_idle(request_id, &request.session_id)
            || !self.begin_request(request_id)
        {
            return false;
        }

        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        let session_id = request.session_id.clone();
        let input_tokens = request.input_tokens;
        let was_new = request.prior_program.is_none();
        let Some(program) = self.programs.get(&session_id) else {
            return false;
        };
        let lifecycle = program.lifecycle;
        let assigned_worker = program.assigned_worker;

        if lifecycle == ProgramLifecycle::Paused {
            return self.defer_program(&session_id, now, assigned_worker.is_some());
        }

        if !capacities.has_usable_capacity() {
            return self.release_request(
                request_id,
                assigned_worker.filter(|worker| capacities.is_live(*worker)),
            );
        }

        let mut changed = false;
        if let Some(worker) = assigned_worker {
            if capacities.is_live(worker) {
                return self.release_request(request_id, Some(worker));
            }
            self.set_assignment(&session_id, None);
            changed = true;
        }

        if was_new && !self.paused_programs.is_empty() {
            return self.defer_program(&session_id, now, false) || changed;
        }

        let required = self.request_cost(input_tokens);
        let selected = capacities
            .iter()
            .filter(|(worker, _)| capacities.is_live(*worker))
            .filter_map(|(worker, capacity)| {
                let used = self.normal_usage.get(&worker).copied().unwrap_or(0);
                capacity
                    .checked_sub(used)
                    .is_some_and(|remaining| remaining >= required)
                    .then_some((worker, used))
            })
            .min_by_key(|(worker, used)| (*used, *worker))
            .map(|(worker, _)| worker);
        match selected {
            Some(worker) => self.release_request(request_id, Some(worker)) || changed,
            None => self.defer_program(&session_id, now, false) || changed,
        }
    }

    fn begin_request(&mut self, request_id: &str) -> bool {
        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        if request.began_program {
            return true;
        }
        let session_id = request.session_id.clone();
        let input_tokens = request.input_tokens;
        let prior_program = self.programs.get(&session_id).cloned();

        if self.programs.contains_key(&session_id) {
            self.update_program(&session_id, |program| {
                program.status = ProgramStatus::Reasoning;
                if input_tokens > 0 {
                    program.token_total = input_tokens;
                }
                program.step_count = program.step_count.saturating_add(1);
                program.acting_since = None;
            });
        } else {
            self.insert_program(session_id, Program::new(input_tokens));
        }
        if let Some(request) = self.requests.get_mut(request_id) {
            request.prior_program = prior_program;
            request.began_program = true;
        }
        true
    }

    fn defer_program(&mut self, session_id: &str, now: Instant, preserve_assignment: bool) -> bool {
        let Some(program) = self.programs.get(session_id) else {
            return false;
        };
        let lifecycle_changed = program.lifecycle != ProgramLifecycle::Paused;
        let timer_changed = program.deferred_since.is_none();
        let assignment_changed = !preserve_assignment && program.assigned_worker.is_some();
        self.update_program(session_id, |program| {
            program.lifecycle = ProgramLifecycle::Paused;
            program.deferred_since.get_or_insert(now);
            if !preserve_assignment {
                program.assigned_worker = None;
            }
        });
        lifecycle_changed || timer_changed || assignment_changed
    }

    fn release_request(&mut self, request_id: &str, worker: Option<WorkerWithDpRank>) -> bool {
        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        if request.phase != RequestPhase::Waiting
            || !request.began_program
            || !self.is_front_and_idle(request_id, &request.session_id)
        {
            return false;
        }
        let session_id = request.session_id.clone();
        let Some(program) = self.programs.get(&session_id) else {
            return false;
        };
        let assigned_worker = worker.or(program.assigned_worker);
        if self
            .sessions
            .get(&session_id)
            .and_then(|session| session.waiting.front())
            .map(String::as_str)
            != Some(request_id)
        {
            return false;
        }

        self.update_program(&session_id, |program| {
            program.lifecycle = ProgramLifecycle::Active;
            program.deferred_since = None;
            program.assigned_worker = assigned_worker;
        });

        let Some(session) = self.sessions.get_mut(&session_id) else {
            return false;
        };
        session.waiting.pop_front();
        session.current = Some(request_id.to_owned());

        let notify = if let Some(request) = self.requests.get_mut(request_id) {
            request.phase = RequestPhase::Released;
            Arc::clone(&request.notify)
        } else {
            return false;
        };
        self.remove_waiting_arrival(request_id);
        notify.notify_waiters();
        true
    }

    fn greedy_resume(
        &mut self,
        capacities: &WorkerCapacitySnapshot,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) -> bool {
        let ceiling = (self.config.pause_threshold - self.config.resume_hysteresis).max(0.0);
        let mut remaining: Vec<(WorkerWithDpRank, usize)> = capacities
            .iter()
            .filter(|(worker, _)| capacities.is_live(*worker))
            .filter_map(|(worker, capacity)| {
                let limit = scale_tokens(capacity, ceiling);
                let available = limit.saturating_sub(usage.get(&worker).map_or(0, |u| u.used));
                (available > self.config.buffer_per_program).then_some((worker, available))
            })
            .collect();
        sort_capacities(&mut remaining);

        let mut paused: Vec<(u8, usize, String)> = self
            .paused_programs
            .iter()
            .filter_map(|session_id| {
                let program = self.programs.get(session_id)?;
                let group = if program.step_count <= 1 {
                    1
                } else if program.status == ProgramStatus::Reasoning {
                    0
                } else {
                    2
                };
                Some((group, program.token_total, session_id.clone()))
            })
            .collect();
        paused.sort_unstable();

        let mut changed = false;
        for (_, _, session_id) in paused {
            let Some(program) = self.programs.get(&session_id) else {
                continue;
            };
            let required = program
                .token_total
                .saturating_add(self.config.buffer_per_program);
            let assigned = program.assigned_worker;
            let Some(position) = remaining.iter().position(|(worker, available)| {
                assigned.is_none_or(|assigned| assigned == *worker) && required <= *available
            }) else {
                continue;
            };
            let (worker, available) = remaining[position];
            if self.resume_program(&session_id, Some(worker)) {
                let Some(program) = self.programs.get(&session_id) else {
                    continue;
                };
                usage.entry(worker).or_default().add_program(
                    self.program_tokens(program, false, now),
                    self.program_tokens(program, true, now),
                    self.config.buffer_per_program,
                );
                remaining[position].1 = available - required;
                sort_capacities(&mut remaining);
                changed = true;
            }
        }
        changed
    }

    fn force_timed_out(
        &mut self,
        capacities: &WorkerCapacitySnapshot,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) -> bool {
        let timeout = self.config.resume_timeout();
        let timed_out: Vec<String> = self
            .paused_programs
            .iter()
            .filter(|session_id| {
                self.programs.get(*session_id).is_some_and(|program| {
                    program
                        .deferred_since
                        .is_some_and(|since| now.saturating_duration_since(since) >= timeout)
                })
            })
            .cloned()
            .collect();

        let mut changed = false;
        for session_id in timed_out {
            let Some(program) = self.programs.get(&session_id) else {
                continue;
            };
            let assigned = program.assigned_worker;
            let target = capacities
                .iter()
                .filter(|(worker, _)| {
                    capacities.is_live(*worker)
                        && assigned.is_none_or(|assigned| assigned == *worker)
                })
                .max_by_key(|(worker, capacity)| {
                    (
                        *capacity as i128
                            - usage.get(worker).map_or(0, |usage| usage.decayed) as i128,
                        Reverse(*worker),
                    )
                })
                .map(|(worker, _)| worker);
            if target.is_none() && !capacities.has_live_worker() {
                continue;
            }
            if self.resume_program(&session_id, target) {
                if let Some(worker) = target {
                    let Some(program) = self.programs.get(&session_id) else {
                        continue;
                    };
                    usage.entry(worker).or_default().add_program(
                        self.program_tokens(program, false, now),
                        self.program_tokens(program, true, now),
                        self.config.buffer_per_program,
                    );
                }
                changed = true;
            }
        }
        changed
    }

    fn resume_program(&mut self, session_id: &str, worker: Option<WorkerWithDpRank>) -> bool {
        let Some(program) = self.programs.get(session_id) else {
            return false;
        };
        if program.lifecycle != ProgramLifecycle::Paused {
            return false;
        }
        self.update_program(session_id, |program| {
            program.lifecycle = ProgramLifecycle::Active;
            program.deferred_since = None;
            program.assigned_worker = worker;
        });

        let pending = self.sessions.get(session_id).and_then(|session| {
            session
                .current
                .is_none()
                .then(|| session.waiting.front().cloned())
                .flatten()
        });
        if let Some(request_id) = pending {
            self.release_request(&request_id, worker);
        }
        true
    }

    fn pause_until_safe(
        &mut self,
        capacities: &WorkerCapacitySnapshot,
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) -> bool {
        let mut changed = false;
        for (worker, capacity) in capacities.iter() {
            let threshold = scale_tokens(capacity, self.config.pause_threshold);
            if usage.get(&worker).map_or(0, |usage| usage.used) <= threshold {
                continue;
            }
            let target = scale_tokens(capacity, self.config.pause_target);
            let mut acting = self.programs_for_worker(worker, ProgramStatus::Acting);
            let mut reasoning = self.programs_for_worker(worker, ProgramStatus::Reasoning);
            acting.sort_by_key(|(tokens, _)| *tokens);
            reasoning.sort_by_key(|(tokens, _)| *tokens);

            for (_, session_id) in acting {
                if usage.get(&worker).map_or(0, |usage| usage.used) <= target {
                    break;
                }
                let Some(program) = self.programs.get(&session_id) else {
                    continue;
                };
                let normal = self.program_tokens(program, false, now);
                let decayed = self.program_tokens(program, true, now);
                if self.pause_acting(&session_id) {
                    usage.entry(worker).or_default().remove_program(
                        normal,
                        decayed,
                        self.config.buffer_per_program,
                    );
                    changed = true;
                }
            }
            if usage.get(&worker).map_or(0, |usage| usage.used) > target {
                for (_, session_id) in reasoning {
                    if self
                        .programs
                        .get(&session_id)
                        .is_some_and(|program| !program.marked_for_pause)
                    {
                        self.update_program(&session_id, |program| {
                            program.marked_for_pause = true;
                        });
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    fn pause_acting(&mut self, session_id: &str) -> bool {
        let Some(program) = self.programs.get(session_id) else {
            return false;
        };
        if program.lifecycle != ProgramLifecycle::Active || program.status != ProgramStatus::Acting
        {
            return false;
        }
        self.update_program(session_id, |program| {
            program.lifecycle = ProgramLifecycle::Paused;
            program.assigned_worker = None;
        });
        true
    }

    fn programs_for_worker(
        &self,
        worker: WorkerWithDpRank,
        status: ProgramStatus,
    ) -> Vec<(usize, String)> {
        self.programs
            .iter()
            .filter(|(_, program)| {
                program.lifecycle == ProgramLifecycle::Active
                    && program.status == status
                    && program.assigned_worker == Some(worker)
                    && !program.marked_for_pause
            })
            .map(|(session_id, program)| (program.token_total, session_id.clone()))
            .collect()
    }

    pub(crate) fn on_event(
        &mut self,
        event: ClassifyEvent<'_>,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> bool {
        match event {
            ClassifyEvent::Sent { request_id, worker } => self.sent(request_id, worker, now),
            ClassifyEvent::Completed {
                request_id,
                context_tokens,
                ..
            } => self.finish_request(request_id, true, context_tokens, capacities, now),
            ClassifyEvent::Aborted { request_id, .. } => {
                self.finish_request(request_id, false, None, capacities, now)
            }
            ClassifyEvent::Received { .. } | ClassifyEvent::Responding { .. } => false,
            _ => false,
        }
    }

    fn sent(&mut self, request_id: &str, worker: WorkerWithDpRank, now: Instant) -> bool {
        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        if request.phase != RequestPhase::Released {
            return false;
        }
        let session_id = request.session_id.clone();
        self.set_assignment(&session_id, Some(worker));
        self.request_reconcile(now);
        true
    }

    pub(crate) fn cancel_request(
        &mut self,
        request_id: &str,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> bool {
        self.finish_request(request_id, false, None, capacities, now)
    }

    fn finish_request(
        &mut self,
        request_id: &str,
        completed: bool,
        context_tokens: Option<usize>,
        capacities: &WorkerCapacitySnapshot,
        now: Instant,
    ) -> bool {
        let Some(request) = self.requests.remove(request_id) else {
            return false;
        };
        let notify = Arc::clone(&request.notify);
        let was_waiting = self.remove_waiting_arrival(request_id);

        if let Some(session) = self.sessions.get_mut(&request.session_id) {
            if session.current.as_deref() == Some(request_id) {
                session.current = None;
            } else if was_waiting {
                session.stale_waiting = session.stale_waiting.saturating_add(1);
            }
        }
        self.compact_session_waiting(&request.session_id);

        if completed && request.phase == RequestPhase::Released {
            if request.session_final {
                self.remove_program(&request.session_id);
            } else {
                let pause = self
                    .update_program(&request.session_id, |program| {
                        program.status = ProgramStatus::Acting;
                        program.token_total = context_tokens.unwrap_or(request.input_tokens);
                        program.acting_since = Some(now);
                        program.deferred_since = None;
                        std::mem::take(&mut program.marked_for_pause)
                    })
                    .unwrap_or(false);
                if pause {
                    self.pause_acting(&request.session_id);
                }
            }
        } else if request.began_program {
            match request.prior_program {
                Some(prior) => {
                    self.replace_program(request.session_id.clone(), prior);
                }
                None => {
                    self.remove_program(&request.session_id);
                }
            }
        }

        let mut usage = self.normal_worker_usage();
        self.pause_until_safe(capacities, &mut usage, now);

        let next = self.sessions.get(&request.session_id).and_then(|session| {
            session
                .current
                .is_none()
                .then(|| session.waiting.front().cloned())
                .flatten()
        });
        if let Some(next) = next {
            self.admit_request(&next, capacities, now);
        }

        if self
            .sessions
            .get(&request.session_id)
            .is_some_and(|session| session.current.is_none() && session.waiting.is_empty())
        {
            self.sessions.remove(&request.session_id);
        }
        self.schedule_retention_expiry(&request.session_id);

        self.request_reconcile(now);
        self.compact_arrival_order();
        notify.notify_waiters();
        true
    }

    fn compact_session_waiting(&mut self, session_id: &str) {
        let Some(session) = self.sessions.get_mut(session_id) else {
            return;
        };
        while session
            .waiting
            .front()
            .is_some_and(|request_id| !self.waiting_arrivals.contains(request_id))
        {
            session.waiting.pop_front();
            session.stale_waiting = session.stale_waiting.saturating_sub(1);
        }

        if session.stale_waiting >= 8
            && session.stale_waiting.saturating_mul(2) >= session.waiting.len()
        {
            session
                .waiting
                .retain(|request_id| self.waiting_arrivals.contains(request_id));
            session.stale_waiting = 0;
        }
    }

    fn clear_removed_workers(&mut self, capacities: &WorkerCapacitySnapshot) -> bool {
        if self.capacity_snapshot_id == Some(capacities.id()) {
            return false;
        }
        self.capacity_snapshot_id = Some(capacities.id());
        if capacities.is_empty() && !capacities.has_liveness() {
            return false;
        }
        let removed: Vec<String> = self
            .programs
            .iter()
            .filter(|(_, program)| {
                program
                    .assigned_worker
                    .is_some_and(|worker| !capacities.is_live(worker))
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();
        for session_id in &removed {
            self.set_assignment(session_id, None);
        }
        !removed.is_empty()
    }

    fn expire_retained_programs(&mut self, now: Instant) -> bool {
        if self
            .next_retention_expiry
            .is_none_or(|deadline| now < deadline)
        {
            return false;
        }

        let retention = self.config.session_retention();
        let mut expired = Vec::new();
        let mut next_expiry = None;
        for (session_id, program) in &self.programs {
            let has_request = self
                .sessions
                .get(session_id)
                .is_some_and(|session| session.current.is_some() || !session.waiting.is_empty());
            let Some(since) = (!has_request && program.status == ProgramStatus::Acting)
                .then_some(program.acting_since)
                .flatten()
            else {
                continue;
            };
            let deadline = since + retention;
            if now >= deadline {
                expired.push(session_id.clone());
            } else {
                next_expiry =
                    Some(next_expiry.map_or(deadline, |next: Instant| next.min(deadline)));
            }
        }
        self.next_retention_expiry = next_expiry;
        for session_id in &expired {
            self.remove_program(session_id);
        }
        !expired.is_empty()
    }

    fn schedule_retention_expiry(&mut self, session_id: &str) {
        if self.sessions.contains_key(session_id) {
            return;
        }
        let Some(program) = self.programs.get(session_id) else {
            return;
        };
        let Some(since) = (program.status == ProgramStatus::Acting)
            .then_some(program.acting_since)
            .flatten()
        else {
            return;
        };
        let deadline = since + self.config.session_retention();
        self.next_retention_expiry = Some(
            self.next_retention_expiry
                .map_or(deadline, |next| next.min(deadline)),
        );
    }

    fn worker_usage(&self, now: Instant) -> HashMap<WorkerWithDpRank, WorkerUsage> {
        let mut usage = self.normal_worker_usage();
        for program in self.programs.values() {
            if program.lifecycle == ProgramLifecycle::Active
                && let Some(worker) = program.assigned_worker
            {
                let decayed = self
                    .program_tokens(program, true, now)
                    .saturating_add(self.config.buffer_per_program);
                let worker_usage = usage.entry(worker).or_default();
                worker_usage.decayed = worker_usage.decayed.saturating_add(decayed);
            }
        }
        usage
    }

    fn normal_worker_usage(&self) -> HashMap<WorkerWithDpRank, WorkerUsage> {
        self.normal_usage
            .iter()
            .map(|(&worker, &used)| (worker, WorkerUsage { used, decayed: 0 }))
            .collect()
    }

    fn program_tokens(&self, program: &Program, decayed: bool, now: Instant) -> usize {
        if program.status != ProgramStatus::Acting {
            return program.token_total;
        }
        let weight = if decayed {
            let idle = program
                .acting_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
            2.0_f64.powf(-idle.as_secs_f64() / self.config.acting_decay_tau_seconds)
        } else {
            self.config.acting_token_weight
        };
        scale_tokens(program.token_total, weight)
    }

    fn request_cost(&self, input_tokens: usize) -> usize {
        input_tokens.saturating_add(self.config.buffer_per_program)
    }

    fn set_assignment(&mut self, session_id: &str, worker: Option<WorkerWithDpRank>) {
        self.update_program(session_id, |program| {
            program.assigned_worker = worker;
        });
    }

    fn is_front_and_idle(&self, request_id: &str, session_id: &str) -> bool {
        self.sessions.get(session_id).is_some_and(|session| {
            session.current.is_none()
                && session.waiting.front().map(String::as_str) == Some(request_id)
        })
    }

    fn remove_waiting_arrival(&mut self, request_id: &str) -> bool {
        if !self.waiting_arrivals.remove(request_id) {
            return false;
        }
        if self.timer_owner.as_deref() == Some(request_id) {
            while self
                .arrival_order
                .front()
                .is_some_and(|candidate| !self.waiting_arrivals.contains(candidate))
            {
                self.arrival_order.pop_front();
            }
            self.timer_owner = self.arrival_order.front().cloned();
            if let Some(owner) = self.timer_owner.as_deref()
                && let Some(request) = self.requests.get(owner)
            {
                request.notify.notify_waiters();
            }
        }
        true
    }

    fn request_reconcile(&mut self, now: Instant) {
        let Some(owner) = self.timer_owner.as_deref() else {
            return;
        };
        self.next_reconcile_at = now;
        if let Some(request) = self.requests.get(owner) {
            request.notify.notify_waiters();
        }
    }

    fn compact_arrival_order(&mut self) {
        let retained = self.waiting_arrivals.len();
        if self.arrival_order.len() <= retained.saturating_mul(2).saturating_add(256) {
            return;
        }
        self.arrival_order
            .retain(|request_id| self.waiting_arrivals.contains(request_id));
    }
}

fn scale_tokens(tokens: usize, factor: f64) -> usize {
    ((tokens as f64) * factor).clamp(0.0, usize::MAX as f64) as usize
}

fn sort_capacities(capacities: &mut [(WorkerWithDpRank, usize)]) {
    capacities.sort_unstable_by_key(|(worker, remaining)| (Reverse(*remaining), *worker));
}
