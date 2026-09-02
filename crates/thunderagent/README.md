# ThunderAgent

`thunderagent-dynamo-policy` implements ThunderAgent's program-aware flow control on Dynamo's asynchronous request-classifier API. It registers a request classifier and a worker-selection policy through Dynamo's statically linked router-plugin catalog; no Python adapter is required.

This prototype is stacked on [ai-dynamo/dynamo#14123](https://github.com/ai-dynamo/dynamo/pull/14123) and its request-classifier catalog follow-up.

## Source layout

```text
src/
├── lib.rs                       # Public exports and catalog registration façade
├── config.rs                    # ThunderAgent configuration and validation
├── request_classifier/
│   ├── mod.rs                   # RequestClassifier implementation and plugin factory
│   ├── capacity.rs              # Cached worker capacity and liveness view
│   └── scheduler.rs             # Program state, admission, pause, resume, and repacking
└── worker_selection/
    └── mod.rs                   # Stateless load scorer, target-aware picker, and plugin factory
```

## Ownership and routing

`ThunderAgentClassifier` owns the complete program table: per-session serialization, reasoning/acting state, pause state, retained token usage, and worker assignment. The state is internally reference-counted because `classify` must return independently pollable `'static` futures, but it is not shared with the worker selector.

The classifier chooses a request-local worker target when it releases a request. ThunderAgent's stateless selector honors that target when it is eligible and otherwise falls back to current least-loaded placement. The `Sent` lifecycle event reconciles the program table with the worker Dynamo actually selected. If a worker disappears, the classifier clears the assignment and chooses a live replacement.

Use soft session affinity. Affinity preserves locality between requests, while soft mode allows the classifier to repack a paused program onto a different worker. Hard affinity would prevent that Python-parity behavior before the custom selector runs.

Requests without Dynamo session context return from classification immediately.

```mermaid
flowchart TD
    A["Request enters classify"] --> B{"Has session context?"}
    B -- "No" --> C["Return unchanged"]
    B -- "Yes" --> D["Register in classifier-owned program table"]
    D --> E{"Session busy or program paused?"}
    E -- "Yes" --> F["Keep the classify future pending"]
    E -- "No" --> G{"Known pressure permits release?"}
    G -- "No" --> F
    G -- "Yes" --> H["Attach chosen worker target and return request"]
    H --> I["Stateless selector honors target or falls back"]
    I --> J["Sent reconciles the actual worker"]
    K["Completed or Aborted"] --> L["Commit or roll back program state"]
    L --> M["Notify pending classifier futures"]
    M --> F
```

The catalog factory receives a cached view of Dynamo's existing discovery state. It derives each worker/rank's program-retention budget as `kv_cache_block_size * total_kv_blocks`, while representing worker liveness separately from missing capacity metadata. The callback only reads the cached view; it performs no discovery, model-card parsing, or blocking I/O on the classification path.

ThunderAgent computes live used capacity from its own program table. On each reconciliation, it accounts for active program tokens plus `buffer_per_program`. Acting programs use `acting_token_weight`. When a worker exceeds `pause_threshold`, the classifier pauses smaller acting programs first until usage reaches `pause_target`; in-flight reasoning programs are marked to pause after completion. Pausing clears the program assignment. Normal resume selects a fairness-ordered prefix against aggregate capacity, then uses largest-first best-fit-decreasing packing across workers. A request that waits longer than `resume_timeout_seconds` is assigned to the worker with the most capacity after decayed acting-token usage and force-released.

The request target is advisory rather than a reservation. Caller constraints and worker liveness remain authoritative, and `Sent` corrects the classifier's accounting if selection falls back.

Completion records Dynamo's terminal input-plus-output context size. The current classifier lifecycle does not expose cumulative streaming context progress, so this crate updates program size only at completion. A final session removes its program when the final request is admitted; completion or abort cannot restore it. A continuing idle program retains its observed assignment for `session_retention_seconds`. Idle retention is pruned lazily on the next classification or periodic reconciliation. The tracking limit also bounds retained programs; at the limit, the oldest idle retained program is evicted before admitting a new session.

## Configuration

Link this crate in place of Dynamo's default router-plugin catalog, enable soft Dynamo session affinity, and select ThunderAgent for both classification and aggregated worker selection:

```yaml
request_classifier:
  type: thunderagent
  parameters:
    pause_threshold: 0.95
    pause_target: 0.80
    resume_hysteresis: 0.10
    resume_timeout_seconds: 1800
    session_retention_seconds: 1800
    scheduler_interval_seconds: 5
    acting_token_weight: 1
    acting_decay_tau_seconds: 1
    buffer_per_program: 100
    max_tracked_requests: 10000
worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
      parameters: {}
```

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `pause_threshold` | `0.95` | Worker usage fraction that begins pausing programs |
| `pause_target` | `0.80` | Worker usage fraction that a pause cycle drains toward |
| `resume_hysteresis` | `0.10` | Headroom below the pause threshold required for normal resume |
| `resume_timeout_seconds` | `1800` | Maximum classifier deferral before forced release |
| `session_retention_seconds` | `1800` | Idle continuing-program retention period |
| `scheduler_interval_seconds` | `5` | Fixed cadence for pressure and resume decisions while programs are tracked |
| `acting_token_weight` | `1` | Capacity weight for a program during tool work |
| `acting_decay_tau_seconds` | `1` | Half-life in seconds for acting-token usage during timeout fallback placement |
| `buffer_per_program` | `100` | Fixed token headroom per active program |
| `max_tracked_requests` | `10000` | Independent bounds for active request state and retained program state before classification reaches Dynamo's queue |
