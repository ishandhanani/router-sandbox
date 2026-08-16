# Router sandbox

This Cargo workspace contains external routing strategies for Dynamo.

## ThunderAgent

The [`thunderagent-dynamo-policy`](crates/thunderagent) crate keeps ThunderAgent's session and capacity state outside Dynamo. Dynamo owns request storage and calls the policy serially from its queue actor.

```rust
use std::sync::Arc;

use dynamo_kv_router::WorkerSelectionPolicyFactory;
use thunderagent_dynamo_policy::{ThunderAgentConfig, worker_selection_policy};

let config = ThunderAgentConfig::default();
let factory: WorkerSelectionPolicyFactory = Arc::new(move |router, worker_type, _partition| {
    worker_selection_policy(router.clone(), worker_type, config.clone())
        .expect("valid ThunderAgent configuration")
});
```

The returned `WorkerSelectionPolicy` contains all three external components:

- A `QueueAdmissionPolicy` that implements session fairness by controlling which requests become runnable.
- A scorer used only when no session assignment exists.
- A picker that honors the admission policy's session assignment when that worker remains eligible.

Requests without Dynamo session context bypass ThunderAgent admission and still use the scorer and picker.

The crate also exposes `register`, so it can be linked directly as Dynamo's `dynamo-worker-selection-policy-catalog` dependency. The registered policy type is `thunderagent`; its YAML parameters are the fields of `ThunderAgentConfig`.

For isolated cache-storage validation, the catalog also exposes `thunderagent-storage-handoff-experiment`. It keeps ThunderAgent admission unchanged. A tool-result request stays on the assigned worker and emits `kv.demote`; the next `Other` request moves to an eligible alternate worker and emits `kv.prefetch`. This policy type is for an end-to-end storage-hint run only.

[`worker-selection.yaml`](crates/thunderagent/worker-selection.yaml) is a complete policy configuration using the default ThunderAgent values.

Runnable ordering remains Dynamo-owned. This prototype depends on the Dynamo queue-admission seam in [ai-dynamo/dynamo#13019](https://github.com/ai-dynamo/dynamo/pull/13019).
