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

[`worker-selection.yaml`](crates/thunderagent/worker-selection.yaml) is a complete policy configuration using the default ThunderAgent values.

Runnable ordering remains Dynamo-owned. This prototype depends on the Dynamo queue-admission seam in [ai-dynamo/dynamo#13019](https://github.com/ai-dynamo/dynamo/pull/13019).

## Sticky until saturated

The [`sticky-until-saturated-dynamo-policy`](crates/sticky-until-saturated) crate implements the [llm-d token-aware routing strategy](https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing). It is a linkable policy catalog: use it directly as Dynamo's `dynamo-worker-selection-policy-catalog` dependency, or call its `register` function from a combined catalog.

For aggregated and prefill pools, a worker is warm when its device-resident prefix overlap is at least `affinity_threshold` of the request. The picker stays within the warm set unless a cold worker has a projected prefill load lower than the best warm worker by more than `peak_prefill_tokens_per_second * max_ttft_penalty_ms / 1000`. When no worker is warm, it picks from every eligible worker. Decode pools use active requests as their primary ordering signal but retain the same prefill-load saturation guard.

`worker-selection.yaml` contains the article's defaults: 80% affinity, 15,928 prefill tokens/s, and an 18,000 ms TTFT budget. The first two pool types require Dynamo prefill-token tracking; leave `--router-track-prefill-tokens` enabled.

This first version intentionally treats only device-resident blocks as warm. Dynamo's host and disk tiers have different retrieval costs and weighting, so folding them into the binary warm-set condition would need a separate policy contract.

### Mocker smoke test

Build the Python extension against the same Dynamo revision that this crate uses, replacing Dynamo's empty catalog with this package:

```bash
cargo add --manifest-path /path/to/dynamo/lib/bindings/python/Cargo.toml \
  --optional --rename dynamo-worker-selection-policy-catalog \
  --path /path/to/router-sandbox/crates/sticky-until-saturated \
  sticky-until-saturated-dynamo-policy

cd /path/to/dynamo/lib/bindings/python
CARGO_TARGET_DIR=/path/to/dynamo/target maturin develop --uv --features custom-policy
```

Then run the frontend and two Mockers with the included configuration:

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=sticky-until-saturated \
python -m dynamo.frontend --router-mode kv \
  --router-policy-config /path/to/router-sandbox/crates/sticky-until-saturated/worker-selection.yaml \
  --discovery-backend file

python -m dynamo.mocker --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file --num-workers 2
```
