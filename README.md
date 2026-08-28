# Router sandbox

This Cargo workspace contains external routing strategies for Dynamo.

## Policy crates

- [`thunderagent-dynamo-policy`](crates/thunderagent/README.md) provides session-aware admission and worker selection.
- [`sticky-until-saturated-dynamo-policy`](crates/sticky-until-saturated/README.md) provides token-aware cache-affinity routing.
- [`sglang-cache-aware-dynamo-policy`](crates/sglang-cache-aware/README.md) ports SGLang's cache-aware load gate using Dynamo's authoritative device-KV overlap.
- [`sgl-router-cache-aware-dynamo-policy`](crates/sgl-router-cache-aware/README.md) ports the experimental SGL Router cache-aware policy used in the router comparison.
