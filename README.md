# Router sandbox

This Cargo workspace contains external routing strategies for Dynamo.

## Policy crates

- [`thunderagent-dynamo-policy`](crates/thunderagent/README.md) provides session-aware admission and worker selection.
- [`sticky-until-saturated-dynamo-policy`](crates/sticky-until-saturated/README.md) provides token-aware cache-affinity routing.
