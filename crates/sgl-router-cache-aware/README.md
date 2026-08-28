# Experimental SGL Router cache-aware policy

`sgl-router-cache-aware-dynamo-policy` ports the experimental SGL Router's `cache_aware_zmq` policy to Dynamo's worker-selection plugin API. It is separate from `sglang-cache-aware-dynamo-policy`, which implements the older SGLang Model Gateway policy.

## Algorithm

For each Dynamo-eligible worker, the policy reads device-KV overlap and active-request count.

1. If active-request spread is greater than 32 and the largest count is more than 1.1 times the smallest, select the least-loaded worker.
2. Otherwise, find the largest device-KV overlap. If it is strictly more than 50% of the request's block count, select the least-loaded worker among workers with that exact maximum overlap.
3. Otherwise, select the least-loaded worker.

The thresholds and selection order are the exact implementation benchmarked with the MiniMax AgentX comparison. Dynamo provides authoritative device-KV overlap, while the raw experimental router uses its own cache history.

## Configure Dynamo

Use [`worker-selection.yaml`](worker-selection.yaml) as the router policy configuration. The policy has no parameters: its constants intentionally match the experimental router.

## Validate

```bash
cargo test -p sgl-router-cache-aware-dynamo-policy
cargo clippy -p sgl-router-cache-aware-dynamo-policy -- -D warnings
cargo fmt --check
```
