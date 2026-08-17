// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use criterion::{Criterion, criterion_group, criterion_main};
use sticky_until_saturated_dynamo_policy::{
    StickyUntilSaturatedConfig,
    benchmark::{BenchmarkCandidate, benchmark_prefill_pick_index},
};

const WORKERS: usize = 10_000;
const REQUEST_BLOCKS: u64 = 256;
const BLOCK_SIZE: u32 = 16;

fn candidates(warm_every: Option<usize>) -> Vec<BenchmarkCandidate> {
    (0..WORKERS)
        .map(|worker| BenchmarkCandidate {
            worker_key: worker as u64,
            device_overlap_blocks: if warm_every.is_some_and(|every| worker % every == 0) {
                REQUEST_BLOCKS as f64
            } else {
                0.0
            },
            active_prefill_tokens: worker % 1024,
        })
        .collect()
}

fn bench_prefill_pick(criterion: &mut Criterion) {
    let config = StickyUntilSaturatedConfig::default();
    let cold_start = candidates(None);
    let warm_set = candidates(Some(4));
    let mut group = criterion.benchmark_group("sticky_until_saturated_prefill_10k");
    group.bench_function("cold_start", |bencher| {
        bencher.iter(|| {
            criterion::black_box(benchmark_prefill_pick_index(
                criterion::black_box(&config),
                criterion::black_box(REQUEST_BLOCKS),
                criterion::black_box(BLOCK_SIZE),
                criterion::black_box(&cold_start),
            ))
        });
    });
    group.bench_function("warm_set", |bencher| {
        bencher.iter(|| {
            criterion::black_box(benchmark_prefill_pick_index(
                criterion::black_box(&config),
                criterion::black_box(REQUEST_BLOCKS),
                criterion::black_box(BLOCK_SIZE),
                criterion::black_box(&warm_set),
            ))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_prefill_pick);
criterion_main!(benches);
