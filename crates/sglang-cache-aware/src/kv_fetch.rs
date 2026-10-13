// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental gate on the router's KV fetch hint.
//!
//! When workers consume KV hints, Dynamo lets the selected worker fetch a longer cached prefix
//! from another worker. The load gate is where this matters, because it moves a request away from
//! the worker that holds its prefix. This gate skips fetches that add too few blocks to pay for
//! the transfer.

use dynamo_kv_router::plugins::worker_selection::experimental::kv_hints::{
    KvTransferAction, KvTransferInput, KvTransferPolicy,
};

/// Skips the router's KV fetch hint when the fetch adds fewer than `min_blocks` blocks.
pub(crate) struct MinKvFetch {
    pub(crate) min_blocks: u32,
}

impl KvTransferPolicy for MinKvFetch {
    fn decide(&mut self, input: KvTransferInput<'_>) -> KvTransferAction {
        kv_fetch_action(self.min_blocks, input.additional_blocks())
    }
}

/// Keep the router's default source when the fetch adds at least `min_blocks` blocks.
fn kv_fetch_action(min_blocks: u32, additional_blocks: u32) -> KvTransferAction {
    if additional_blocks >= min_blocks {
        KvTransferAction::Default
    } else {
        KvTransferAction::Skip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetches_only_at_or_above_the_minimum() {
        assert_eq!(kv_fetch_action(0, 1), KvTransferAction::Default);
        assert_eq!(kv_fetch_action(64, 64), KvTransferAction::Default);
        assert_eq!(kv_fetch_action(64, 63), KvTransferAction::Skip);
        assert_eq!(kv_fetch_action(u32::MAX, 100_000), KvTransferAction::Skip);
    }
}
