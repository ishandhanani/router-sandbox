// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use thiserror::Error;

pub const DEFAULT_AFFINITY_THRESHOLD: f64 = 0.8;
pub const DEFAULT_PEAK_PREFILL_TOKENS_PER_SECOND: f64 = 15_928.0;
pub const DEFAULT_MAX_TTFT_PENALTY_MS: u64 = 18_000;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid sticky-until-saturated configuration: {0}")]
    Invalid(&'static str),
}

/// Parameters for the sticky-until-saturated worker-selection policy.
///
/// `peak_prefill_tokens_per_second * max_ttft_penalty_ms / 1000` is the
/// affinity-break margin, in tokens.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StickyUntilSaturatedConfig {
    pub affinity_threshold: f64,
    pub peak_prefill_tokens_per_second: f64,
    pub max_ttft_penalty_ms: u64,
}

impl Default for StickyUntilSaturatedConfig {
    fn default() -> Self {
        Self {
            affinity_threshold: DEFAULT_AFFINITY_THRESHOLD,
            peak_prefill_tokens_per_second: DEFAULT_PEAK_PREFILL_TOKENS_PER_SECOND,
            max_ttft_penalty_ms: DEFAULT_MAX_TTFT_PENALTY_MS,
        }
    }
}

impl StickyUntilSaturatedConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.affinity_threshold.is_finite() || !(0.0..=1.0).contains(&self.affinity_threshold) {
            return Err(ConfigError::Invalid(
                "affinity_threshold must be finite and in [0, 1]",
            ));
        }
        if !self.peak_prefill_tokens_per_second.is_finite()
            || self.peak_prefill_tokens_per_second <= 0.0
        {
            return Err(ConfigError::Invalid(
                "peak_prefill_tokens_per_second must be finite and positive",
            ));
        }
        if !self.saturation_tokens().is_finite() {
            return Err(ConfigError::Invalid(
                "peak_prefill_tokens_per_second * max_ttft_penalty_ms must be finite",
            ));
        }
        Ok(())
    }

    pub(crate) fn saturation_tokens(&self) -> f64 {
        self.peak_prefill_tokens_per_second * self.max_ttft_penalty_ms as f64 / 1_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_defaults_and_rejects_invalid_values() {
        assert!(StickyUntilSaturatedConfig::default().validate().is_ok());
        assert!(
            StickyUntilSaturatedConfig {
                affinity_threshold: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StickyUntilSaturatedConfig {
                affinity_threshold: 1.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StickyUntilSaturatedConfig {
                peak_prefill_tokens_per_second: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StickyUntilSaturatedConfig {
                max_ttft_penalty_ms: 0,
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            StickyUntilSaturatedConfig {
                peak_prefill_tokens_per_second: f64::MAX,
                max_ttft_penalty_ms: u64::MAX,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
