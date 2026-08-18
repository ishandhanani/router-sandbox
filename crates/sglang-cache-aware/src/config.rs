// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use thiserror::Error;

pub const DEFAULT_CACHE_THRESHOLD: f64 = 0.3;
pub const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 64;
pub const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.5;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid sglang-cache-aware configuration: {0}")]
    Invalid(&'static str),
}

/// Parameters for SGLang's cache-aware load-gated worker selection.
///
/// The defaults match SGLang Model Gateway's public CLI and Python bindings.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SglangCacheAwareConfig {
    /// Strict device-cache overlap fraction required to prefer cache affinity.
    pub cache_threshold: f64,
    /// Strict active-request-count gap required for load balancing.
    pub balance_abs_threshold: usize,
    /// Strict active-request-count ratio required for load balancing.
    pub balance_rel_threshold: f64,
}

impl Default for SglangCacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
        }
    }
}

impl SglangCacheAwareConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.cache_threshold.is_finite() || !(0.0..=1.0).contains(&self.cache_threshold) {
            return Err(ConfigError::Invalid(
                "cache_threshold must be finite and in [0, 1]",
            ));
        }
        if !self.balance_rel_threshold.is_finite() || self.balance_rel_threshold < 1.0 {
            return Err(ConfigError::Invalid(
                "balance_rel_threshold must be finite and greater than or equal to 1",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_defaults_and_rejects_invalid_values() {
        assert!(SglangCacheAwareConfig::default().validate().is_ok());
        assert!(
            SglangCacheAwareConfig {
                cache_threshold: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SglangCacheAwareConfig {
                cache_threshold: 1.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SglangCacheAwareConfig {
                balance_rel_threshold: f64::INFINITY,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SglangCacheAwareConfig {
                balance_rel_threshold: 0.99,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
