// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use serde::Deserialize;
use thiserror::Error;

pub const DEFAULT_SLACK_REQUESTS: usize = 4;
pub const DEFAULT_MAX_SESSIONS: usize = 65_536;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid session-affinity configuration: {0}")]
    Invalid(&'static str),
}

/// Parameters for the session-affinity worker-selection policy.
///
/// `slack_requests` is the load gate: the remembered worker is kept only while
/// its in-flight request count is at most `slack_requests` above the
/// least-loaded eligible worker. `0` keeps a session on its worker only while
/// that worker is the least loaded; a very large value never migrates.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionAffinityConfig {
    pub slack_requests: usize,
    pub max_sessions: usize,
}

impl Default for SessionAffinityConfig {
    fn default() -> Self {
        Self {
            slack_requests: DEFAULT_SLACK_REQUESTS,
            max_sessions: DEFAULT_MAX_SESSIONS,
        }
    }
}

impl SessionAffinityConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_sessions == 0 {
            return Err(ConfigError::Invalid("max_sessions must be at least 1"));
        }
        Ok(())
    }

    pub(crate) fn max_sessions(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.max_sessions).unwrap_or(NonZeroUsize::MIN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_defaults_and_rejects_zero_capacity() {
        assert!(SessionAffinityConfig::default().validate().is_ok());
        assert!(
            SessionAffinityConfig {
                max_sessions: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SessionAffinityConfig {
                slack_requests: 0,
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
    }
}
