//! Configuration management for the auto-batching proxy service.

use std::env;

/// Configuration settings for the batching proxy service.
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum time to wait before processing a batch (in milliseconds).
    /// This is a "soft" limit - batches may be processed earlier if size threshold is reached.
    pub soft_max_wait_time_ms: u64,

    /// Maximum number of items to include in a single batch.
    /// This is a "soft" limit - batches may exceed this size to keep requests together.
    pub soft_max_batch_size: usize,

    /// URL of the upstream inference service.
    pub inference_url: String,

    /// Maximum number of items that can be queued (10x `soft_max_batch_size`).
    /// Service returns 503 when this limit is exceeded.
    pub max_queue_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            soft_max_wait_time_ms: 100,
            soft_max_batch_size: 32,
            inference_url: "http://aubapr-inference:8080".to_string(),
            max_queue_size: 320, // 10x soft_max_batch_size
        }
    }
}

impl Config {
    /// Load configuration from environment variables with defaults.
    #[must_use]
    pub fn from_env() -> Self {
        let soft_max_batch_size = env::var("SOFT_MAX_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);

        Self {
            soft_max_wait_time_ms: env::var("SOFT_MAX_WAIT_TIME_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
            soft_max_batch_size,
            inference_url: env::var("INFERENCE_URL")
                .unwrap_or_else(|_| "http://aubapr-inference:8080".to_string()),
            max_queue_size: env::var("MAX_QUEUE_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(soft_max_batch_size * 10),
        }
    }

    /// Validate configuration values.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration values are invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.soft_max_wait_time_ms == 0 {
            return Err("soft_max_wait_time_ms must be greater than 0".to_string());
        }

        if self.soft_max_batch_size == 0 {
            return Err("soft_max_batch_size must be greater than 0".to_string());
        }

        if self.inference_url.is_empty() {
            return Err("inference_url cannot be empty".to_string());
        }

        if self.max_queue_size < self.soft_max_batch_size {
            return Err(
                "max_queue_size must be at least as large as soft_max_batch_size".to_string(),
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.soft_max_wait_time_ms, 100);
        assert_eq!(config.soft_max_batch_size, 32);
        assert_eq!(config.inference_url, "http://aubapr-inference:8080");
        assert_eq!(config.max_queue_size, 320);
    }

    #[test]
    fn test_config_validation_success() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_wait_time() {
        let config = Config {
            soft_max_wait_time_ms: 0,
            ..Config::default()
        };
        assert!(config.validate().is_err());
        assert!(config
            .validate()
            .unwrap_err()
            .contains("soft_max_wait_time_ms"));
    }

    #[test]
    fn test_config_validation_zero_batch_size() {
        let config = Config {
            soft_max_batch_size: 0,
            ..Config::default()
        };
        assert!(config.validate().is_err());
        assert!(config
            .validate()
            .unwrap_err()
            .contains("soft_max_batch_size"));
    }

    #[test]
    fn test_config_validation_empty_url() {
        let config = Config {
            inference_url: String::new(),
            ..Config::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("inference_url"));
    }

    #[test]
    fn test_config_validation_small_queue_size() {
        let config = Config {
            soft_max_batch_size: 32,
            max_queue_size: 16,
            ..Config::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("max_queue_size"));
    }

    #[test]
    fn test_config_from_env_defaults() {
        // Clear any existing env vars that might interfere
        env::remove_var("SOFT_MAX_WAIT_TIME_MS");
        env::remove_var("SOFT_MAX_BATCH_SIZE");
        env::remove_var("INFERENCE_URL");
        env::remove_var("MAX_QUEUE_SIZE");

        let config = Config::from_env();
        assert_eq!(config.soft_max_wait_time_ms, 100);
        assert_eq!(config.soft_max_batch_size, 32);
        assert_eq!(config.inference_url, "http://aubapr-inference:8080");
        assert_eq!(config.max_queue_size, 320);
    }

    #[test]
    fn test_config_from_env_with_values() {
        env::set_var("SOFT_MAX_WAIT_TIME_MS", "200");
        env::set_var("SOFT_MAX_BATCH_SIZE", "64");
        env::set_var("INFERENCE_URL", "http://custom:9000");
        env::set_var("MAX_QUEUE_SIZE", "1000");

        let config = Config::from_env();
        assert_eq!(config.soft_max_wait_time_ms, 200);
        assert_eq!(config.soft_max_batch_size, 64);
        assert_eq!(config.inference_url, "http://custom:9000");
        assert_eq!(config.max_queue_size, 1000);

        // Clean up
        env::remove_var("SOFT_MAX_WAIT_TIME_MS");
        env::remove_var("SOFT_MAX_BATCH_SIZE");
        env::remove_var("INFERENCE_URL");
        env::remove_var("MAX_QUEUE_SIZE");
    }

    #[test]
    fn test_config_from_env_invalid_numbers() {
        env::set_var("SOFT_MAX_WAIT_TIME_MS", "invalid");
        env::set_var("SOFT_MAX_BATCH_SIZE", "not_a_number");

        let config = Config::from_env();
        // Should fall back to defaults for invalid values
        assert_eq!(config.soft_max_wait_time_ms, 100);
        assert_eq!(config.soft_max_batch_size, 32);

        env::remove_var("SOFT_MAX_WAIT_TIME_MS");
        env::remove_var("SOFT_MAX_BATCH_SIZE");
    }
}
