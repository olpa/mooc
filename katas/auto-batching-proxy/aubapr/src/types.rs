//! Core types for batch processing functionality.

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

/// Type alias for a batch of strings (text inputs).
pub type BatchOfStrings = Vec<String>;

/// Type alias for a batch of embeddings (vectors of f64).
pub type BatchOfEmbeddings = Vec<Vec<f64>>;

/// Request structure for embedding generation.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// List of input strings to generate embeddings for.
    pub inputs: BatchOfStrings,
}

/// Response structure containing generated embeddings.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// Generated embeddings as vectors of floating-point numbers.
    pub embeddings: BatchOfEmbeddings,
}

/// A single item in a batch waiting to be processed.
#[derive(Debug)]
pub struct BatchItem {
    /// The text to embed.
    pub text: String,
    /// Channel sender to return the result to the original request (only first item has this).
    pub sender: Option<tokio::sync::oneshot::Sender<BatchResult>>,
}

/// Result type for batch processing operations (multiple embeddings).
pub type BatchResult = Result<BatchOfEmbeddings, BatchError>;

/// Errors that can occur during batch processing.
#[derive(Debug, Clone)]
pub enum BatchError {
    /// Error from upstream inference service.
    UpstreamError(StatusCode, String),
}

impl BatchError {
    /// Convert `BatchError` to HTTP status code for API responses.
    #[must_use]
    pub const fn to_status_code(&self) -> StatusCode {
        match self {
            Self::UpstreamError(status, _) => *status,
        }
    }

    /// Get error message for API responses.
    #[must_use]
    pub fn to_message(&self) -> String {
        match self {
            Self::UpstreamError(_, msg) => msg.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_item_creation() {
        let item = BatchItem {
            text: "test text".to_string(),
            sender: None,
        };

        assert_eq!(item.text, "test text");
        assert!(item.sender.is_none());
    }

    #[test]
    fn test_batch_error_status_codes() {
        let upstream_error =
            BatchError::UpstreamError(StatusCode::BAD_REQUEST, "Bad input".to_string());
        assert_eq!(upstream_error.to_status_code(), StatusCode::BAD_REQUEST);

    }

    #[test]
    fn test_batch_error_messages() {
        let upstream_error =
            BatchError::UpstreamError(StatusCode::BAD_REQUEST, "Custom error".to_string());
        assert_eq!(upstream_error.to_message(), "Custom error");

    }
}
