//! Core types for batch processing functionality.

use axum::http::StatusCode;
use tokio::time::Instant;
use tokio::sync::oneshot;
use uuid::Uuid;

/// A single item in a batch waiting to be processed.
#[derive(Debug)]
pub struct BatchItem {
    /// The text to embed.
    pub text: String,
    /// Unique identifier for the original request this item belongs to.
    pub request_id: Uuid,
    /// Channel sender to return the result to the original request.
    pub sender: oneshot::Sender<BatchResult>,
    /// When this item was added to the batch queue.
    pub timestamp: Instant,
}

/// Result type for batch processing operations.
pub type BatchResult = Result<Vec<f64>, BatchError>;

/// Errors that can occur during batch processing.
#[derive(Debug, Clone)]
pub enum BatchError {
    /// Error from upstream inference service.
    UpstreamError(StatusCode, String),
    /// Request timed out waiting for batch processing.
    Timeout,
    /// Service is temporarily unavailable due to overload.
    ServiceUnavailable,
}

impl BatchError {
    /// Convert `BatchError` to HTTP status code for API responses.
    #[must_use]
    pub const fn to_status_code(&self) -> StatusCode {
        match self {
            Self::UpstreamError(status, _) => *status,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::ServiceUnavailable => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// Get error message for API responses.
    #[must_use]
    pub fn to_message(&self) -> String {
        match self {
            Self::UpstreamError(_, msg) => msg.clone(),
            Self::Timeout => "Request timeout waiting for batch processing".to_string(),
            Self::ServiceUnavailable => {
                "Service temporarily unavailable due to overload".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_item_creation() {
        let (tx, _rx) = oneshot::channel();
        let item = BatchItem {
            text: "test text".to_string(),
            request_id: Uuid::new_v4(),
            sender: tx,
            timestamp: Instant::now(),
        };

        assert_eq!(item.text, "test text");
        assert!(item.request_id.to_string().len() > 0);
    }

    #[test]
    fn test_batch_error_status_codes() {
        let upstream_error =
            BatchError::UpstreamError(StatusCode::BAD_REQUEST, "Bad input".to_string());
        assert_eq!(upstream_error.to_status_code(), StatusCode::BAD_REQUEST);

        let timeout_error = BatchError::Timeout;
        assert_eq!(timeout_error.to_status_code(), StatusCode::REQUEST_TIMEOUT);

        let unavailable_error = BatchError::ServiceUnavailable;
        assert_eq!(
            unavailable_error.to_status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_batch_error_messages() {
        let upstream_error =
            BatchError::UpstreamError(StatusCode::BAD_REQUEST, "Custom error".to_string());
        assert_eq!(upstream_error.to_message(), "Custom error");

        let timeout_error = BatchError::Timeout;
        assert!(timeout_error.to_message().contains("timeout"));

        let unavailable_error = BatchError::ServiceUnavailable;
        assert!(unavailable_error.to_message().contains("unavailable"));
    }
}
