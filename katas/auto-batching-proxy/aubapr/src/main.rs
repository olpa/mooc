//! Auto-batching proxy service for ML inference.
//!
//! This service acts as a proxy between clients and an ML inference service,
//! providing health checks and request forwarding capabilities.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

/// Request structure for embedding generation.
#[derive(Debug, Serialize, Deserialize)]
struct EmbedRequest {
    /// List of input strings to generate embeddings for.
    inputs: Vec<String>,
}

/// Response structure containing generated embeddings.
#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    /// Generated embeddings as vectors of floating-point numbers.
    embeddings: Vec<Vec<f64>>,
}

/// Application state shared across handlers.
#[derive(Clone)]
struct AppState {
    /// HTTP client for making requests to the inference service.
    client: Client,
    /// URL of the inference service.
    inference_url: String,
}

/// Handles embedding requests by forwarding them to the inference service.
///
/// # Errors
///
/// Returns an error response with the same status code and body as the upstream service
/// if the request fails or the upstream service returns an error.
async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!(
        "Received embed request with {} inputs",
        request.inputs.len()
    );

    // Forward request to inference service
    let response = state
        .client
        .post(format!("{}/embed", state.inference_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send request to inference service: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Failed to send request to inference service"})),
            )
        })?;

    if !response.status().is_success() {
        let status = response.status();
        error!("Inference service returned error: {}", status);

        // Try to get the response body as text first
        let response_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Upstream service error".to_string());

        // Try to parse as JSON, fallback to wrapping text in error object
        let error_body = serde_json::from_str::<serde_json::Value>(&response_text)
            .unwrap_or_else(|_| serde_json::json!({"error": response_text}));

        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(error_body),
        ));
    }

    let embed_response: EmbedResponse = response.json().await.map_err(|e| {
        error!("Failed to parse response from inference service: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Failed to parse response from inference service"})),
        )
    })?;

    info!("Successfully processed embed request");
    Ok(Json(embed_response))
}

/// Health check endpoint that returns service status.
async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let inference_url = std::env::var("INFERENCE_URL")
        .unwrap_or_else(|_| "http://aubapr-inference:8080".to_string());

    info!("Starting aubapr proxy service");
    info!("Inference service URL: {}", inference_url);

    let state = Arc::new(AppState {
        client: Client::new(),
        inference_url,
    });

    let app = Router::new()
        .route("/embed", post(embed_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;

    info!("Server listening on 0.0.0.0:8080");

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn create_app_state(inference_url: String) -> Arc<AppState> {
        Arc::new(AppState {
            client: Client::new(),
            inference_url,
        })
    }

    fn create_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/embed", post(embed_handler))
            .route("/health", get(health_handler))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state);
        let server = TestServer::new(app).unwrap();

        let response = server.get("/health").await;

        response.assert_status_ok();
        response.assert_json(&json!({"status": "healthy"}));
    }

    #[tokio::test]
    async fn test_single_input_proxy() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": [[0.1, 0.2, 0.3]]
            })))
            .mount(&mock_server)
            .await;

        let state = create_app_state(mock_server.uri());
        let app = create_app(state);
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status_ok();
        response.assert_json(&json!({
            "embeddings": [[0.1, 0.2, 0.3]]
        }));
    }

    #[tokio::test]
    async fn test_upstream_error_handling() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(422).set_body_json(json!({
                "error": "Validation error"
            })))
            .mount(&mock_server)
            .await;

        let state = create_app_state(mock_server.uri());
        let app = create_app(state);
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
        response.assert_json(&json!({
            "error": "Validation error"
        }));
    }

    #[tokio::test]
    async fn test_upstream_non_json_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(422).set_body_string("Invalid input format"))
            .mount(&mock_server)
            .await;

        let state = create_app_state(mock_server.uri());
        let app = create_app(state);
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
        response.assert_json(&json!({
            "error": "Invalid input format"
        }));
    }
}
