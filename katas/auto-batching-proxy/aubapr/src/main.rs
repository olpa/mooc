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

#[derive(Debug, Serialize, Deserialize)]
struct EmbedRequest {
    inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct AppState {
    client: Client,
    inference_url: String,
}

async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, StatusCode> {
    info!("Received embed request with {} inputs", request.inputs.len());

    // Forward request to inference service
    let response = state
        .client
        .post(&format!("{}/embed", state.inference_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send request to inference service: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !response.status().is_success() {
        error!("Inference service returned error: {}", response.status());
        return Err(StatusCode::BAD_GATEWAY);
    }

    let embed_response: EmbedResponse = response.json().await.map_err(|e| {
        error!("Failed to parse response from inference service: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    info!("Successfully processed embed request");
    Ok(Json(embed_response))
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

#[tokio::main]
async fn main() {
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

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("Failed to bind to port 8080");

    info!("Server listening on 0.0.0.0:8080");

    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}