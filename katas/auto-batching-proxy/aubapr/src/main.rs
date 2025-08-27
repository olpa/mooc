//! Auto-batching proxy service for ML inference.
//!
//! This service acts as a proxy between clients and an ML inference service,
//! providing health checks and request forwarding capabilities.

mod batcher;
mod config;
mod timer;
mod tray;
mod types;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use batcher::Batcher;
use config::Config;
use reqwest::Client;
use std::sync::Arc;
use tokio::signal;
use tracing::{debug, error, info};
use tracing_subscriber::{filter::LevelFilter, EnvFilter};
use types::{EmbedRequest, EmbedResponse};

/// Application state shared across handlers.
#[derive(Clone)]
struct AppState {
    /// Batching coordinator for request processing.
    batcher: Arc<Batcher>,
}

/// Handles embedding requests by submitting them to the batcher.
///
/// # Errors
///
/// Returns an error response based on batch processing results or service availability.
async fn embed_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!(
        "Received embed request with {} inputs",
        request.inputs.len()
    );

    // Submit request to batcher
    let rx = state.batcher.submit_request(request.inputs).await;
    
    // Wait for the batch processing result
    if let Ok(batch_result) = rx.await {
        match batch_result {
            Ok(embeddings) => {
                debug!("Successfully processed embed request");
                Ok(Json(EmbedResponse { embeddings }))
            }
            Err(batch_error) => {
                let status = batch_error.to_status_code();
                let message = batch_error.to_message();

                error!("Batch processing failed: {}", message);

                // Try to parse the error message as JSON, fallback to wrapping it
                let error_body = serde_json::from_str::<serde_json::Value>(&message)
                    .unwrap_or_else(|_| serde_json::json!({"error": message}));

                Err((status, Json(error_body)))
            }
        }
    } else {
        error!("Channel closed while waiting for batch result");
        Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Internal server error"})),
        ))
    }
}

/// Health check endpoint that returns service status.
async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let log_level = std::env::var("AUBAPR_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .parse_lossy(&log_level);
    
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    // Load configuration
    let config = Config::from_env();
    config
        .validate()
        .map_err(|e| format!("Invalid configuration: {e}"))?;

    info!("Starting aubapr proxy service");
    info!("Configuration: {:?}", config);

    // Create batcher
    let client = Client::new();
    let (batcher, channels) = Batcher::new(config, client);
    let batcher = Arc::new(batcher);

    // Clone batcher before consuming it
    let batcher_for_state = batcher.clone();
    let batcher_for_shutdown = batcher.clone();

    // Start background processor (consumes batcher)
    batcher.spawn_background_processor(channels).await;

    // Create application state
    let state = Arc::new(AppState {
        batcher: batcher_for_state,
    });

    let app = Router::new()
        .route("/embed", post(embed_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;

    info!("Server listening on 0.0.0.0:8080");

    // Run server with graceful shutdown
    let server = axum::serve(listener, app);

    tokio::select! {
        result = server => {
            if let Err(e) = result {
                error!("Server error: {}", e);
            }
        }
        _ = signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    // Graceful shutdown
    info!("Shutting down gracefully...");
    if let Err(e) = batcher_for_shutdown.shutdown().await {
        error!("Failed to shutdown gracefully: {:?}", e);
    }

    info!("Shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use types::BatchOfEmbeddings;

    /// Initialize debug logging for tests
    #[cfg(test)]
    fn init_test_logging() {
        let log_level = std::env::var("AUBAPR_LOG_LEVEL").unwrap_or_else(|_| "debug".to_string());
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::DEBUG.into())
            .parse_lossy(&log_level);
        
        if let Err(e) = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .try_init()
        {
            eprintln!("Failed to initialize tracing subscriber: {}", e);
        }
    }

    /// Helper function to generate embeddings for test stub.
    /// Creates 16-dimensional vectors from input strings using ASCII codes.
    #[cfg(test)]
    fn generate_test_embedding(input: &str) -> Vec<f64> {
        let mut embedding = vec![0.0; 16];
        let chars: Vec<char> = input.chars().take(16).collect();

        for (i, &ch) in chars.iter().enumerate() {
            embedding[i] = ch as u8 as f64;
        }

        embedding
    }

    /// Helper function to create dynamic mock responses using generate_test_embedding.
    /// Parses the incoming request and generates embeddings for all inputs.
    #[cfg(test)]
    fn create_dynamic_embedding_response(req: &wiremock::Request) -> ResponseTemplate {
        // Parse the incoming request to get the inputs
        let body: serde_json::Value = req.body_json().expect("Request should have JSON body");
        let inputs = body["inputs"].as_array().expect("Should have inputs array");

        // Generate embeddings dynamically for each input
        let embeddings: BatchOfEmbeddings = inputs
            .iter()
            .map(|input| {
                let text = input.as_str().expect("Input should be string");
                generate_test_embedding(text)
            })
            .collect();

        // Return the dynamic response
        ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": embeddings
        }))
    }

    #[test]
    fn test_generate_test_embedding() {
        // Test with short string
        let embedding = generate_test_embedding("hi");
        assert_eq!(embedding.len(), 16);
        assert_eq!(embedding[0], 'h' as u8 as f64); // 104.0
        assert_eq!(embedding[1], 'i' as u8 as f64); // 105.0
        assert_eq!(embedding[2], 0.0); // padding

        // Test with exactly 16 characters
        let embedding = generate_test_embedding("1234567890123456");
        assert_eq!(embedding.len(), 16);
        assert_eq!(embedding[0], '1' as u8 as f64); // 49.0
        assert_eq!(embedding[15], '6' as u8 as f64); // 54.0

        // Test with more than 16 characters (should be truncated)
        let test_str = "this is a very long string that exceeds 16 chars";
        let first_16: String = test_str.chars().take(16).collect();
        let embedding = generate_test_embedding(test_str);
        assert_eq!(embedding.len(), 16);
        assert_eq!(embedding[0], 't' as u8 as f64); // 116.0
        assert_eq!(
            embedding[15],
            first_16.chars().nth(15).unwrap() as u8 as f64
        ); // 16th char
    }


    fn create_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/embed", post(embed_handler))
            .route("/health", get(health_handler))
            .with_state(state)
    }


    async fn setup_embed_test_with_batch_size_and_timeout(batch_size: usize, timeout_ms: u64) -> (MockServer, TestServer, Arc<AppState>) {
        init_test_logging();
        let mock_server = MockServer::start().await;

        // Set up dynamic mock that generates embeddings using generate_test_embedding
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(create_dynamic_embedding_response)
            .mount(&mock_server)
            .await;

        // Create state with proper batcher setup
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        config.soft_max_batch_size = batch_size;
        config.soft_max_wait_time_ms = timeout_ms;
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);
        
        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        (mock_server, server, state)
    }


    #[tokio::test]
    async fn test_health_endpoint() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server.get("/health").await;

        response.assert_status_ok();
        response.assert_json(&json!({"status": "healthy"}));
    }

    #[tokio::test]
    async fn test_one_client_one_input() {
        let (mock_server, server, state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let test_input = "test string";

        let response = server
            .post("/embed")
            .json(&json!({"inputs": [test_input]}))
            .await;

        response.assert_status_ok();

        // Use literal pre-calculated embedding for assertion
        // Expected embedding for "test string": t=116, e=101, s=115, t=116, (space)=32, s=115, t=116, r=114, i=105, n=110, g=103, then zeros
        let expected_embedding = vec![
            116.0, 101.0, 115.0, 116.0, 32.0, 115.0, 116.0, 114.0, 105.0, 110.0, 103.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];

        response.assert_json(&json!({
            "embeddings": [expected_embedding]
        }));

        // Verify mock server received exactly one request
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 1);

        // Verify the request body was as expected
        let request = &received_requests[0];
        let body: serde_json::Value = request.body_json().expect("Request should have JSON body");
        let expected_body = json!({"inputs": [test_input]});
        assert_eq!(body, expected_body);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_one_client_multiple_inputs() {
        let (mock_server, server, state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let test_inputs = vec!["hello", "world", "test"];

        let response = server
            .post("/embed")
            .json(&json!({"inputs": test_inputs.clone()}))
            .await;

        response.assert_status_ok();

        // Use literal pre-calculated embeddings for assertion
        let expected_embeddings = vec![
            vec![
                104.0, 101.0, 108.0, 108.0, 111.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ], // "hello"
            vec![
                119.0, 111.0, 114.0, 108.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ], // "world"
            vec![
                116.0, 101.0, 115.0, 116.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ], // "test"
        ];

        response.assert_json(&json!({
            "embeddings": expected_embeddings
        }));

        // Verify mock server received exactly one request
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 1);

        // Verify the request body was as expected
        let request = &received_requests[0];
        let body: serde_json::Value = request.body_json().expect("Request should have JSON body");
        let expected_body = json!({"inputs": test_inputs});
        assert_eq!(body, expected_body);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_multiple_clients_multiple_inputs() {
        // Client 1 inputs: "hi", "bye"
        let client1_inputs = vec!["hi", "bye"];
        // Client 2 inputs: "foo", "bar", "baz"
        let client2_inputs = vec!["foo", "bar", "baz"];
        // Client 3 inputs: "x", "y"
        let client3_inputs = vec!["x", "y"];

        // Total inputs: 2 + 3 + 2 = 7
        let total_inputs = client1_inputs.len() + client2_inputs.len() + client3_inputs.len();
        let (mock_server, server, state) = setup_embed_test_with_batch_size_and_timeout(total_inputs, 3_600_000).await;

        // Make concurrent requests from multiple clients
        let client1_request = server
            .post("/embed")
            .json(&json!({"inputs": client1_inputs.clone()}));

        let client2_request = server
            .post("/embed")
            .json(&json!({"inputs": client2_inputs.clone()}));

        let client3_request = server
            .post("/embed")
            .json(&json!({"inputs": client3_inputs.clone()}));

        // Execute all requests concurrently with timeout
        let (response1, response2, response3) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async { tokio::join!(client1_request, client2_request, client3_request) }
        )
        .await
        .expect("All requests should complete within timeout - batching may not be working correctly");

        // Verify all responses are successful
        response1.assert_status_ok();
        response2.assert_status_ok();
        response3.assert_status_ok();

        // Verify each client gets their correct embeddings back (using literal values)
        response1.assert_json(&json!({
            "embeddings": [
                vec![104.0, 105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "hi"
                vec![98.0, 121.0, 101.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "bye"
            ]
        }));

        response2.assert_json(&json!({
            "embeddings": [
                vec![102.0, 111.0, 111.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "foo"
                vec![98.0, 97.0, 114.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // "bar"
                vec![98.0, 97.0, 122.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // "baz"
            ]
        }));

        response3.assert_json(&json!({
            "embeddings": [
                vec![120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "x"
                vec![121.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "y"
            ]
        }));

        // Verify mock server received exactly one request
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 1);

        // Verify the request body contains all inputs from all clients
        let request = &received_requests[0];
        let body: serde_json::Value = request.body_json().expect("Request should have JSON body");
        let all_inputs: Vec<&str> = client1_inputs.iter()
            .chain(client2_inputs.iter())
            .chain(client3_inputs.iter())
            .map(|s| *s)
            .collect();
        let expected_body = json!({"inputs": all_inputs});
        assert_eq!(body, expected_body);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_multiple_batching_by_size() {
        // Setup with batch size 3
        let (mock_server, server, state) = setup_embed_test_with_batch_size_and_timeout(3, 3_600_000).await;

        // First batch: client 1 (2 items) + client 2 (2 items) = 4 items total
        // This should trigger batching when we reach 3 items
        let client1_inputs = vec!["a", "b"];
        let client2_inputs = vec!["c", "d"];

        // Make concurrent requests from first two clients
        let client1_request = server
            .post("/embed")
            .json(&json!({"inputs": client1_inputs.clone()}));

        let client2_request = server
            .post("/embed")
            .json(&json!({"inputs": client2_inputs.clone()}));

        // Wait for first batch to complete with timeout
        let (response1, response2) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async { tokio::join!(client1_request, client2_request) }
        )
        .await
        .expect("First batch should complete within timeout");

        // Verify first batch responses
        response1.assert_status_ok();
        response2.assert_status_ok();

        // Check resolved embeddings for first batch
        response1.assert_json(&json!({
            "embeddings": [
                vec![97.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "a"
                vec![98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "b"
            ]
        }));

        response2.assert_json(&json!({
            "embeddings": [
                vec![99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "c"
                vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "d"
            ]
        }));

        // Check first batch call to mock - should be first 3 items plus 1 from overflow
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 1, "Should have exactly 1 upstream call after first batch");

        let request = &received_requests[0];
        let body: serde_json::Value = request.body_json().expect("Request should have JSON body");
        let expected_first_batch = json!({"inputs": ["a", "b", "c", "d"]});
        assert_eq!(body, expected_first_batch);

        // Second batch: client 3 (1 item) + client 4 (4 items) = 5 items total
        // This should trigger batching when we reach 3 items
        let client3_inputs = vec!["e"];
        let client4_inputs = vec!["f", "g", "h", "i"];

        // Make concurrent requests from next two clients
        let client3_request = server
            .post("/embed")
            .json(&json!({"inputs": client3_inputs.clone()}));

        let client4_request = server
            .post("/embed")
            .json(&json!({"inputs": client4_inputs.clone()}));

        // Wait for second batch to complete with timeout
        let (response3, response4) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async { tokio::join!(client3_request, client4_request) }
        )
        .await
        .expect("Second batch should complete within timeout");

        // Verify second batch responses
        response3.assert_status_ok();
        response4.assert_status_ok();

        // Check resolved embeddings for second batch
        response3.assert_json(&json!({
            "embeddings": [
                vec![101.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "e"
            ]
        }));

        response4.assert_json(&json!({
            "embeddings": [
                vec![102.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "f"
                vec![103.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "g"
                vec![104.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "h"
                vec![105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "i"
            ]
        }));

        // Check second batch call to mock - should be exactly 2 total calls now
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 2, "Should have exactly 2 upstream calls after second batch");

        let second_request = &received_requests[1];
        let second_body: serde_json::Value = second_request.body_json().expect("Request should have JSON body");
        let expected_second_batch = json!({"inputs": ["e", "f", "g", "h", "i"]});
        assert_eq!(second_body, expected_second_batch);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test(start_paused = true)]
    async fn test_multiple_batching_by_wait() {
        // Setup with large batch size (1000) and timeout 200ms
        let (mock_server, server, state) = setup_embed_test_with_batch_size_and_timeout(1000, 200).await;

        // First batch: client 1 sends 2 items, should be triggered by timeout
        let client1_inputs = vec!["a", "b"];

        let client1_request = server
            .post("/embed")
            .json(&json!({"inputs": client1_inputs.clone()}));

        // Advance time by 300ms to trigger the 200ms timeout
        tokio::time::advance(std::time::Duration::from_millis(300)).await;

        // Now await the response
        let response1 = client1_request.await;

        // Verify first batch response
        response1.assert_status_ok();

        // Check resolved embeddings for first batch
        response1.assert_json(&json!({
            "embeddings": [
                vec![97.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "a"
                vec![98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "b"
            ]
        }));

        // Check first batch call to mock
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 1, "Should have exactly 1 upstream call after first batch");

        let request = &received_requests[0];
        let body: serde_json::Value = request.body_json().expect("Request should have JSON body");
        let expected_first_batch = json!({"inputs": ["a", "b"]});
        assert_eq!(body, expected_first_batch);

        // Second batch: client 2 (1 item) + client 3 (2 items) = 3 items total
        // Should be triggered by timeout since batch size is 1000
        let client2_inputs = vec!["c"];
        let client3_inputs = vec!["d", "e"];

        let client2_request = server
            .post("/embed")
            .json(&json!({"inputs": client2_inputs.clone()}));

        let client3_request = server
            .post("/embed")
            .json(&json!({"inputs": client3_inputs.clone()}));

        // Advance time by 300ms to trigger the 200ms timeout
        tokio::time::advance(std::time::Duration::from_millis(300)).await;

        // Now await the responses
        let (response2, response3) = tokio::join!(client2_request, client3_request);

        // Verify second batch responses
        response2.assert_status_ok();
        response3.assert_status_ok();

        // Check resolved embeddings for second batch
        response2.assert_json(&json!({
            "embeddings": [
                vec![99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "c"
            ]
        }));

        response3.assert_json(&json!({
            "embeddings": [
                vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "d"
                vec![101.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "e"
            ]
        }));

        // Check second batch call to mock - should be exactly 2 total calls now
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 2, "Should have exactly 2 upstream calls after second batch");

        let second_request = &received_requests[1];
        let second_body: serde_json::Value = second_request.body_json().expect("Request should have JSON body");
        let expected_second_batch = json!({"inputs": ["c", "d", "e"]});
        assert_eq!(second_body, expected_second_batch);

        // Third batch: client 4 sends 1 item, should be triggered by timeout
        let client4_inputs = vec!["f"];

        let client4_request = server
            .post("/embed")
            .json(&json!({"inputs": client4_inputs.clone()}));

        // Advance time by 300ms to trigger the 200ms timeout
        tokio::time::advance(std::time::Duration::from_millis(300)).await;

        // Now await the response
        let response4 = client4_request.await;

        // Verify third batch response
        response4.assert_status_ok();

        // Check resolved embeddings for third batch
        response4.assert_json(&json!({
            "embeddings": [
                vec![102.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "f"
            ]
        }));

        // Check third batch call to mock - should be exactly 3 total calls now
        let received_requests = mock_server.received_requests().await.unwrap();
        assert_eq!(received_requests.len(), 3, "Should have exactly 3 upstream calls after third batch");

        let third_request = &received_requests[2];
        let third_body: serde_json::Value = third_request.body_json().expect("Request should have JSON body");
        let expected_third_batch = json!({"inputs": ["f"]});
        assert_eq!(third_body, expected_third_batch);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_upstream_error_handling() {
        init_test_logging();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(422).set_body_json(json!({
                "error": "Validation error"
            })))
            .mount(&mock_server)
            .await;

        // Create state with proper batcher setup
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
        // The error message will be wrapped differently now

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_upstream_non_json_error() {
        init_test_logging();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(422).set_body_string("Invalid input format"))
            .mount(&mock_server)
            .await;

        // Create state with proper batcher setup
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
        // The error message will be wrapped differently now

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_user_input_empty_array() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server
            .post("/embed")
            .json(&json!({"inputs": []}))
            .await;

        // Empty inputs should be accepted (could be valid business logic)
        response.assert_status_ok();
        response.assert_json(&json!({"embeddings": []}));
    }

    #[tokio::test]
    async fn test_invalid_user_input_malformed_json() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server
            .post("/embed")
            .add_header("content-type", "application/json")
            .text("{ invalid json }")
            .await;

        // Axum rejects malformed JSON at the HTTP layer, returning UNSUPPORTED_MEDIA_TYPE
        // This is the actual behavior, so we test for it
        response.assert_status(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn test_invalid_user_input_json_parse_error() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        // Send valid JSON but with wrong structure (inputs should be array, not string)
        let response = server
            .post("/embed")
            .json(&json!({"inputs": "should be array"}))
            .await;

        // Axum returns UNPROCESSABLE_ENTITY for JSON deserialization errors
        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_invalid_user_input_missing_inputs_field() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server
            .post("/embed")
            .json(&json!({"data": ["test"]}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_invalid_user_input_null_inputs() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server
            .post("/embed")
            .json(&json!({"inputs": null}))
            .await;

        response.assert_status(StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_invalid_user_input_wrong_content_type() {
        let (_mock_server, server, _state) = setup_embed_test_with_batch_size_and_timeout(0, 3_600_000).await;

        let response = server
            .post("/embed")
            .add_header("content-type", "text/plain")
            .text("not json")
            .await;

        response.assert_status(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn test_upstream_500_error() {
        init_test_logging();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "error": "Internal server error"
            })))
            .mount(&mock_server)
            .await;

        // Create state with proper batcher setup
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::INTERNAL_SERVER_ERROR);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_upstream_malformed_response() {
        init_test_logging();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{ malformed json }"))
            .mount(&mock_server)
            .await;

        // Create state with proper batcher setup
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::BAD_GATEWAY);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }

    #[tokio::test]
    async fn test_upstream_connection_error() {
        init_test_logging();
        // Create state with proper batcher setup using invalid URL that will cause connection failure
        let mut config = Config::default();
        config.inference_url = "http://localhost:1".to_string();
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        batcher.clone().spawn_background_processor(channels).await;

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        // Should get a server error due to connection failure
        response.assert_status(StatusCode::BAD_GATEWAY);

        // Cleanup
        state.batcher.shutdown().await.expect("Shutdown should succeed");
    }
}
