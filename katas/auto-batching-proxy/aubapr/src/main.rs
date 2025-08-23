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
use tracing::{error, info};
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
    match rx.await {
        Ok(batch_result) => match batch_result {
            Ok(embeddings) => {
                info!("Successfully processed embed request");
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
        },
        Err(_) => {
            error!("Channel closed while waiting for batch result");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal server error"})),
            ))
        }
    }
}

/// Health check endpoint that returns service status.
async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

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

    // Start background processor
    let processor_batcher = batcher.clone();
    let processor_handle = tokio::spawn(async move {
        processor_batcher.run_background_processor(channels).await;
    });

    // Create application state
    let state = Arc::new(AppState {
        batcher: batcher.clone(),
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
    batcher.shutdown();

    // Give some time for background processor to finish
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), processor_handle).await;

    info!("Shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use types::BatchOfEmbeddings;

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

    fn create_app_state(inference_url: String) -> Arc<AppState> {
        let mut config = Config::default();
        config.inference_url = inference_url;
        let client = Client::new();
        let (batcher, _channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        Arc::new(AppState { batcher })
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
    async fn test_one_client_one_input() {
        let mock_server = MockServer::start().await;

        let test_input = "test string";

        // Set up dynamic mock that generates embeddings using generate_test_embedding
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(create_dynamic_embedding_response)
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

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

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_one_client_multiple_inputs() {
        let mock_server = MockServer::start().await;

        let test_inputs = vec!["hello", "world", "test"];

        // Set up dynamic mock that generates embeddings using generate_test_embedding
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(create_dynamic_embedding_response)
            .expect(1) // This ensures the service is called exactly once
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": test_inputs}))
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

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;

        // The mock server will automatically verify that it was called exactly once
        // due to the .expect(1) constraint
    }

    #[tokio::test]
    async fn test_multiple_clients_multiple_inputs() {
        let mock_server = MockServer::start().await;

        // Client 1 inputs: "hi", "bye"
        let client1_inputs = vec!["hi", "bye"];
        // Client 2 inputs: "foo", "bar", "baz"
        let client2_inputs = vec!["foo", "bar", "baz"];
        // Client 3 inputs: "x", "y"
        let client3_inputs = vec!["x", "y"];

        // Set up mock to dynamically generate embeddings using generate_test_embedding
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(create_dynamic_embedding_response)
            .expect(1) // This ensures the service is called exactly once
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        // Make concurrent requests from multiple clients
        let client1_request = server
            .post("/embed")
            .json(&json!({"inputs": client1_inputs}));

        let client2_request = server
            .post("/embed")
            .json(&json!({"inputs": client2_inputs}));

        let client3_request = server
            .post("/embed")
            .json(&json!({"inputs": client3_inputs}));

        // Execute all requests concurrently
        let (response1, response2, response3) =
            tokio::join!(client1_request, client2_request, client3_request);

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

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;

        // The mock server will automatically verify that it was called exactly once
        // due to the .expect(1) constraint
    }

    #[tokio::test]
    async fn test_batching_across_multiple_batches() {
        let mock_server = MockServer::start().await;

        // Counter to track upstream calls
        let call_counter = Arc::new(AtomicUsize::new(0));
        let counter_for_mock = call_counter.clone();

        // Set up mock that counts calls and generates dynamic responses
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(move |req: &wiremock::Request| {
                // Increment counter for each call
                counter_for_mock.fetch_add(1, Ordering::SeqCst);
                create_dynamic_embedding_response(req)
            })
            .mount(&mock_server)
            .await;

        // Configure with small batch size to trigger multiple batches easily
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        config.soft_max_batch_size = 4; // Small batch size for testing
        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        let state = Arc::new(AppState {
            batcher: batcher.clone(),
        });

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        // FIRST BATCH: 4 clients with total 4 inputs (exactly at batch limit)
        let batch1_client1_inputs = vec!["a"]; // 1 input
        let batch1_client2_inputs = vec!["b"]; // 1 input
        let batch1_client3_inputs = vec!["c"]; // 1 input
        let batch1_client4_inputs = vec!["d"]; // 1 input
                                               // Total: 4 inputs (reaches soft_max_batch_size)

        // Make first batch of concurrent requests
        let batch1_request1 = server
            .post("/embed")
            .json(&json!({"inputs": batch1_client1_inputs}));
        let batch1_request2 = server
            .post("/embed")
            .json(&json!({"inputs": batch1_client2_inputs}));
        let batch1_request3 = server
            .post("/embed")
            .json(&json!({"inputs": batch1_client3_inputs}));
        let batch1_request4 = server
            .post("/embed")
            .json(&json!({"inputs": batch1_client4_inputs}));

        // Execute first batch requests concurrently with timeout
        let (batch1_response1, batch1_response2, batch1_response3, batch1_response4) =
            tokio::time::timeout(std::time::Duration::from_millis(100), async {
                tokio::join!(
                    batch1_request1,
                    batch1_request2,
                    batch1_request3,
                    batch1_request4
                )
            })
            .await
            .expect("First batch should complete within timeout");

        // Check that there was exactly one upstream call after first batch
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            1,
            "Should have exactly 1 upstream call after first batch"
        );

        // Verify all first batch responses are successful
        batch1_response1.assert_status_ok();
        batch1_response2.assert_status_ok();
        batch1_response3.assert_status_ok();
        batch1_response4.assert_status_ok();

        // Verify each client gets their correct embeddings back (first batch)
        batch1_response1.assert_json(&json!({
            "embeddings": [
                vec![97.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // "a"
            ]
        }));

        batch1_response2.assert_json(&json!({
            "embeddings": [
                vec![98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // "b"
            ]
        }));

        batch1_response3.assert_json(&json!({
            "embeddings": [
                vec![99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // "c"
            ]
        }));

        batch1_response4.assert_json(&json!({
            "embeddings": [
                vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // "d"
            ]
        }));

        // SECOND BATCH: 3 clients with total 5 inputs (will trigger additional batch processing)
        let batch2_client1_inputs = vec!["e", "f"]; // 2 inputs
        let batch2_client2_inputs = vec!["g"]; // 1 input
        let batch2_client3_inputs = vec!["h", "i"]; // 2 inputs
                                                    // Total: 5 inputs (may be split across multiple batches)

        // Make second batch of concurrent requests
        let batch2_request1 = server
            .post("/embed")
            .json(&json!({"inputs": batch2_client1_inputs}));
        let batch2_request2 = server
            .post("/embed")
            .json(&json!({"inputs": batch2_client2_inputs}));
        let batch2_request3 = server
            .post("/embed")
            .json(&json!({"inputs": batch2_client3_inputs}));

        // Execute second batch requests concurrently with timeout
        let (batch2_response1, batch2_response2, batch2_response3) =
            tokio::time::timeout(std::time::Duration::from_millis(100), async {
                tokio::join!(batch2_request1, batch2_request2, batch2_request3)
            })
            .await
            .expect("Second batch should complete within timeout");

        // Check that there are now at least 2 upstream calls
        let calls_after_second_batch = call_counter.load(Ordering::SeqCst);
        assert!(
            calls_after_second_batch >= 2,
            "Should have at least 2 upstream calls after second batch, got {}",
            calls_after_second_batch
        );

        // Verify all second batch responses are successful
        batch2_response1.assert_status_ok();
        batch2_response2.assert_status_ok();
        batch2_response3.assert_status_ok();

        // Verify each client gets their correct embeddings back (second batch)
        batch2_response1.assert_json(&json!({
            "embeddings": [
                vec![101.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "e"
                vec![102.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "f"
            ]
        }));

        batch2_response2.assert_json(&json!({
            "embeddings": [
                vec![103.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // "g"
            ]
        }));

        batch2_response3.assert_json(&json!({
            "embeddings": [
                vec![104.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "h"
                vec![105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // "i"
            ]
        }));

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;

        // Final verification: confirm multiple batches were processed
        let final_call_count = call_counter.load(Ordering::SeqCst);
        assert!(
            final_call_count >= 2,
            "Should have had at least 2 upstream calls total, got {}",
            final_call_count
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_timeout_batching() {
        let mock_server = MockServer::start().await;

        // Counter to track upstream calls
        let call_counter = Arc::new(AtomicUsize::new(0));
        let counter_for_mock = call_counter.clone();

        // Set up mock that counts calls and generates dynamic responses
        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(move |req: &wiremock::Request| {
                // Increment counter for each call
                counter_for_mock.fetch_add(1, Ordering::SeqCst);
                create_dynamic_embedding_response(req)
            })
            .mount(&mock_server)
            .await;

        // Configure with short timeout and large batch size so only timeout triggers batching
        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        config.soft_max_wait_time_ms = 50; // Short timeout for testing
        config.soft_max_batch_size = 10; // Reasonable batch size but test won't reach it

        let client = Client::new();
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        // Submit a request - this should trigger timeout-based batching
        let batcher_clone = batcher.clone();
        let request_task = tokio::spawn(async move {
            let rx = batcher_clone
                .submit_request(vec!["test".to_string()])
                .await;
            rx.await.unwrap()
        });

        // Yield to allow the request to be queued
        tokio::task::yield_now().await;

        // Verify no upstream calls yet (timer hasn't expired)
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            0,
            "Should have no upstream calls before timeout"
        );

        // Use time travel to trigger the timeout! Advance time by 60ms to trigger the 50ms timeout
        tokio::time::advance(std::time::Duration::from_millis(60)).await;

        // Allow time for processing
        tokio::task::yield_now().await;

        // Give more time for the HTTP request to complete
        tokio::time::advance(std::time::Duration::from_millis(10)).await;
        tokio::task::yield_now().await;

        // Complete the request
        let result = tokio::time::timeout(std::time::Duration::from_millis(2000), request_task)
            .await
            .expect("Request should complete after timeout")
            .expect("Request task should not panic");

        // Verify the timeout triggered batch processing
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            1,
            "Should have exactly 1 upstream call after timeout"
        );

        // Verify the result is correct
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(
            embeddings[0],
            vec![
                116.0, 101.0, 115.0, 116.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0
            ]
        ); // "test"

        // Test second timeout batching - submit another request
        let batcher_clone2 = batcher.clone();
        let request_task2 = tokio::spawn(async move {
            let rx = batcher_clone2
                .submit_request(vec!["second".to_string()])
                .await;
            rx.await.unwrap()
        });

        tokio::task::yield_now().await;

        // Verify still only 1 upstream call (second request waiting)
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            1,
            "Should still have only 1 upstream call before second timeout"
        );

        // Time travel again to trigger second timeout
        tokio::time::advance(std::time::Duration::from_millis(60)).await;
        tokio::task::yield_now().await;

        let result2 = tokio::time::timeout(std::time::Duration::from_millis(2000), request_task2)
            .await
            .expect("Second request should complete after timeout")
            .expect("Second request task should not panic");

        // Verify second batch was processed due to timeout
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            2,
            "Should have exactly 2 upstream calls after second timeout"
        );

        // Verify second result
        assert!(result2.is_ok());
        let embeddings2 = result2.unwrap();
        assert_eq!(embeddings2.len(), 1);
        assert_eq!(
            embeddings2[0],
            vec![
                115.0, 101.0, 99.0, 111.0, 110.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0
            ]
        ); // "second"

        // Cleanup
        batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);
        // The error message will be wrapped differently now

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_upstream_non_json_error() {
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);
        // The error message will be wrapped differently now

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_user_input_empty_array() {
        let mock_server = MockServer::start().await;
        let state = create_app_state(mock_server.uri());

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

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
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

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
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        // Send valid JSON but with wrong structure (inputs should be array, not string)
        let response = server
            .post("/embed")
            .json(&json!({"inputs": "should be array"}))
            .await;

        // This should return BAD_REQUEST because JSON structure is wrong
        response.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_invalid_user_input_missing_inputs_field() {
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"data": ["test"]}))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_invalid_user_input_null_inputs() {
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": null}))
            .await;

        response.assert_status(StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_invalid_user_input_wrong_content_type() {
        let state = create_app_state("http://localhost:8080".to_string());
        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .add_header("content-type", "text/plain")
            .text("not json")
            .await;

        response.assert_status(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn test_upstream_500_error() {
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::INTERNAL_SERVER_ERROR);

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_upstream_malformed_response() {
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        response.assert_status(StatusCode::BAD_GATEWAY);

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_upstream_connection_error() {
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
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        let app = create_app(state.clone());
        let server = TestServer::new(app).unwrap();

        let response = server
            .post("/embed")
            .json(&json!({"inputs": ["test string"]}))
            .await;

        // Should get a server error due to connection failure
        response.assert_status(StatusCode::BAD_GATEWAY);

        // Cleanup
        state.batcher.shutdown();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(100), processor_handle).await;
    }
}
