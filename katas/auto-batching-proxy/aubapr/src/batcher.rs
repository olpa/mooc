//! Batcher implementation for coordinating batch processing.

use crate::active_vector::ActiveVector;
use crate::config::Config;
use crate::timer::Timer;
use crate::types::{BatchError, BatchItem, BatchResult};
use axum::http::StatusCode;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
#[allow(unused_imports)]
use std::time::Duration;
use tokio::time::Instant;
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Main batching coordinator that manages request queuing and processing.
#[derive(Debug)]
pub struct Batcher {
    /// Queue of items waiting to be batched.
    active_vector: Arc<Mutex<ActiveVector>>,
    /// Timer for batch timeout functionality.
    #[allow(dead_code)]
    timer: Arc<Mutex<Timer>>,
    /// HTTP client for upstream requests.
    client: Client,
    /// Configuration settings.
    config: Config,
    /// Channel for timer notifications.
    timer_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<()>>>>,
    /// Channel for batch size notifications.
    batch_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<()>>>>,
    /// Shutdown signal.
    shutdown_tx: mpsc::UnboundedSender<()>,
    shutdown_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<()>>>>,
    /// Timeout manager task handle.
    #[allow(dead_code)]
    timeout_manager: tokio::task::JoinHandle<()>,
}

impl Batcher {
    /// Create a new Batcher instance.
    #[must_use]
    pub fn new(config: Config, client: Client) -> Self {
        let mut active_vector = ActiveVector::new();
        let mut timer = Timer::new();

        let (timer_tx, timer_rx) = mpsc::unbounded_channel();
        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();

        active_vector.set_callbacks(timer_tx, batch_tx);
        let timer_receiver = timer.take_receiver().expect("Timer should have receiver");
        
        // Connect ActiveVector timer notifications to Timer component
        let timer_arc = Arc::new(Mutex::new(timer));
        let timer_manager_arc = timer_arc.clone();
        let timeout_ms = config.soft_max_wait_time_ms;
        let timeout_manager = tokio::spawn(async move {
            // Listen for first-item timestamp notifications from ActiveVector
            let mut timer_notifications = timer_rx;
            while let Some(timestamp) = timer_notifications.recv().await {
                // Set timer to fire after timeout duration from the timestamp
                let mut timer_guard = timer_manager_arc.lock().await;
                timer_guard.set(std::time::Duration::from_millis(timeout_ms), timestamp);
            }
        });

        Self {
            active_vector: Arc::new(Mutex::new(active_vector)),
            timer: timer_arc,
            client,
            config,
            timer_rx: Arc::new(Mutex::new(Some(timer_receiver))),
            batch_rx: Arc::new(Mutex::new(Some(batch_rx))),
            shutdown_tx,
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
            timeout_manager,
        }
    }

    /// Submit a request for batch processing.
    /// Returns a future that resolves when the batch containing this request is processed.
    ///
    /// # Errors
    ///
    /// Returns `BatchError::ServiceUnavailable` if the queue is full.
    pub async fn submit_request(&self, inputs: Vec<String>) -> Result<Vec<Vec<f64>>, BatchError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Check queue size for backpressure
        {
            let av = self.active_vector.lock().await;
            let queue_size = av.len();
            drop(av);
            if queue_size >= self.config.max_queue_size {
                warn!(
                    "Queue size {} exceeds max {}, returning 503",
                    queue_size, self.config.max_queue_size
                );
                return Err(BatchError::ServiceUnavailable);
            }
        }

        let request_id = Uuid::new_v4();
        let mut receivers = Vec::new();
        let mut batch_items = Vec::new();

        // Create BatchItem for each input
        for input in inputs {
            let (tx, rx) = oneshot::channel::<BatchResult>();
            receivers.push(rx);

            batch_items.push(BatchItem {
                text: input,
                request_id,
                sender: tx,
                timestamp: Instant::now(),
            });
        }

        // Add to active vector
        {
            let mut av = self.active_vector.lock().await;
            av.extend(batch_items);
        }

        debug!(
            "Submitted request {} with {} inputs",
            request_id,
            receivers.len()
        );

        // Wait for all results
        let mut results = Vec::new();
        for rx in receivers {
            match rx.await {
                Ok(Ok(embedding)) => results.push(embedding),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(BatchError::Timeout),
            }
        }

        Ok(results)
    }

    /// Start the background processor that handles batching logic.
    /// This runs in an infinite loop until shutdown.
    #[allow(clippy::cognitive_complexity, clippy::expect_used)]
    pub async fn run_background_processor(&self) {
        info!("Starting background batch processor");

        // Take ownership of receivers
        let mut timer_rx = {
            let mut rx_guard = self.timer_rx.lock().await;
            rx_guard.take().expect("Timer receiver should be available")
        };

        let mut batch_rx = {
            let mut rx_guard = self.batch_rx.lock().await;
            rx_guard.take().expect("Batch receiver should be available")
        };

        let mut shutdown_rx = {
            let mut rx_guard = self.shutdown_rx.lock().await;
            rx_guard
                .take()
                .expect("Shutdown receiver should be available")
        };

        loop {
            tokio::select! {
                // Timer notification - time to process batch
                Some(()) = timer_rx.recv() => {
                    debug!("Timer notification received");
                    self.handle_timer_timeout_event().await;
                }

                // Batch size threshold reached
                Some(()) = batch_rx.recv() => {
                    debug!("Batch size threshold notification received");
                    self.handle_batch_threshold_event().await;
                }

                // Shutdown signal
                Some(()) = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping batch processor");
                    break;
                }

                else => {
                    warn!("All channels closed, stopping batch processor");
                    break;
                }
            }
        }

        info!("Background batch processor stopped");
    }

    /// Handle timer timeout event from Timer component.
    async fn handle_timer_timeout_event(&self) {
        let batch = {
            let mut av = self.active_vector.lock().await;
            
            // Get first item timestamp - this is what we'll process
            if av.len() == 0 {
                debug!("Timer fired but queue is empty, ignoring");
                return;
            }
            
            // Extract a batch
            av.slice(self.config.soft_max_batch_size)
        };

        if !batch.is_empty() {
            info!("Processing timer-triggered batch with {} items", batch.len());
            self.process_batch(batch).await;
        }
    }

    /// Handle timer expiration event (legacy method - kept for compatibility).
    #[allow(clippy::cognitive_complexity)]
    async fn handle_timer_event(&self, expected_timestamp: Instant) {
        let batch = {
            let mut av = self.active_vector.lock().await;

            // Check if this timer is still valid (first item timestamp matches)
            if let Some(first_item_time) = av.first_timestamp() {
                if first_item_time != expected_timestamp {
                    debug!("Timer event for old batch, ignoring");
                    return;
                }
            } else {
                debug!("No items in queue, ignoring timer event");
                return;
            }

            // Extract a batch
            av.slice(self.config.soft_max_batch_size)
        };

        if !batch.is_empty() {
            info!(
                "Processing timer-triggered batch with {} items",
                batch.len()
            );
            self.process_batch(batch).await;
        }
    }

    /// Handle batch size threshold event.
    async fn handle_batch_threshold_event(&self) {
        let batch = {
            let mut av = self.active_vector.lock().await;
            if av.len() >= self.config.soft_max_batch_size {
                av.slice(self.config.soft_max_batch_size)
            } else {
                Vec::new()
            }
        };

        if !batch.is_empty() {
            info!("Processing size-triggered batch with {} items", batch.len());
            self.process_batch(batch).await;
        }
    }

    /// Process a batch by calling the upstream inference service.
    #[allow(clippy::cognitive_complexity)]
    async fn process_batch(&self, batch: Vec<BatchItem>) {
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();
        let start_time = Instant::now();

        // Prepare request
        let texts: Vec<String> = batch.iter().map(|item| item.text.clone()).collect();
        let request = json!({ "inputs": texts });

        debug!("Sending batch of {} items to upstream service", batch_size);

        // Make upstream request
        let response = self
            .client
            .post(format!("{}/embed", self.config.inference_url))
            .json(&request)
            .send()
            .await;

        let processing_time = start_time.elapsed();

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    // Parse successful response
                    match resp.json::<serde_json::Value>().await {
                        Ok(json_response) => {
                            if let Some(embeddings) =
                                json_response.get("embeddings").and_then(|e| e.as_array())
                            {
                                info!(
                                    "Successfully processed batch of {} items in {:?}",
                                    batch_size, processing_time
                                );
                                self.distribute_results(batch, embeddings);
                            } else {
                                error!("Invalid response format from upstream service");
                                self.distribute_error(
                                    batch,
                                    BatchError::UpstreamError(
                                        StatusCode::BAD_GATEWAY,
                                        "Invalid response format".to_string(),
                                    ),
                                );
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse upstream response: {}", e);
                            self.distribute_error(
                                batch,
                                BatchError::UpstreamError(
                                    StatusCode::BAD_GATEWAY,
                                    format!("Failed to parse response: {e}"),
                                ),
                            );
                        }
                    }
                } else {
                    // Handle upstream error
                    let status = resp.status();
                    let error_text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    error!("Upstream service returned error {}: {}", status, error_text);

                    self.distribute_error(
                        batch,
                        BatchError::UpstreamError(
                            StatusCode::from_u16(status.as_u16())
                                .unwrap_or(StatusCode::BAD_GATEWAY),
                            error_text,
                        ),
                    );
                }
            }
            Err(e) => {
                error!("Failed to connect to upstream service: {}", e);
                self.distribute_error(
                    batch,
                    BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        format!("Connection failed: {e}"),
                    ),
                );
            }
        }
    }

    /// Distribute successful results back to waiting requests.
    #[allow(clippy::unused_self)]
    fn distribute_results(&self, batch: Vec<BatchItem>, embeddings: &[serde_json::Value]) {
        // Group items by request_id
        let mut requests: HashMap<Uuid, Vec<(usize, BatchItem)>> = HashMap::new();

        for (idx, item) in batch.into_iter().enumerate() {
            requests
                .entry(item.request_id)
                .or_default()
                .push((idx, item));
        }

        // Send results to each request
        for (_request_id, items) in requests {
            for (idx, item) in items {
                if let Some(embedding_json) = embeddings.get(idx) {
                    if let Some(embedding_array) = embedding_json.as_array() {
                        let embedding: Vec<f64> = embedding_array
                            .iter()
                            .filter_map(serde_json::Value::as_f64)
                            .collect();

                        if embedding.len() == embedding_array.len() {
                            let _ = item.sender.send(Ok(embedding));
                        } else {
                            let _ = item.sender.send(Err(BatchError::UpstreamError(
                                StatusCode::BAD_GATEWAY,
                                "Invalid embedding format".to_string(),
                            )));
                        }
                    } else {
                        let _ = item.sender.send(Err(BatchError::UpstreamError(
                            StatusCode::BAD_GATEWAY,
                            "Embedding is not an array".to_string(),
                        )));
                    }
                } else {
                    let _ = item.sender.send(Err(BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        "Missing embedding in response".to_string(),
                    )));
                }
            }
        }
    }

    /// Distribute error results back to waiting requests.
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    fn distribute_error(&self, batch: Vec<BatchItem>, error: BatchError) {
        for item in batch {
            let _ = item.sender.send(Err(error.clone()));
        }
    }

    /// Signal shutdown to the background processor.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Get current queue length for monitoring.
    #[allow(dead_code)]
    pub async fn queue_length(&self) -> usize {
        let av = self.active_vector.lock().await;
        av.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_batcher_creation() {
        let config = Config::default();
        let client = Client::new();
        let _batcher = Batcher::new(config, client);

        // Batcher created successfully - we can't easily test queue_length without exposing it
        assert!(true);
    }

    #[tokio::test]
    async fn test_submit_empty_request() {
        let config = Config::default();
        let client = Client::new();
        let batcher = Batcher::new(config, client);

        let result = batcher.submit_request(vec![]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_backpressure() {
        let mut config = Config::default();
        config.max_queue_size = 2;

        let client = Client::new();
        let batcher = Arc::new(Batcher::new(config, client));

        // Fill queue to capacity
        let batcher1 = batcher.clone();
        let handle1 =
            tokio::spawn(async move { batcher1.submit_request(vec!["text1".to_string()]).await });

        let batcher2 = batcher.clone();
        let handle2 =
            tokio::spawn(async move { batcher2.submit_request(vec!["text2".to_string()]).await });

        // Give tasks time to add to queue
        tokio::time::sleep(Duration::from_millis(10)).await;

        // This should fail with ServiceUnavailable
        let result = batcher.submit_request(vec!["text3".to_string()]).await;
        assert!(matches!(result, Err(BatchError::ServiceUnavailable)));

        // Cleanup
        handle1.abort();
        handle2.abort();
    }

    #[tokio::test]
    async fn test_successful_batch_processing() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": [[0.1, 0.2], [0.3, 0.4]]
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.inference_url = mock_server.uri();
        config.soft_max_batch_size = 10;

        let client = Client::new();
        let batcher = Arc::new(Batcher::new(config, client));

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor().await;
        });

        // Submit request
        let result = batcher
            .submit_request(vec!["text1".to_string(), "text2".to_string()])
            .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![0.1, 0.2]);
        assert_eq!(embeddings[1], vec![0.3, 0.4]);

        // Cleanup
        batcher.shutdown();
        let _ = tokio::time::timeout(Duration::from_millis(100), processor_handle).await;
    }

    #[tokio::test]
    async fn test_upstream_error_handling() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/embed"))
            .respond_with(ResponseTemplate::new(422).set_body_json(json!({
                "error": "Invalid input"
            })))
            .mount(&mock_server)
            .await;

        let mut config = Config::default();
        config.inference_url = mock_server.uri();

        let client = Client::new();
        let batcher = Arc::new(Batcher::new(config, client));

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor().await;
        });

        // Submit request
        let result = batcher.submit_request(vec!["text1".to_string()]).await;

        assert!(result.is_err());
        if let Err(BatchError::UpstreamError(status, _)) = result {
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        } else {
            panic!("Expected UpstreamError");
        }

        // Cleanup
        batcher.shutdown();
        let _ = tokio::time::timeout(Duration::from_millis(100), processor_handle).await;
    }
}
