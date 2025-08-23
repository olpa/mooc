//! Batcher implementation for coordinating batch processing.

use crate::config::Config;
use crate::timer::Timer;
use crate::tray::Tray;
use crate::types::{BatchError, BatchItem, BatchOfStrings, BatchResult, EmbedResponse};
use axum::http::StatusCode;
use reqwest::Client;
use serde_json::json;
#[allow(unused_imports)]
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

/// Channel receivers for the background processor.
#[derive(Debug)]
pub struct BatcherChannelRcx {
    /// Channel for batch size notifications.
    pub batch_rx: mpsc::UnboundedReceiver<Vec<BatchItem>>,
    /// Channel for Tray to notify Batcher to set timer.
    pub tray_timer_rx: mpsc::UnboundedReceiver<(u64, Instant)>,
    /// Channel for timer notifications.
    pub timer_rx: mpsc::UnboundedReceiver<u64>,
    /// Shutdown signal.
    pub shutdown_rx: mpsc::UnboundedReceiver<()>,
}

/// Main batching coordinator that manages request queuing and processing.
#[derive(Debug)]
pub struct Batcher {
    /// Queue of items waiting to be batched.
    tray: Mutex<Tray>,
    /// Timer for batch timeout functionality.
    timer: Mutex<Timer>,
    /// HTTP client for upstream requests.
    client: Client,
    /// Configuration settings.
    config: Config,
    /// Shutdown signal.
    shutdown_tx: mpsc::UnboundedSender<()>,
}

impl Batcher {
    /// Create a new Batcher instance and its channel receivers.
    #[must_use]
    pub fn new(config: Config, client: Client) -> (Self, BatcherChannelRcx) {
        // Create channels for the main event loop
        let (batch_tx, batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();

        // Create channel for Tray to notify Batcher when first item is added
        let (tray_timer_tx, tray_timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        
        // Create Timer and its notification channel
        let (timer_tx, timer_rx) = mpsc::unbounded_channel::<u64>();
        let timer = Timer::new(Duration::from_millis(config.soft_max_wait_time_ms), timer_tx);
        
        // Create Tray with callbacks to Batcher for timer requests and batch processing
        let tray = Tray::new(config.soft_max_batch_size, tray_timer_tx, batch_tx);

        let batcher = Self {
            tray: Mutex::new(tray),
            timer: Mutex::new(timer),
            client,
            config,
            shutdown_tx,
        };

        let channels = BatcherChannelRcx {
            batch_rx,
            tray_timer_rx,
            timer_rx,
            shutdown_rx,
        };

        (batcher, channels)
    }

    /// Submit a request for batch processing.
    /// Returns a receiver that will receive the result when the batch containing this request is processed.
    pub async fn submit_request(
        &self,
        inputs: BatchOfStrings,
    ) -> oneshot::Receiver<BatchResult> {
        if inputs.is_empty() {
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(Ok(Vec::new()));
            return rx;
        }

        let (tx, rx) = oneshot::channel::<BatchResult>();

        // Create BatchItem for each input, putting sender only in the first item
        let mut tx_option = Some(tx);
        let batch_items: Vec<BatchItem> = inputs
            .into_iter()
            .map(|input| BatchItem {
                text: input,
                sender: tx_option.take(),
            })
            .collect();

        let batch_size = batch_items.len();

        // Add to tray
        {
            let mut tray = self.tray.lock().await;
            tray.append(batch_items);
        }

        debug!("Submitted request with {} inputs", batch_size);

        rx
    }

    /// Start the background processor that handles batching logic.
    /// This runs in an infinite loop until shutdown.
    #[allow(clippy::cognitive_complexity)]
    pub async fn run_background_processor(&self, channels: BatcherChannelRcx) {
        info!("Starting background batch processor");

        let BatcherChannelRcx {
            mut batch_rx,
            mut tray_timer_rx,
            mut timer_rx,
            mut shutdown_rx,
        } = channels;

        loop {
            tokio::select! {
                // Timer notification - time to process batch
                Some(seqno) = timer_rx.recv() => {
                    debug!("Timer notification received with seqno {}", seqno);
                    self.handle_timer_timeout_event(seqno).await;
                }

                // Batch size threshold reached
                Some(batch_items) = batch_rx.recv() => {
                    debug!("Batch size threshold notification received with {} items", batch_items.len());
                    self.process_batch(batch_items).await;
                }

                // Tray requests timer to be set
                Some((seqno, _timestamp)) = tray_timer_rx.recv() => {
                    debug!("Tray timer request received with seqno {}", seqno);
                    let mut timer = self.timer.lock().await;
                    timer.set(seqno);
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
    async fn handle_timer_timeout_event(&self, seqno: u64) {
        let mut tray = self.tray.lock().await;

        // Use the seqno that was stored when the timer was set
        // This ensures we only trigger if the timer is still valid
        tray.trigger_batch(seqno);
    }


    /// Process a batch by calling the upstream inference service.
    async fn process_batch(&self, batch: Vec<BatchItem>) {
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();
        let start_time = Instant::now();

        // Prepare request
        let texts: BatchOfStrings = batch.iter().map(|item| item.text.clone()).collect();
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

        // Handle connection failure
        let resp = match response {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to connect to upstream service: {}", e);
                self.distribute_error(
                    batch,
                    BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        format!("Connection failed: {e}"),
                    ),
                );
                return;
            }
        };

        // Handle non-success status codes
        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("Upstream service returned error {}: {}", status, error_text);

            self.distribute_error(
                batch,
                BatchError::UpstreamError(
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    error_text,
                ),
            );
            return;
        }

        // Parse JSON response and extract embeddings in one step
        let embed_response = match resp.json::<EmbedResponse>().await {
            Ok(embed_response) => embed_response,
            Err(e) => {
                error!("Failed to parse upstream response: {}", e);
                self.distribute_error(
                    batch,
                    BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        format!("Failed to parse response: {e}"),
                    ),
                );
                return;
            }
        };

        let embeddings = &embed_response.embeddings;

        // Success case
        debug!(
            "Successfully processed batch of {} items in {:?}",
            batch_size, processing_time
        );
        self.distribute_results(batch, embeddings);
    }

    /// Distribute successful results back to waiting requests.
    fn distribute_results(&self, batch: Vec<BatchItem>, embeddings: &[Vec<f64>]) {
        // Check that batch and embeddings have the same length
        if batch.len() != embeddings.len() {
            warn!("Batch size ({}) does not match embeddings count ({})", batch.len(), embeddings.len());
            // Send error to all senders in the batch
            for item in batch {
                if let Some(sender) = item.sender {
                    if let Err(_) = sender.send(Err(BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        "Batch size mismatch with embeddings".to_string(),
                    ))) {
                        debug!("Client request receiver dropped for error result");
                    }
                }
            }
            return;
        }

        let mut current_request_results = Vec::new();
        let mut current_sender: Option<oneshot::Sender<BatchResult>> = None;

        // Process each item in the batch paired with its corresponding embedding
        for (item, embedding) in batch.into_iter().zip(embeddings.iter()) {
            // If this item has a sender, it means we're starting a new request
            if let Some(sender) = item.sender {
                // Send results for the previous request if we have one
                if let Some(prev_sender) = current_sender.take() {
                    if let Err(_) = prev_sender.send(Ok(current_request_results)) {
                        debug!("Client request receiver dropped for successful result");
                    }
                    current_request_results = Vec::new();
                }
                current_sender = Some(sender);
            } else {
                warn!("Found orphan BatchItem without sender - cannot distribute result");
            }

            // Add the embedding to results
            current_request_results.push(embedding.clone());
        }

        // Send results for the last request
        if let Some(sender) = current_sender {
            if let Err(_) = sender.send(Ok(current_request_results)) {
                debug!("Client request receiver dropped for successful result");
            }
        } else {
            warn!("No sender found for final request - all BatchItems were orphans");
        }
    }

    /// Distribute error results back to waiting requests.
    #[allow(clippy::needless_pass_by_value)]
    fn distribute_error(&self, batch: Vec<BatchItem>, error: BatchError) {
        // Send error to all senders in the batch
        for item in batch {
            if let Some(sender) = item.sender {
                if let Err(_) = sender.send(Err(error.clone())) {
                    debug!("Client request receiver dropped for error result");
                }
            }
        }
    }

    /// Signal shutdown to the background processor.
    pub fn shutdown(&self) {
        if let Err(e) = self.shutdown_tx.send(()) {
            error!("Failed to send shutdown signal: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_batcher_creation() {
        let config = Config::default();
        let client = Client::new();
        let (_batcher, _channels) = Batcher::new(config, client);

        // Batcher created successfully - we can't easily test queue_length without exposing it
        assert!(true);
    }

    #[tokio::test]
    async fn test_submit_empty_request() {
        let config = Config::default();
        let client = Client::new();
        let (batcher, _channels) = Batcher::new(config, client);

        let rx = batcher.submit_request(vec![]).await;
        let final_result = rx.await.unwrap();
        assert!(final_result.is_ok());
        assert!(final_result.unwrap().is_empty());
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
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        // Submit request
        let rx = batcher
            .submit_request(vec!["text1".to_string(), "text2".to_string()])
            .await;
        let final_result = rx.await.unwrap();
        assert!(final_result.is_ok());
        let embeddings = final_result.unwrap();
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
        let (batcher, channels) = Batcher::new(config, client);
        let batcher = Arc::new(batcher);

        // Start background processor
        let processor_batcher = batcher.clone();
        let processor_handle = tokio::spawn(async move {
            processor_batcher.run_background_processor(channels).await;
        });

        // Submit request
        let rx = batcher.submit_request(vec!["text1".to_string()]).await;
        let final_result = rx.await.unwrap();
        assert!(final_result.is_err());
        if let Err(BatchError::UpstreamError(status, _)) = final_result {
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        } else {
            panic!("Expected UpstreamError");
        }

        // Cleanup
        batcher.shutdown();
        let _ = tokio::time::timeout(Duration::from_millis(100), processor_handle).await;
    }
}
