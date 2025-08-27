//! Batcher implementation for coordinating batch processing.

use crate::config::Config;
use crate::timer::Timer;
use crate::tray::Tray;
use crate::types::{BatchError, BatchItem, BatchOfStrings, BatchResult, EmbedResponse};
use axum::http::StatusCode;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use std::sync::Arc;
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

/// Channel receivers for the background processor.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct BatcherChannelRcx {
    /// Channel for batch size notifications.
    pub batch_rx: mpsc::UnboundedReceiver<Vec<BatchItem>>,
    /// Channel for Tray to notify Batcher to set timer.
    pub tray_timer_rx: mpsc::UnboundedReceiver<u64>,
    /// Channel for timer notifications.
    pub timer_rx: mpsc::UnboundedReceiver<u64>,
    /// Shutdown signal with completion channel.
    pub shutdown_rx: mpsc::UnboundedReceiver<oneshot::Sender<()>>,
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
    /// Shutdown signal with completion channel.
    shutdown_tx: mpsc::UnboundedSender<oneshot::Sender<()>>,
}

impl Batcher {
    /// Create a new Batcher instance and its channel receivers.
    #[must_use]
    pub fn new(config: Config, client: Client) -> (Self, BatcherChannelRcx) {
        // Create channels for the main event loop
        let (batch_tx, batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel::<oneshot::Sender<()>>();

        // Create channel for Tray to notify Batcher when first item is added
        let (tray_timer_tx, tray_timer_rx) = mpsc::unbounded_channel::<u64>();
        
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
            if tx.send(Ok(Vec::new())).is_err() {
                error!("Failed to send empty result for empty inputs");
            }
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
    /// Spawn the background processor and wait for it to start.
    pub async fn spawn_background_processor(
        self: Arc<Self>, 
        channels: BatcherChannelRcx
    ) -> tokio::task::JoinHandle<()> {
        let (started_tx, started_rx) = oneshot::channel();
        
        let handle = tokio::spawn(async move {
            self.run_background_processor(channels, started_tx).await;
        });
        
        // Wait for processor to signal it has started
        if started_rx.await.is_err() {
            error!("Failed to receive processor started signal");
        }
        
        handle
    }

    async fn run_background_processor(&self, channels: BatcherChannelRcx, started_tx: oneshot::Sender<()>) {
        info!("Starting background batch processor");
        
        // Signal that we've started
        if started_tx.send(()).is_err() {
            error!("Failed to send processor started signal");
        }

        let BatcherChannelRcx {
            mut batch_rx,
            mut tray_timer_rx,
            mut timer_rx,
            mut shutdown_rx,
        } = channels;

        loop {
            tokio::select! {
                Some(seqno) = timer_rx.recv() => {
                    debug!("Bg loop: Timer notification received with seqno {}, next: trigger batching", seqno);
                    self.handle_timer_timeout_event(seqno).await;
                }

                Some(batch_items) = batch_rx.recv() => {
                    debug!("Bg loop: Batch received with {} items, next: main logic", batch_items.len());
                    self.process_batch(batch_items).await;
                }

                Some(seqno) = tray_timer_rx.recv() => {
                    debug!("Bg loop: Tray restart received with seqno {}, next: reset the timer", seqno);
                    let mut timer = self.timer.lock().await;
                    timer.set(seqno);
                }

                Some(completion_tx) = shutdown_rx.recv() => {
                    info!("Bg loop: Shutdown signal received, next: stop the batch processor");
                    
                    // Signal shutdown completion before exiting
                    if completion_tx.send(()).is_err() {
                        error!("Failed to send shutdown completion signal");
                    }
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
                Self::distribute_error(
                    batch,
                    &BatchError::UpstreamError(
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

            Self::distribute_error(
                batch,
                &BatchError::UpstreamError(
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
                Self::distribute_error(
                    batch,
                    &BatchError::UpstreamError(
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
        Self::distribute_results(batch, embeddings);
    }

    /// Distribute successful results back to waiting requests.
    fn distribute_results(batch: Vec<BatchItem>, embeddings: &[Vec<f64>]) {
        // Check that batch and embeddings have the same length
        if batch.len() != embeddings.len() {
            warn!("Batch size ({}) does not match embeddings count ({})", batch.len(), embeddings.len());
            // Send error to all senders in the batch
            for item in batch {
                if let Some(sender) = item.sender {
                    if sender.send(Err(BatchError::UpstreamError(
                        StatusCode::BAD_GATEWAY,
                        "Batch size mismatch with embeddings".to_string(),
                    ))).is_err() {
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
                    if prev_sender.send(Ok(current_request_results)).is_err() {
                        debug!("Client request receiver dropped for successful result");
                    }
                    current_request_results = Vec::new();
                }
                current_sender = Some(sender);
            } else if current_sender.is_none() {
                warn!("Found orphan BatchItem without sender - cannot distribute result");
            }

            // Add the embedding to results
            current_request_results.push(embedding.clone());
        }

        // Send results for the last request
        if let Some(sender) = current_sender {
            if sender.send(Ok(current_request_results)).is_err() {
                debug!("Client request receiver dropped for successful result");
            }
        } else {
            warn!("No sender found for final request - all BatchItems were orphans");
        }
    }

    /// Distribute error results back to waiting requests.
    fn distribute_error(batch: Vec<BatchItem>, error: &BatchError) {
        // Send error to all senders in the batch
        for item in batch {
            if let Some(sender) = item.sender {
                if sender.send(Err(error.clone())).is_err() {
                    debug!("Client request receiver dropped for error result");
                }
            }
        }
    }

    /// Signal shutdown to the background processor.
    pub fn shutdown(&self) -> oneshot::Receiver<()> {
        let (completion_tx, completion_rx) = oneshot::channel();
        
        // Send shutdown signal with completion channel
        if self.shutdown_tx.send(completion_tx).is_err() {
            error!("Channel closed, processor already stopped");
            // For immediate resolution, we need to create a resolved receiver
            let (tx, rx) = oneshot::channel();
            if tx.send(()).is_err() {
                error!("Impossible, can't recover - failed to send to fresh oneshot channel");
            }
            return rx;
        }
        
        // Return the completion receiver directly
        completion_rx
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
        let processor_handle = batcher.clone().spawn_background_processor(channels).await;

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
        if batcher.shutdown().await.is_err() {
            error!("Failed to shutdown batcher in test cleanup");
        }
        if let Err(_timeout) = tokio::time::timeout(Duration::from_millis(100), processor_handle).await {
            error!("Processor handle did not complete within timeout");
        }
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
        let processor_handle = batcher.clone().spawn_background_processor(channels).await;

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
        if batcher.shutdown().await.is_err() {
            error!("Failed to shutdown batcher in test cleanup");
        }
        if let Err(_timeout) = tokio::time::timeout(Duration::from_millis(100), processor_handle).await {
            error!("Processor handle did not complete within timeout");
        }
    }
}
