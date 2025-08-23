//! `Tray` implementation for managing batching queue.

use crate::types::BatchItem;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::error;

/// A wrapper around Vec<BatchItem> that provides controlled access and triggers callbacks.
#[derive(Debug)]
pub struct Tray {
    /// Internal storage for batch items.
    items: Vec<BatchItem>,
    /// Callback sender for timer notifications when first item changes.
    timer_callback: mpsc::UnboundedSender<(u64, Instant)>,
    /// Callback sender for batch processing - sends the actual batch.
    batch_callback: mpsc::UnboundedSender<Vec<BatchItem>>,
    /// Soft maximum batch size threshold for triggering batch processing.
    soft_max_batch_size: usize,
    /// Current batch sequence number for invalidating old timer events.
    batch_seqno: u64,
}

impl Tray {
    /// Create a new empty `Tray` with the specified soft maximum batch size and callbacks.
    #[must_use]
    pub fn new(
        soft_max_batch_size: usize,
        timer_callback: mpsc::UnboundedSender<(u64, Instant)>,
        batch_callback: mpsc::UnboundedSender<Vec<BatchItem>>,
    ) -> Self {
        Self {
            items: Vec::new(),
            timer_callback,
            batch_callback,
            soft_max_batch_size,
            batch_seqno: 0,
        }
    }

    /// Add multiple batch items from the same request.
    /// This preserves the invariant that items from the same request stay together.
    pub fn append(&mut self, mut new_items: Vec<BatchItem>) {
        let was_empty = self.items.is_empty();

        // Add all items from this request
        self.items.append(&mut new_items);

        // Check if we should trigger batch processing based on size threshold
        // If threshold is reached, send all items and clear the vector
        if self.items.len() >= self.soft_max_batch_size {
            let items_to_send = std::mem::take(&mut self.items);
            self.batch_seqno += 1;
            if let Err(e) = self.batch_callback.send(items_to_send) {
                error!("Failed to send batch items to callback: {:?}", e);
            }
            return;
        }

        // If this was the first item being added and we didn't batch, notify timer
        if was_empty {
            if let Err(e) = self.timer_callback.send((self.batch_seqno, Instant::now())) {
                error!(
                    "Failed to send timer notification: seqno={}, error={:?}",
                    self.batch_seqno, e
                );
            }
        }
    }

    /// Trigger batch processing if the sequence number matches the current one.
    /// Sends all items to the batch callback and clears the vector.
    pub fn trigger_batch(&mut self, expected_batch_seqno: u64) {
        // Only trigger if seqno matches and we have items
        if expected_batch_seqno == self.batch_seqno && !self.items.is_empty() {
            // Send all items and clear the vector
            let items_to_send = std::mem::take(&mut self.items);
            self.batch_seqno += 1;
            if let Err(e) = self.batch_callback.send(items_to_send) {
                error!(
                    "Failed to send timer-triggered batch items to callback: seqno={}, error={:?}",
                    expected_batch_seqno, e
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn create_batch_item(text: &str) -> BatchItem {
        BatchItem {
            text: text.to_string(),
            sender: None,
        }
    }

    #[tokio::test]
    async fn test_append_below_threshold() {
        let (timer_tx, mut timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx); // Threshold 10

        let items = vec![create_batch_item("text1"), create_batch_item("text2")];

        tray.append(items);

        // Should not trigger batch callback (below threshold)
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), batch_rx.recv()).await;
        assert!(result.is_err()); // Should timeout

        // Should trigger timer callback (first item added)
        let timer_notification = timer_rx.recv().await;
        assert!(timer_notification.is_some());
        let (seqno, _timestamp) = timer_notification.unwrap();
        assert_eq!(seqno, 0); // Initial seqno

        // Vector should have items and seqno unchanged
        assert_eq!(tray.items.len(), 2);
        assert_eq!(tray.batch_seqno, 0);
    }

    #[tokio::test]
    async fn test_append_reaches_threshold() {
        let (timer_tx, mut timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(2, timer_tx, batch_tx); // Threshold 2

        let items = vec![create_batch_item("text1"), create_batch_item("text2")];

        tray.append(items);

        // Should trigger batch callback (threshold reached)
        let batch_items = batch_rx.recv().await;
        assert!(batch_items.is_some());
        let items = batch_items.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "text1");
        assert_eq!(items[1].text, "text2");

        // Vector should be empty and seqno incremented
        assert!(tray.items.is_empty());
        assert_eq!(tray.batch_seqno, 1);

        // Should NOT get timer notification since we batched immediately
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), timer_rx.recv()).await;
        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_append_multiple_requests() {
        let (timer_tx, mut timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(5, timer_tx, batch_tx);

        // Add first request (2 items)
        let items1 = vec![create_batch_item("text1a"), create_batch_item("text1b")];
        tray.append(items1);

        // Should get timer notification for first item
        let timer_notification = timer_rx.recv().await;
        assert!(timer_notification.is_some());

        // Add second request (3 items) - total 5, reaches threshold
        let items2 = vec![
            create_batch_item("text2a"),
            create_batch_item("text2b"),
            create_batch_item("text2c"),
        ];
        tray.append(items2);

        // Should trigger batch callback
        let batch_items = batch_rx.recv().await;
        assert!(batch_items.is_some());
        let items = batch_items.unwrap();
        assert_eq!(items.len(), 5);

        // Verify order is preserved
        assert_eq!(items[0].text, "text1a");
        assert_eq!(items[1].text, "text1b");
        assert_eq!(items[2].text, "text2a");
        assert_eq!(items[3].text, "text2b");
        assert_eq!(items[4].text, "text2c");

        // Verify all items are present with correct text
        assert_eq!(items.len(), 5);

        // Should NOT get additional timer notifications since the batch was processed
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), timer_rx.recv()).await;
        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_timer_callback_only_on_first_item() {
        let (timer_tx, mut timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, _batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx);

        // Add first item - should trigger timer
        tray.append(vec![create_batch_item("text1")]);
        let timer_notification = timer_rx.recv().await;
        assert!(timer_notification.is_some());

        // Add more items - should NOT trigger timer again
        tray.append(vec![create_batch_item("text2")]);
        tray.append(vec![create_batch_item("text3")]);

        // Should not receive additional timer notifications
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), timer_rx.recv()).await;
        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_trigger_batch_with_matching_seqno() {
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx);

        // Add items without reaching threshold
        let items = vec![create_batch_item("text1"), create_batch_item("text2")];
        tray.append(items);

        assert_eq!(tray.items.len(), 2);
        assert_eq!(tray.batch_seqno, 0);

        // Trigger batch with matching seqno
        tray.trigger_batch(0);

        // Should receive batch items
        let batch_items = batch_rx.recv().await;
        assert!(batch_items.is_some());
        let items = batch_items.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "text1");
        assert_eq!(items[1].text, "text2");

        // Vector should be empty and seqno incremented
        assert!(tray.items.is_empty());
        assert_eq!(tray.batch_seqno, 1);
    }

    #[tokio::test]
    async fn test_trigger_batch_with_wrong_seqno() {
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx);

        let items = vec![create_batch_item("text1")];
        tray.append(items);

        assert_eq!(tray.items.len(), 1);
        assert_eq!(tray.batch_seqno, 0);

        // Trigger batch with wrong seqno
        tray.trigger_batch(999);

        // Should NOT receive batch items
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), batch_rx.recv()).await;
        assert!(result.is_err()); // Should timeout

        // Vector should be unchanged
        assert_eq!(tray.items.len(), 1);
        assert_eq!(tray.batch_seqno, 0);
    }

    #[tokio::test]
    async fn test_trigger_batch_empty_vector() {
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx);

        // Trigger batch on empty vector
        tray.trigger_batch(0);

        // Should NOT receive batch items
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), batch_rx.recv()).await;
        assert!(result.is_err()); // Should timeout

        // Seqno should be unchanged
        assert_eq!(tray.batch_seqno, 0);
    }

    #[tokio::test]
    async fn test_seqno_increments_correctly() {
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(1, timer_tx, batch_tx); // Threshold 1

        assert_eq!(tray.batch_seqno, 0);

        // First append - should trigger batch and increment seqno
        tray.append(vec![create_batch_item("text1")]);
        let _batch = batch_rx.recv().await.unwrap();
        assert_eq!(tray.batch_seqno, 1);

        // Second append - should trigger batch and increment seqno
        tray.append(vec![create_batch_item("text2")]);
        let _batch = batch_rx.recv().await.unwrap();
        assert_eq!(tray.batch_seqno, 2);

        // Third append - should trigger batch and increment seqno
        tray.append(vec![create_batch_item("text3")]);
        let _batch = batch_rx.recv().await.unwrap();
        assert_eq!(tray.batch_seqno, 3);

        // Verify seqno keeps incrementing
        assert_eq!(tray.batch_seqno, 3);
        assert!(tray.items.is_empty());
    }

    #[tokio::test]
    async fn test_seqno_invalidation() {
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel::<(u64, Instant)>();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel::<Vec<BatchItem>>();

        let mut tray = Tray::new(10, timer_tx, batch_tx); // High threshold

        // Add item, should NOT auto-trigger (below threshold)
        tray.append(vec![create_batch_item("text1")]);
        assert_eq!(tray.batch_seqno, 0);
        assert_eq!(tray.items.len(), 1);

        // Manually trigger with correct seqno
        tray.trigger_batch(0);
        let _batch = batch_rx.recv().await.unwrap();
        assert_eq!(tray.batch_seqno, 1);
        assert!(tray.items.is_empty());

        // Add new item
        tray.append(vec![create_batch_item("text2")]);
        assert_eq!(tray.batch_seqno, 1);
        assert_eq!(tray.items.len(), 1);

        // Try to trigger with old seqno (should fail)
        tray.trigger_batch(0);

        // Should not receive batch
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), batch_rx.recv()).await;
        assert!(result.is_err());

        // Items should still be there, seqno unchanged
        assert_eq!(tray.items.len(), 1);
        assert_eq!(tray.batch_seqno, 1);
    }
}
