//! `ActiveVector` implementation for managing batching queue.

use crate::types::BatchItem;
use tokio::time::Instant;
use tokio::sync::mpsc;

/// A wrapper around Vec<BatchItem> that provides controlled access and triggers callbacks.
#[derive(Debug)]
pub struct ActiveVector {
    /// Internal storage for batch items.
    items: Vec<BatchItem>,
    /// Optional callback sender for timer notifications when first item changes.
    timer_callback: Option<mpsc::UnboundedSender<Instant>>,
    /// Optional callback sender for batch processing when size threshold is reached.
    batch_callback: Option<mpsc::UnboundedSender<()>>,
}

impl ActiveVector {
    /// Create a new empty `ActiveVector`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items: Vec::new(),
            timer_callback: None,
            batch_callback: None,
        }
    }

    /// Set callback channels for notifications.
    pub fn set_callbacks(
        &mut self,
        timer_tx: mpsc::UnboundedSender<Instant>,
        batch_tx: mpsc::UnboundedSender<()>,
    ) {
        self.timer_callback = Some(timer_tx);
        self.batch_callback = Some(batch_tx);
    }

    /// Add multiple batch items from the same request.
    /// This preserves the invariant that items from the same request stay together.
    pub fn extend(&mut self, mut new_items: Vec<BatchItem>) {
        let was_empty = self.items.is_empty();

        // Add all items from this request
        self.items.append(&mut new_items);

        // If this was the first item being added, notify timer with timestamp
        if was_empty && !self.items.is_empty() {
            if let Some(ref timer_tx) = self.timer_callback {
                if let Some(first_item) = self.items.first() {
                    let _ = timer_tx.send(first_item.timestamp);
                }
            }
        }

        // Check if we should trigger batch processing
        self.check_batch_threshold();
    }

    /// Extract a batch of items up to the specified maximum size.
    /// Items from the same request are never split across batches.
    /// Returns the extracted items, preserving order.
    pub fn slice(&mut self, max_size: usize) -> Vec<BatchItem> {
        if self.items.is_empty() || max_size == 0 {
            return Vec::new();
        }

        let mut extracted = Vec::new();
        let mut i = 0;

        while i < self.items.len() {
            let current_request_id = self.items[i].request_id;
            let mut request_items = Vec::new();

            // Collect all items for this request
            while i < self.items.len() && self.items[i].request_id == current_request_id {
                request_items.push(i);
                i += 1;
            }

            // Check if adding this request would exceed max_size
            if extracted.len() + request_items.len() <= max_size {
                // Add this entire request to the batch
                for &idx in &request_items {
                    extracted.push(idx);
                }
            } else {
                // If extracted is empty, we must take this request anyway (soft limit)
                if extracted.is_empty() {
                    for &idx in &request_items {
                        extracted.push(idx);
                    }
                }
                // Otherwise, we're done - this request would exceed the limit
                break;
            }
        }

        // Extract items in reverse order to maintain indices
        extracted.sort_by(|a, b| b.cmp(a));
        let mut result = Vec::new();
        for idx in extracted {
            result.push(self.items.remove(idx));
        }

        // Reverse to restore original order
        result.reverse();

        // If there are still items and we removed some items, notify timer with new first timestamp
        if !result.is_empty() && !self.items.is_empty() {
            if let Some(ref timer_tx) = self.timer_callback {
                if let Some(first_item) = self.items.first() {
                    let _ = timer_tx.send(first_item.timestamp);
                }
            }
        }

        result
    }

    /// Get the current number of items in the vector.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the vector is empty.
    #[must_use]
    #[allow(dead_code)]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the timestamp of the first item, if any.
    #[must_use]
    pub fn first_timestamp(&self) -> Option<Instant> {
        self.items.first().map(|item| item.timestamp)
    }

    /// Check if we should trigger batch processing based on size threshold.
    fn check_batch_threshold(&self) {
        // Trigger batch processing when we have items
        // The batcher will determine if the actual size threshold is met
        if !self.items.is_empty() {
            if let Some(ref batch_tx) = self.batch_callback {
                let _ = batch_tx.send(());
            }
        }
    }
}

impl Default for ActiveVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BatchResult;
    use tokio::sync::oneshot;
    use uuid::Uuid;

    fn create_batch_item(text: &str, request_id: Uuid) -> BatchItem {
        let (tx, _rx) = oneshot::channel::<BatchResult>();
        BatchItem {
            text: text.to_string(),
            request_id,
            sender: tx,
            timestamp: Instant::now(),
        }
    }

    #[test]
    fn test_new_active_vector() {
        let av = ActiveVector::new();
        assert_eq!(av.len(), 0);
        assert!(av.is_empty());
    }

    #[test]
    fn test_extend_single_request() {
        let mut av = ActiveVector::new();
        let request_id = Uuid::new_v4();
        let items = vec![
            create_batch_item("text1", request_id),
            create_batch_item("text2", request_id),
        ];

        av.extend(items);
        assert_eq!(av.len(), 2);
        assert!(!av.is_empty());
    }

    #[test]
    fn test_slice_empty_vector() {
        let mut av = ActiveVector::new();
        let result = av.slice(10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_slice_with_single_request() {
        let mut av = ActiveVector::new();
        let request_id = Uuid::new_v4();
        let items = vec![
            create_batch_item("text1", request_id),
            create_batch_item("text2", request_id),
        ];

        av.extend(items);
        let extracted = av.slice(5);

        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].text, "text1");
        assert_eq!(extracted[1].text, "text2");
        assert_eq!(av.len(), 0);
    }

    #[test]
    fn test_slice_keeps_requests_together() {
        let mut av = ActiveVector::new();
        let request1 = Uuid::new_v4();
        let request2 = Uuid::new_v4();

        let items1 = vec![
            create_batch_item("text1a", request1),
            create_batch_item("text1b", request1),
        ];
        let items2 = vec![
            create_batch_item("text2a", request2),
            create_batch_item("text2b", request2),
        ];

        av.extend(items1);
        av.extend(items2);

        // Try to extract only 2 items, but we should get the complete first request
        let extracted = av.slice(2);

        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].request_id, request1);
        assert_eq!(extracted[1].request_id, request1);
        assert_eq!(av.len(), 2); // Second request should remain
    }

    #[test]
    fn test_slice_respects_max_size_soft_limit() {
        let mut av = ActiveVector::new();
        let request1 = Uuid::new_v4();
        let request2 = Uuid::new_v4();

        // Add 3 items from first request
        let items1 = vec![
            create_batch_item("text1a", request1),
            create_batch_item("text1b", request1),
            create_batch_item("text1c", request1),
        ];
        av.extend(items1);

        // Add 2 items from second request
        let items2 = vec![
            create_batch_item("text2a", request2),
            create_batch_item("text2b", request2),
        ];
        av.extend(items2);

        // Try to extract only 2 items - should take the complete first request (3 items)
        // because we don't split requests
        let extracted = av.slice(2);

        assert_eq!(extracted.len(), 3); // Soft limit exceeded to keep request together
        assert!(extracted.iter().all(|item| item.request_id == request1));
        assert_eq!(av.len(), 2); // Second request should remain
    }

    #[tokio::test]
    async fn test_timer_callback_on_first_item() {
        let mut av = ActiveVector::new();
        let (timer_tx, mut timer_rx) = mpsc::unbounded_channel();
        let (batch_tx, _batch_rx) = mpsc::unbounded_channel();

        av.set_callbacks(timer_tx, batch_tx);

        let request_id = Uuid::new_v4();
        let items = vec![create_batch_item("text1", request_id)];

        av.extend(items);

        // Should receive timer notification
        let timestamp = timer_rx.recv().await;
        assert!(timestamp.is_some());
    }

    #[tokio::test]
    async fn test_batch_callback_on_extend() {
        let mut av = ActiveVector::new();
        let (timer_tx, _timer_rx) = mpsc::unbounded_channel();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel();

        av.set_callbacks(timer_tx, batch_tx);

        let request_id = Uuid::new_v4();
        let items = vec![create_batch_item("text1", request_id)];

        av.extend(items);

        // Should receive batch notification
        let notification = batch_rx.recv().await;
        assert!(notification.is_some());
    }

    #[test]
    fn test_slice_zero_max_size() {
        let mut av = ActiveVector::new();
        let request_id = Uuid::new_v4();
        let items = vec![create_batch_item("text1", request_id)];

        av.extend(items);
        let extracted = av.slice(0);

        assert!(extracted.is_empty());
        assert_eq!(av.len(), 1); // Original item should remain
    }
}
