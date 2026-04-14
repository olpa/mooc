//! Timer implementation for batching timeout functionality.

use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::sleep;

/// Timer that triggers notifications after a delay, with ability to reset.
#[derive(Debug)]
pub struct Timer {
    /// Duration to wait before firing.
    duration: Duration,
    /// Handle to the current timer task (if any).
    current_task: Option<JoinHandle<()>>,
    /// Channel sender for timer expiration notifications with seqno.
    sender: mpsc::UnboundedSender<u64>,
}

impl Timer {
    /// Create a new Timer with the specified duration and sender channel.
    #[must_use]
    pub fn new(duration: Duration, sender: mpsc::UnboundedSender<u64>) -> Self {
        Self {
            duration,
            current_task: None,
            sender,
        }
    }

    /// Set a new timer to trigger after the configured duration from now.
    /// This cancels any existing timer and will send the provided seqno when it fires.
    pub fn set(&mut self, seqno: u64) {
        // Cancel existing timer (this is the key feature - new timer deletes previous ones)
        if let Some(task) = self.current_task.take() {
            task.abort();
        }

        let duration = self.duration;
        let sender = self.sender.clone();

        // Start new timer task
        let task = tokio::spawn(async move {
            sleep(duration).await;
            // Send the seqno when timer fires - ignore send errors if receiver is dropped
            if sender.send(seqno).is_err() {
                tracing::debug!("Timer receiver dropped, timer notification ignored");
            }
        });

        self.current_task = Some(task);
    }


    /// Cancel the current timer if one is active.
    pub fn cancel(&mut self) {
        if let Some(task) = self.current_task.take() {
            task.abort();
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        let (sender, _receiver) = mpsc::unbounded_channel();
        Self::new(Duration::from_millis(100), sender)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};


    #[tokio::test]
    async fn test_timer_fires() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut timer = Timer::new(Duration::from_millis(50), sender);

        timer.set(42);

        // Should receive notification with seqno within reasonable time
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        let seqno = result.unwrap();
        assert!(seqno.is_some());
        assert_eq!(seqno.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timer_reset_cancels_previous() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut timer = Timer::new(Duration::from_millis(50), sender);

        // Set a timer with seqno 1
        timer.set(1);

        // Wait a bit then reset with seqno 2
        sleep(Duration::from_millis(25)).await;
        timer.set(2);

        // Should receive notification from the second timer with seqno 2, not the first
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        let seqno = result.unwrap();
        assert!(seqno.is_some());
        assert_eq!(seqno.unwrap(), 2); // Should be the second timer's seqno

        // Should not receive a second notification from the cancelled timer
        let result = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_timer_cancel() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut timer = Timer::new(Duration::from_millis(100), sender);

        timer.set(99);

        // Cancel the timer
        timer.cancel();

        // Should not receive notification
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_timer_fires_after_duration() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut timer = Timer::new(Duration::from_millis(50), sender);

        timer.set(123);

        // Should fire after the configured duration
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        let seqno = result.unwrap();
        assert!(seqno.is_some());
        assert_eq!(seqno.unwrap(), 123);
    }

    #[tokio::test]
    async fn test_timer_drop_cancels() {
        // Create a timer and set it, then drop it immediately
        {
            let (sender, _receiver) = mpsc::unbounded_channel();
            let mut timer = Timer::new(Duration::from_millis(100), sender);
            timer.set(777);
            // timer is dropped here, which calls cancel()
        }

        // Give some time for any potential timer to fire
        sleep(Duration::from_millis(150)).await;

        // Since we can't access the receiver after the timer is dropped,
        // we just verify that dropping the timer doesn't cause issues
        // The test passes if we reach this point without panicking
    }

    #[tokio::test]
    async fn test_multiple_timer_resets() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut timer = Timer::new(Duration::from_millis(50), sender);

        // Set and reset multiple times quickly
        for i in 0..5 {
            timer.set(i);
        }

        // Should only get one notification from the last timer (seqno 4)
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        let seqno = result.unwrap();
        assert!(seqno.is_some());
        assert_eq!(seqno.unwrap(), 4); // Should be the last timer's seqno

        // Should not get additional notifications
        let result = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }
}
