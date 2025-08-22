//! Timer implementation for batching timeout functionality.

use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{sleep_until, Instant};

/// Timer that triggers notifications after a delay, with ability to reset.
#[derive(Debug)]
pub struct Timer {
    /// Handle to the current timer task (if any).
    current_task: Option<JoinHandle<()>>,
    /// Channel sender for timer expiration notifications.
    #[allow(dead_code)]
    sender: mpsc::UnboundedSender<()>,
    /// Channel receiver for timer expiration notifications.
    receiver: Option<mpsc::UnboundedReceiver<()>>,
}

impl Timer {
    /// Create a new Timer with an unbounded notification channel.
    #[must_use]
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            current_task: None,
            sender,
            receiver: Some(receiver),
        }
    }

    /// Take ownership of the receiver channel.
    /// This can only be called once.
    #[must_use]
    pub const fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<()>> {
        self.receiver.take()
    }

    /// Set a new timer to trigger after the specified duration from the given timestamp.
    /// This cancels any existing timer.
    #[allow(dead_code)]
    pub fn set(&mut self, duration: Duration, timestamp: Instant) {
        // Cancel existing timer
        if let Some(task) = self.current_task.take() {
            task.abort();
        }

        // Calculate when the timer should fire
        let fire_time = timestamp + duration;
        let sender = self.sender.clone();

        // Start new timer task
        let task = tokio::spawn(async move {
            sleep_until(fire_time).await;
            // Ignore send errors - receiver might be dropped
            let _ = sender.send(());
        });

        self.current_task = Some(task);
    }

    /// Check if there's an active timer.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        self.current_task
            .as_ref()
            .is_some_and(|task| !task.is_finished())
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
        Self::new()
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
    async fn test_timer_creation() {
        let timer = Timer::new();
        assert!(!timer.is_active());
    }

    #[tokio::test]
    async fn test_timer_fires() {
        let mut timer = Timer::new();
        let mut receiver = timer.take_receiver().expect("Should have receiver");

        let now = Instant::now();
        timer.set(Duration::from_millis(50), now);
        assert!(timer.is_active());

        // Should receive notification within reasonable time
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_timer_reset_cancels_previous() {
        let mut timer = Timer::new();
        let mut receiver = timer.take_receiver().expect("Should have receiver");

        let now = Instant::now();

        // Set a long timer
        timer.set(Duration::from_millis(500), now);
        assert!(timer.is_active());

        // Wait a bit then reset with shorter timer
        sleep(Duration::from_millis(50)).await;
        timer.set(Duration::from_millis(50), Instant::now());

        // Should receive notification from the second timer, not the first
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Should not receive a second notification from the cancelled timer
        let result = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_timer_cancel() {
        let mut timer = Timer::new();
        let mut receiver = timer.take_receiver().expect("Should have receiver");

        let now = Instant::now();
        timer.set(Duration::from_millis(100), now);
        assert!(timer.is_active());

        // Cancel the timer
        timer.cancel();
        assert!(!timer.is_active());

        // Should not receive notification
        let result = timeout(Duration::from_millis(200), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_timer_with_past_timestamp() {
        let mut timer = Timer::new();
        let mut receiver = timer.take_receiver().expect("Should have receiver");

        // Set timer with timestamp in the past
        let past = Instant::now() - Duration::from_millis(100);
        timer.set(Duration::from_millis(50), past);

        // Should fire immediately since the target time is in the past
        let result = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_timer_drop_cancels() {
        let mut receiver = {
            let mut timer = Timer::new();
            let receiver = timer.take_receiver().expect("Should have receiver");

            let now = Instant::now();
            timer.set(Duration::from_millis(100), now);
            assert!(timer.is_active());

            receiver
            // timer is dropped here, which calls cancel()
        };

        // Give some time for any potential timer to fire
        sleep(Duration::from_millis(150)).await;

        // The receiver should either get nothing or the channel should be closed
        // Since we dropped the timer, the sender is also dropped, so the channel is closed
        let result = receiver.recv().await;
        assert!(result.is_none()); // Channel closed
    }

    #[tokio::test]
    async fn test_multiple_timer_resets() {
        let mut timer = Timer::new();
        let mut receiver = timer.take_receiver().expect("Should have receiver");

        let now = Instant::now();

        // Set and reset multiple times quickly
        for i in 0..5 {
            timer.set(Duration::from_millis(100 + i * 10), now);
        }

        // Should only get one notification from the last timer
        let result = timeout(Duration::from_millis(300), receiver.recv()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Should not get additional notifications
        let result = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[test]
    fn test_take_receiver_only_once() {
        let mut timer = Timer::new();

        let receiver1 = timer.take_receiver();
        assert!(receiver1.is_some());

        let receiver2 = timer.take_receiver();
        assert!(receiver2.is_none());
    }
}
