use std::sync::mpsc;

pub(super) trait Accounting {
    fn queued(&self) -> u64;
    fn discarded(&self);
    fn finished(&self, sequence: u64);
}

pub(super) struct Entry<T: Accounting> {
    resource: T,
    sequence: u64,
    delivered: bool,
}

impl<T: Accounting> Entry<T> {
    pub(super) fn resource(&self) -> &T {
        &self.resource
    }
}

impl<T: Accounting> Drop for Entry<T> {
    fn drop(&mut self) {
        if !self.delivered {
            self.resource.discarded();
        }
        self.resource.finished(self.sequence);
    }
}

pub(super) struct Sender<T: Accounting>(mpsc::Sender<Entry<T>>);

impl<T: Accounting> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Accounting> std::fmt::Debug for Sender<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Sender").finish_non_exhaustive()
    }
}

impl<T: Accounting> Sender<T> {
    pub(super) fn send(&self, resource: T) -> Result<(), ()> {
        let sequence = resource.queued();
        self.0
            .send(Entry {
                resource,
                sequence,
                delivered: false,
            })
            .map_err(|_| ())
    }
}

pub(super) struct Receiver<T: Accounting>(mpsc::Receiver<Entry<T>>);

impl<T: Accounting> Receiver<T> {
    pub(super) fn try_iter(&self) -> impl Iterator<Item = Entry<T>> + '_ {
        self.0.try_iter().map(|mut entry| {
            entry.delivered = true;
            entry
        })
    }
}

pub(super) fn channel<T: Accounting>() -> (Sender<T>, Receiver<T>) {
    let (sender, receiver) = mpsc::channel();
    (Sender(sender), Receiver(receiver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Counts {
        queued: AtomicUsize,
        discarded: AtomicUsize,
        tracker: super::super::debug_queue_tracker::QueueTracker,
    }

    impl Default for Counts {
        fn default() -> Self {
            Self {
                queued: AtomicUsize::new(0),
                discarded: AtomicUsize::new(0),
                tracker: super::super::debug_queue_tracker::QueueTracker::new(),
            }
        }
    }

    impl Accounting for Arc<Counts> {
        fn queued(&self) -> u64 {
            self.queued.fetch_add(1, Ordering::Relaxed);
            self.tracker.submit()
        }

        fn discarded(&self) {
            self.discarded.fetch_add(1, Ordering::Relaxed);
        }

        fn finished(&self, sequence: u64) {
            self.tracker.complete(sequence);
        }
    }

    #[test]
    fn receiver_retirement_accounts_for_pending_resources() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        sender.send(counts.clone()).unwrap();
        drop(receiver);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rejected_send_is_discarded_exactly_once() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        drop(receiver);
        assert!(sender.send(counts.clone()).is_err());
        assert_eq!(counts.queued.load(Ordering::Relaxed), 1);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn received_resources_are_not_discarded() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        sender.send(counts.clone()).unwrap();
        assert_eq!(receiver.try_iter().count(), 1);
        drop(receiver);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retirement_discards_only_resources_not_received() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        sender.send(counts.clone()).unwrap();
        sender.clone().send(counts.clone()).unwrap();
        let resource = receiver.try_iter().next().unwrap();
        drop(receiver);
        assert_eq!(counts.queued.load(Ordering::Relaxed), 2);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 1);
        drop(resource);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sender_clones_can_submit_after_receiver_retirement() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        let second = sender.clone();
        sender.send(counts.clone()).unwrap();
        drop(receiver);
        assert!(second.send(counts.clone()).is_err());
        assert_eq!(counts.queued.load(Ordering::Relaxed), 2);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn concurrent_send_and_retirement_account_for_every_request() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..1000 {
                    let _ = sender.send(counts.clone());
                }
            });
            drop(receiver);
        });
        assert_eq!(counts.queued.load(Ordering::Relaxed), 1000);
        assert_eq!(counts.discarded.load(Ordering::Relaxed), 1000);
        assert_eq!(counts.tracker.snapshot().pending, 0);
    }

    #[test]
    fn newer_channel_drains_do_not_hide_a_stuck_request() {
        let counts = Arc::new(Counts::default());
        let (main_sender, main_receiver) = channel();
        let (surface_sender, surface_receiver) = channel();
        main_sender.send(counts.clone()).unwrap();
        surface_sender.send(counts.clone()).unwrap();
        assert_eq!(surface_receiver.try_iter().count(), 1);
        let snapshot = counts.tracker.snapshot();
        assert_eq!((snapshot.submitted, snapshot.oldest, snapshot.pending), (2, 1, 1));
        drop(main_receiver);
        assert_eq!(counts.tracker.snapshot().pending, 0);
    }

    #[test]
    fn received_request_stays_pending_through_cleanup_body() {
        let counts = Arc::new(Counts::default());
        let (sender, receiver) = channel();
        sender.send(counts.clone()).unwrap();
        let received = receiver.try_iter().next().unwrap();
        assert!(Arc::ptr_eq(received.resource(), &counts));
        assert_eq!(counts.tracker.snapshot().oldest, 1);
        drop(received);
        assert_eq!(counts.tracker.snapshot().oldest, 0);
    }
}
