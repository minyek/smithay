use std::sync::mpsc;

pub(super) trait Accounting {
    fn queued(&self);
    fn discarded(&self);
}

struct Entry<T: Accounting>(Option<T>);

impl<T: Accounting> Drop for Entry<T> {
    fn drop(&mut self) {
        if let Some(resource) = &self.0 {
            resource.discarded();
        }
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
        resource.queued();
        self.0.send(Entry(Some(resource))).map_err(|_| ())
    }
}

pub(super) struct Receiver<T: Accounting>(mpsc::Receiver<Entry<T>>);

impl<T: Accounting> Receiver<T> {
    pub(super) fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        self.0.try_iter().map(|mut entry| entry.0.take().unwrap())
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

    #[derive(Default)]
    struct Counts {
        queued: AtomicUsize,
        discarded: AtomicUsize,
    }

    impl Accounting for Arc<Counts> {
        fn queued(&self) {
            self.queued.fetch_add(1, Ordering::Relaxed);
        }

        fn discarded(&self) {
            self.discarded.fetch_add(1, Ordering::Relaxed);
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
    }
}
