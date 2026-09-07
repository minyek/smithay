#[path = "../src/backend/renderer/gles/debug_queue_tracker.rs"]
mod debug_queue_tracker;

use debug_queue_tracker::QueueTracker;

#[test]
fn newer_completions_do_not_hide_the_oldest_pending_request() {
    let tracker = QueueTracker::new();
    let stuck = tracker.submit();
    let watermark = tracker.snapshot().submitted;
    let later = tracker.submit();
    tracker.complete(later);
    let newest = tracker.submit();
    tracker.complete(newest);
    let snapshot = tracker.snapshot();
    assert_eq!((snapshot.submitted, snapshot.oldest, snapshot.pending), (3, 1, 1));
    assert!(snapshot.oldest <= watermark);
    tracker.complete(stuck);
    let empty = tracker.snapshot();
    assert_eq!((empty.submitted, empty.oldest, empty.pending), (3, 0, 0));
}

#[test]
fn oldest_advances_after_out_of_order_completion() {
    let tracker = QueueTracker::new();
    let first = tracker.submit();
    let second = tracker.submit();
    let third = tracker.submit();
    tracker.complete(second);
    assert_eq!(tracker.snapshot().oldest, 1);
    tracker.complete(first);
    assert_eq!(tracker.snapshot().oldest, 3);
    tracker.complete(third);
    assert_eq!(tracker.snapshot().oldest, 0);
}

#[test]
fn concurrent_submissions_and_completions_keep_oldest_pending_visible() {
    let tracker = QueueTracker::new();
    let stuck = tracker.submit();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..1000 {
                    let sequence = tracker.submit();
                    tracker.complete(sequence);
                    assert_eq!(tracker.snapshot().oldest, 1);
                }
            });
        }
    });
    let snapshot = tracker.snapshot();
    assert_eq!(
        (snapshot.submitted, snapshot.oldest, snapshot.pending),
        (4001, 1, 1)
    );
    tracker.complete(stuck);
    assert_eq!(tracker.snapshot().pending, 0);
}
