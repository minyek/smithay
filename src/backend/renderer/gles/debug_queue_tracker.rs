use std::{collections::BTreeSet, sync::Mutex};

/// Cleanup submission order and outstanding requests for one resource class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueSnapshot {
    /// Highest submitted sequence, including requests rejected by retired queues.
    pub submitted: u64,
    /// Oldest outstanding sequence, or zero when all requests have completed.
    pub oldest: u64,
    /// Requests awaiting explicit cleanup or discard.
    pub pending: u64,
}

struct State {
    submitted: u64,
    pending: BTreeSet<u64>,
}

pub(super) struct QueueTracker(Mutex<State>);

impl QueueTracker {
    pub(super) const fn new() -> Self {
        Self(Mutex::new(State {
            submitted: 0,
            pending: BTreeSet::new(),
        }))
    }

    pub(super) fn submit(&self) -> u64 {
        let mut state = self.0.lock().unwrap();
        state.submitted += 1;
        let sequence = state.submitted;
        state.pending.insert(sequence);
        sequence
    }

    pub(super) fn complete(&self, sequence: u64) {
        self.0.lock().unwrap().pending.remove(&sequence);
    }

    pub(super) fn snapshot(&self) -> QueueSnapshot {
        let state = self.0.lock().unwrap();
        QueueSnapshot {
            submitted: state.submitted,
            oldest: state.pending.first().copied().unwrap_or(0),
            pending: state.pending.len() as u64,
        }
    }
}
