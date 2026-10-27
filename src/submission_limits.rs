//! Learning the worst deadline a pool still accepts.
//!
//! Pools refuse shares whose deadline is worse than a limit of their own
//! ("Deadline exceeds maximum allowed deadline") and never advertise that
//! limit: `getMiningInfo` carries no such field, on either pool. The limit is
//! not a stable number of seconds either - it follows the pool's own view of
//! the round, and on a small chain the base target moves a lot - so the only
//! way to know it is to learn it from the pool's answers.
//!
//! Every refusal at a plausible deadline lowers the limit for the chain the
//! share belonged to: if the pool refused a share, its limit is below it.
//! Refusals of implausibly good shares are ignored - pools refuse those for
//! other reasons (a zero-second deadline is a known case), and learning from
//! them would silence submissions for good.
//!
//! The learned limit only ever goes down, so a restart is the way to start
//! over (for example after a pool raises its limit).

use std::collections::HashMap;
use std::sync::Mutex;

/// A refusal at or below this many seconds is not taken as evidence that the
/// deadline was too high.
const MIN_LEARNABLE_DEADLINE: u64 = 2;

/// How much of the height is dropped when bucketing chains. Mining two chains
/// through one proxy mixes their submissions, and their heights are far apart,
/// while one bucket lasts days at real block times.
const HEIGHT_BUCKET_SHIFT: u64 = 16;

/// No limit learned yet.
pub const NO_LIMIT: u64 = u64::MAX;

/// Does this pool error say that the share's deadline was too high?
pub fn is_deadline_too_high(message: &str) -> bool {
    let message = message.to_lowercase();
    message.contains("exceeds maximum") || message.contains("deadline too high")
}

#[derive(Default)]
pub struct SubmissionLimits {
    limits: Mutex<HashMap<u64, u64>>,
}

impl SubmissionLimits {
    pub fn new() -> Self {
        Self::default()
    }

    fn bucket(height: u64) -> u64 {
        height >> HEIGHT_BUCKET_SHIFT
    }

    /// The largest normalized deadline worth submitting for this height.
    pub fn max_deadline(&self, height: u64) -> u64 {
        self.limits
            .lock()
            .unwrap()
            .get(&Self::bucket(height))
            .copied()
            .unwrap_or(NO_LIMIT)
    }

    /// A pool refused a share with this deadline: the limit is below it.
    /// Returns the new limit when the refusal actually taught us something.
    pub fn on_rejected(&self, height: u64, deadline: u64) -> Option<u64> {
        if deadline < MIN_LEARNABLE_DEADLINE {
            return None;
        }
        let mut limits = self.limits.lock().unwrap();
        let limit = limits.entry(Self::bucket(height)).or_insert(NO_LIMIT);
        if deadline <= *limit {
            *limit = deadline - 1;
            return Some(*limit);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_the_limit_from_refusals() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        // Nothing learned yet.
        assert_eq!(limits.max_deadline(height), NO_LIMIT);

        // A refusal at 79s means the pool refuses 79s and above.
        assert_eq!(limits.on_rejected(height, 79), Some(78));
        assert_eq!(limits.max_deadline(height), 78);

        // A share at 78s is still allowed, and a refusal at 35s tightens it.
        assert_eq!(limits.on_rejected(height, 35), Some(34));
        assert_eq!(limits.max_deadline(height), 34);

        // A refusal above the known limit teaches nothing.
        assert_eq!(limits.on_rejected(height, 200), None);
        assert_eq!(limits.max_deadline(height), 34);
    }

    #[test]
    fn keeps_chains_apart() {
        let limits = SubmissionLimits::new();
        let mainnet = 1_595_708;
        let testnet = 998_828;

        assert_eq!(limits.on_rejected(testnet, 79), Some(78));

        // The limit learned on the testnet must not filter mainnet shares.
        assert_eq!(limits.max_deadline(mainnet), NO_LIMIT);
        assert_eq!(limits.max_deadline(testnet), 78);
    }

    #[test]
    fn ignores_implausible_refusals() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        // A refused zero-second share is not a "deadline too high" signal.
        assert_eq!(limits.on_rejected(height, 0), None);
        assert_eq!(limits.on_rejected(height, 1), None);
        assert_eq!(limits.max_deadline(height), NO_LIMIT);
    }

    #[test]
    fn recognises_the_pool_message() {
        assert!(is_deadline_too_high("Deadline exceeds maximum allowed deadline"));
        assert!(is_deadline_too_high("deadline too high"));
        assert!(!is_deadline_too_high("limit exceeded"));
        assert!(!is_deadline_too_high("success"));
        assert!(!is_deadline_too_high(""));
    }
}
