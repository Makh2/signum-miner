//! Learning what a pool accepts, from its refusals.
//!
//! Pools refuse shares whose deadline is worse than a limit of their own
//! ("Deadline exceeds maximum allowed deadline") and never advertise that
//! limit: `getMiningInfo` carries no such field, on either pool. The limit is
//! not a stable number of seconds either - it follows the pool's own view of
//! the round, and on a small chain the base target moves a lot - so the only
//! way to know it is to learn it from the pool's answers.
//!
//! Two things are learned, both only from an actual refusal:
//!
//! * A refusal at a normal deadline lowers the ceiling for the chain the share
//!   belonged to: if the pool refused a share, its limit is below it.
//!
//! * A refusal of an extreme deadline - one of the best shares that can exist,
//!   a fraction of a second - is not a ceiling at all: a pool doing that
//!   refuses the very best shares for its own reasons (foxy-pool does, on the
//!   testnet), and lowering a ceiling to there would silence submissions for
//!   good. The extreme itself is remembered instead, and no longer offered.
//!   Only the exact values refused are dropped; a share a second better is
//!   untouched.
//!
//! Whatever is learned only ever tightens, so a restart is the way to start
//! over (for example after a pool raises its limit).

use std::collections::HashMap;
use std::sync::Mutex;

/// A refusal at or above this many seconds is taken as evidence about the
/// ceiling. Refusals below it say nothing about the ceiling and are treated as
/// extreme values instead - see `Limit::floor`.
const MIN_CEILING_DEADLINE: u64 = 2;

/// How much of the height is dropped when bucketing chains. Mining two chains
/// through one proxy mixes their submissions, and their heights are far apart,
/// while one bucket lasts days at real block times.
const HEIGHT_BUCKET_SHIFT: u64 = 16;

/// No ceiling learned yet.
pub const NO_LIMIT: u64 = u64::MAX;

/// Does this pool error say that the share's deadline was too high?
pub fn is_deadline_too_high(message: &str) -> bool {
    let message = message.to_lowercase();
    message.contains("exceeds maximum") || message.contains("deadline too high")
}

/// What a refusal taught us. Logged by the submission loop.
#[derive(Debug, PartialEq, Eq)]
pub enum Learned {
    /// The pool refuses deadlines above this many seconds.
    Ceiling(u64),
    /// The pool refuses deadlines at or below this many seconds.
    Floor(u64),
}

#[derive(Clone, Copy)]
struct Limit {
    /// Largest deadline worth submitting, `NO_LIMIT` while unknown.
    max_deadline: u64,
    /// Largest refused deadline below the ceiling threshold, if any.
    floor: Option<u64>,
}

impl Default for Limit {
    fn default() -> Self {
        Limit {
            max_deadline: NO_LIMIT,
            floor: None,
        }
    }
}

#[derive(Default)]
pub struct SubmissionLimits {
    limits: Mutex<HashMap<u64, Limit>>,
}

impl SubmissionLimits {
    pub fn new() -> Self {
        Self::default()
    }

    fn bucket(height: u64) -> u64 {
        height >> HEIGHT_BUCKET_SHIFT
    }

    /// Is this normalized deadline worth sending to the pool?
    pub fn allows(&self, height: u64, deadline: u64) -> bool {
        match self.limits.lock().unwrap().get(&Self::bucket(height)) {
            None => true,
            Some(limit) => match limit.floor {
                Some(floor) if deadline <= floor => false,
                _ => deadline < limit.max_deadline,
            },
        }
    }

    /// The largest normalized deadline worth submitting for this height.
    pub fn max_deadline(&self, height: u64) -> u64 {
        self.limits
            .lock()
            .unwrap()
            .get(&Self::bucket(height))
            .map(|limit| limit.max_deadline)
            .unwrap_or(NO_LIMIT)
    }

    /// A pool refused a share with this deadline: learn from it.
    /// Returns what was learned, or `None` when the refusal taught nothing.
    pub fn on_rejected(&self, height: u64, deadline: u64) -> Option<Learned> {
        let mut limits = self.limits.lock().unwrap();
        let limit = limits.entry(Self::bucket(height)).or_default();

        if deadline < MIN_CEILING_DEADLINE {
            // Too good to say anything about the ceiling; remember the extreme
            // itself. Only a tighter one is news.
            if limit.floor.map_or(true, |floor| deadline > floor) {
                limit.floor = Some(deadline);
                return Some(Learned::Floor(deadline));
            }
            return None;
        }

        if deadline <= limit.max_deadline {
            limit.max_deadline = deadline - 1;
            return Some(Learned::Ceiling(limit.max_deadline));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_the_ceiling_from_refusals() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        // Nothing learned yet.
        assert_eq!(limits.max_deadline(height), NO_LIMIT);
        assert!(limits.allows(height, 12_000));

        // A refusal at 79s means the pool refuses 79s and above.
        assert_eq!(limits.on_rejected(height, 79), Some(Learned::Ceiling(78)));
        assert_eq!(limits.max_deadline(height), 78);
        assert!(limits.allows(height, 77));
        assert!(!limits.allows(height, 78));

        // A share at 78s is still allowed, and a refusal at 35s tightens it.
        assert_eq!(limits.on_rejected(height, 35), Some(Learned::Ceiling(34)));
        assert_eq!(limits.max_deadline(height), 34);

        // A refusal above the known limit teaches nothing.
        assert_eq!(limits.on_rejected(height, 200), None);
        assert_eq!(limits.max_deadline(height), 34);
    }

    #[test]
    fn remembers_extreme_refusals() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        // The best possible share is worth a try while the pool hasn't refused it.
        assert!(limits.allows(height, 0));
        assert!(limits.allows(height, 1));

        // A refused zero-second share is remembered, without turning into a
        // ceiling: one second onwards is a normal share.
        assert_eq!(limits.on_rejected(height, 0), Some(Learned::Floor(0)));
        assert!(!limits.allows(height, 0));
        assert!(limits.allows(height, 1));
        assert!(limits.allows(height, 2));
        assert_eq!(limits.max_deadline(height), NO_LIMIT);

        // A refused one-second share tightens it, and repeating teaches nothing.
        assert_eq!(limits.on_rejected(height, 1), Some(Learned::Floor(1)));
        assert!(!limits.allows(height, 1));
        assert!(limits.allows(height, 2));
        assert_eq!(limits.on_rejected(height, 1), None);
        assert_eq!(limits.on_rejected(height, 0), None);
    }

    #[test]
    fn the_floor_only_tightens() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        assert_eq!(limits.on_rejected(height, 1), Some(Learned::Floor(1)));
        // A later refusal of something better adds nothing.
        assert_eq!(limits.on_rejected(height, 0), None);
        assert!(!limits.allows(height, 1));
        assert!(limits.allows(height, 2));
    }

    #[test]
    fn keeps_chains_apart() {
        let limits = SubmissionLimits::new();
        let mainnet = 1_595_708;
        let testnet = 998_828;

        assert_eq!(limits.on_rejected(testnet, 79), Some(Learned::Ceiling(78)));
        assert_eq!(limits.on_rejected(testnet, 0), Some(Learned::Floor(0)));

        // What was learned on the testnet must not filter mainnet shares.
        assert_eq!(limits.max_deadline(mainnet), NO_LIMIT);
        assert!(limits.allows(mainnet, 0));
        assert_eq!(limits.max_deadline(testnet), 78);
        assert!(!limits.allows(testnet, 0));
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
