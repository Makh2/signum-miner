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
//! What was learned is kept in a file next to the miner's config, so a restart
//! does not cost another round of refusals, and neither bound is trusted
//! forever: for a share to be refused again is the only way the limits move, so
//! a pool that raises its limit would otherwise stay filtered out for good.
//!
//! The two bounds are forgotten on their own clocks, because they cost
//! different things. A ceiling that has become stricter than the pool silently
//! drops shares the pool would have accepted, so a ceiling is trusted for
//! [`CEILING_TTL`] (hours). A floor only ever drops the best shares that can
//! exist, so it is trusted for [`FLOOR_TTL`] (a day) and costs nothing while
//! it lives.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A refusal at or above this many seconds is taken as evidence about the
/// ceiling. Refusals below it say nothing about the ceiling and are treated as
/// extreme values instead - see `Limit::floor`.
const MIN_CEILING_DEADLINE: u64 = 2;

/// How much of the height is dropped when bucketing chains. Mining two chains
/// through one proxy mixes their submissions, and their heights are far apart,
/// while one bucket lasts days at real block times.
const HEIGHT_BUCKET_SHIFT: u64 = 16;

/// How long a learned ceiling is trusted. Ceilings move (a testnet of ours
/// refused 35s, accepted 38s, and hours later refused 3s), and a ceiling that
/// has become stricter than the pool drops shares silently, so it is learned
/// again a few times a day.
pub const CEILING_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// How long a learned floor is trusted. A floor only ever drops the best shares
/// that can exist, so it can be kept for a day.
pub const FLOOR_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Layout version of the file. Bumping it makes older files be ignored.
const FORMAT_VERSION: u32 = 2;

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

/// Was something learned at `learned_at` less than `ttl` ago? `0` means nothing
/// was ever learned.
fn fresh(learned_at: u64, now: u64, ttl: Duration) -> bool {
    learned_at != 0 && now.saturating_sub(learned_at) <= ttl.as_secs()
}

#[derive(Clone, Copy)]
struct Limit {
    /// Largest deadline worth submitting, `NO_LIMIT` while unknown.
    max_deadline: u64,
    /// When the ceiling was learned, in seconds since the epoch.
    ceiling_at: u64,
    /// Largest refused deadline below the ceiling threshold, if any.
    floor: Option<u64>,
    /// When the floor was learned, in seconds since the epoch.
    floor_at: u64,
}

impl Default for Limit {
    fn default() -> Self {
        Limit {
            max_deadline: NO_LIMIT,
            ceiling_at: 0,
            floor: None,
            floor_at: 0,
        }
    }
}

impl Limit {
    /// The largest deadline worth submitting, while it is still trusted.
    fn ceiling(&self, now: u64) -> Option<u64> {
        (self.max_deadline != NO_LIMIT && fresh(self.ceiling_at, now, CEILING_TTL))
            .then_some(self.max_deadline)
    }

    /// The smallest refused deadline, while it is still trusted.
    fn floor(&self, now: u64) -> Option<u64> {
        self.floor.filter(|_| fresh(self.floor_at, now, FLOOR_TTL))
    }

    /// Does this entry still say anything?
    fn is_trusted(&self, now: u64) -> bool {
        self.ceiling(now).is_some() || self.floor(now).is_some()
    }
}

/// The file format. Deadlines live in the same seconds as everywhere else, the
/// keys are chain buckets (`height >> 16`) so two chains mined through one
/// proxy keep their own limits, and each bound carries the moment it was
/// learned so that they can be forgotten independently.
#[derive(Serialize, Deserialize)]
struct StoredLimits {
    version: u32,
    limits: HashMap<u64, StoredLimit>,
}

#[derive(Serialize, Deserialize, Default)]
struct StoredLimit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_deadline: Option<u64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    ceiling_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    floor: Option<u64>,
    #[serde(default, skip_serializing_if = "is_zero")]
    floor_at: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

impl From<Limit> for StoredLimit {
    fn from(limit: Limit) -> Self {
        StoredLimit {
            max_deadline: (limit.max_deadline != NO_LIMIT).then_some(limit.max_deadline),
            ceiling_at: limit.ceiling_at,
            floor: limit.floor,
            floor_at: limit.floor_at,
        }
    }
}

impl From<StoredLimit> for Limit {
    fn from(stored: StoredLimit) -> Self {
        Limit {
            max_deadline: stored.max_deadline.unwrap_or(NO_LIMIT),
            ceiling_at: stored.ceiling_at,
            floor: stored.floor,
            floor_at: stored.floor_at,
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

/// The file the limits are kept in for a given miner config: `scavenger.yaml`
/// keeps them in `scavenger.limits.json` beside it.
fn limits_file_for(config: &Path) -> PathBuf {
    let name = config
        .file_stem()
        .map(|stem| format!("{}.limits.json", stem.to_string_lossy()))
        .unwrap_or_else(|| "submission-limits.json".to_owned());
    config.with_file_name(name)
}

pub struct SubmissionLimits {
    limits: Mutex<HashMap<u64, Limit>>,
    file: Option<PathBuf>,
    /// The file is written on every refusal; if that never works, say so once.
    warned_about_file: AtomicBool,
}

impl Default for SubmissionLimits {
    fn default() -> Self {
        SubmissionLimits {
            limits: Mutex::new(HashMap::new()),
            file: None,
            warned_about_file: AtomicBool::new(false),
        }
    }
}

impl SubmissionLimits {
    /// Limits that are learned for this run only.
    pub fn new() -> Self {
        Self::default()
    }

    /// Limits kept next to the miner's config file, learned again when the
    /// stored ones have outlived their TTL.
    pub fn load_for_config(config: &Path) -> Self {
        let file = limits_file_for(config);
        let limits = match fs::read_to_string(&file) {
            Ok(text) => match serde_json::from_str::<StoredLimits>(&text) {
                Ok(stored) if stored.version == FORMAT_VERSION => {
                    Limit::from_stored(stored, unix_now())
                }
                Ok(stored) => {
                    warn!(
                        "ignoring {} (format version {}, this build speaks {})",
                        file.display(),
                        stored.version,
                        FORMAT_VERSION
                    );
                    HashMap::new()
                }
                Err(err) => {
                    warn!("ignoring {}, it does not parse: {}", file.display(), err);
                    HashMap::new()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(err) => {
                warn!("cannot read {}: {}", file.display(), err);
                HashMap::new()
            }
        };

        if !limits.is_empty() {
            info!(
                "loaded learned submission limits for {} chain(s) from {}",
                limits.len(),
                file.display()
            );
        }

        SubmissionLimits {
            limits: Mutex::new(limits),
            file: Some(file),
            warned_about_file: AtomicBool::new(false),
        }
    }

    /// Pretend a chain's ceiling was learned this many seconds ago, so the TTL
    /// can be tested without waiting hours.
    #[cfg(test)]
    fn backdate_ceiling(&self, height: u64, seconds: u64) {
        if let Some(limit) = self.limits.lock().unwrap().get_mut(&Self::bucket(height)) {
            limit.ceiling_at = limit.ceiling_at.saturating_sub(seconds);
        }
    }

    /// The same for the floor, which is trusted for longer.
    #[cfg(test)]
    fn backdate_floor(&self, height: u64, seconds: u64) {
        if let Some(limit) = self.limits.lock().unwrap().get_mut(&Self::bucket(height)) {
            limit.floor_at = limit.floor_at.saturating_sub(seconds);
        }
    }

    fn bucket(height: u64) -> u64 {
        height >> HEIGHT_BUCKET_SHIFT
    }

    /// Is this normalized deadline worth sending to the pool?
    pub fn allows(&self, height: u64, deadline: u64) -> bool {
        let now = unix_now();
        match self.limits.lock().unwrap().get(&Self::bucket(height)).copied() {
            None => true,
            Some(limit) => {
                // A forgotten bound is no bound: the pool decides again.
                if limit.floor(now).is_some_and(|floor| deadline <= floor) {
                    return false;
                }
                limit.ceiling(now).is_none_or(|ceiling| deadline <= ceiling)
            }
        }
    }

    /// The largest normalized deadline worth submitting for this height.
    pub fn max_deadline(&self, height: u64) -> u64 {
        self.limits
            .lock()
            .unwrap()
            .get(&Self::bucket(height))
            .and_then(|limit| limit.ceiling(unix_now()))
            .unwrap_or(NO_LIMIT)
    }

    /// What is known about a chain, phrased for the log. `None` when nothing
    /// about it is trusted.
    pub fn describe(&self, height: u64) -> Option<String> {
        let now = unix_now();
        let limits = self.limits.lock().unwrap();
        let limit = limits.get(&Self::bucket(height))?;

        let mut parts = Vec::new();
        if let Some(ceiling) = limit.ceiling(now) {
            parts.push(format!("no deadline above {}s", ceiling));
        }
        if let Some(floor) = limit.floor(now) {
            parts.push(format!("no deadline at or below {}s", floor));
        }

        (!parts.is_empty()).then(|| parts.join(", "))
    }

    /// A pool refused a share with this deadline: learn from it.
    /// Returns what was learned, or `None` when the refusal taught nothing.
    pub fn on_rejected(&self, height: u64, deadline: u64) -> Option<Learned> {
        let now = unix_now();
        let learned = {
            let mut limits = self.limits.lock().unwrap();
            let limit = limits.entry(Self::bucket(height)).or_default();

            if deadline < MIN_CEILING_DEADLINE {
                // Too good to say anything about the ceiling; remember the
                // extreme itself. Only a tighter one is news, and a forgotten
                // one is not worth measuring against.
                if limit.floor(now).is_some_and(|floor| floor >= deadline) {
                    return None;
                }
                limit.floor = Some(deadline);
                limit.floor_at = now;
                Some(Learned::Floor(deadline))
            } else {
                // Same for the ceiling: a refused share whose deadline is above
                // what is left of it teaches nothing.
                if limit
                    .ceiling(now)
                    .is_some_and(|ceiling| ceiling < deadline)
                {
                    return None;
                }
                limit.max_deadline = deadline - 1;
                limit.ceiling_at = now;
                Some(Learned::Ceiling(limit.max_deadline))
            }
        };

        if learned.is_some() {
            self.save();
        }

        learned
    }

    fn save(&self) {
        let Some(file) = &self.file else {
            return;
        };
        let now = unix_now();
        let stored = {
            let limits = self.limits.lock().unwrap();
            StoredLimits {
                version: FORMAT_VERSION,
                limits: limits
                    .iter()
                    .filter(|(_, limit)| limit.is_trusted(now))
                    .map(|(bucket, limit)| (*bucket, StoredLimit::from(*limit)))
                    .collect(),
            }
        };

        let json = match serde_json::to_string_pretty(&stored) {
            Ok(json) => json,
            Err(err) => {
                self.warn_about_file(&format!("cannot serialize submission limits: {}", err));
                return;
            }
        };

        // Write beside the real file and move it over, so a crash cannot leave
        // a half-written file that would silently drop everything.
        let temporary = file.with_extension("json.tmp");
        if let Err(err) = fs::write(&temporary, json).and_then(|_| {
            fs::remove_file(file).or_else(|err| match err.kind() {
                std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(err),
            })?;
            fs::rename(&temporary, file)
        }) {
            self.warn_about_file(&format!("cannot write {}: {}", file.display(), err));
        }
    }

    fn warn_about_file(&self, message: &str) {
        if !self.warned_about_file.swap(true, Ordering::Relaxed) {
            warn!("{}, learned limits will only last for this run", message);
        }
    }
}

impl Limit {
    fn from_stored(stored: StoredLimits, now: u64) -> HashMap<u64, Limit> {
        stored
            .limits
            .into_iter()
            .map(|(bucket, limit)| (bucket, Limit::from(limit)))
            .filter(|(_, limit)| limit.is_trusted(now))
            .collect()
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

        // A refusal at 79s means the pool refuses 79s and above, so 78s is the
        // last deadline still worth sending.
        assert_eq!(limits.on_rejected(height, 79), Some(Learned::Ceiling(78)));
        assert_eq!(limits.max_deadline(height), 78);
        assert!(limits.allows(height, 77));
        assert!(limits.allows(height, 78));
        assert!(!limits.allows(height, 79));

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
    fn describes_what_is_learned() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        assert_eq!(limits.describe(height), None);

        assert_eq!(limits.on_rejected(height, 35), Some(Learned::Ceiling(34)));
        assert_eq!(
            limits.describe(height).as_deref(),
            Some("no deadline above 34s")
        );

        assert_eq!(limits.on_rejected(height, 1), Some(Learned::Floor(1)));
        assert_eq!(
            limits.describe(height).as_deref(),
            Some("no deadline above 34s, no deadline at or below 1s")
        );
    }

    #[test]
    fn forgets_a_ceiling_after_its_ttl() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        assert_eq!(limits.on_rejected(height, 35), Some(Learned::Ceiling(34)));
        assert!(!limits.allows(height, 35));

        // A pool that raised its limit must not stay filtered out for good.
        limits.backdate_ceiling(height, CEILING_TTL.as_secs() + 1);
        assert!(limits.allows(height, 35));
        assert!(limits.allows(height, 100_000));
        assert_eq!(limits.max_deadline(height), NO_LIMIT);

        // And the forgotten value must not shade the next answer: the chain is
        // learned from scratch, so a refusal at 200s really does mean 199s.
        assert_eq!(limits.on_rejected(height, 200), Some(Learned::Ceiling(199)));
        assert!(!limits.allows(height, 200));
        assert!(limits.allows(height, 199));
    }

    #[test]
    fn keeps_the_floor_when_the_ceiling_is_forgotten() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        assert_eq!(limits.on_rejected(height, 35), Some(Learned::Ceiling(34)));
        assert_eq!(limits.on_rejected(height, 0), Some(Learned::Floor(0)));

        // The ceiling is trusted for hours, the floor for a day: letting the
        // ceiling go must not throw the floor away with it.
        limits.backdate_ceiling(height, CEILING_TTL.as_secs() + 1);
        assert!(limits.allows(height, 5_000));
        assert_eq!(limits.max_deadline(height), NO_LIMIT);
        assert!(!limits.allows(height, 0));

        // ... and the other way round: letting the floor go is its own clock.
        limits.backdate_floor(height, FLOOR_TTL.as_secs() + 1);
        assert!(limits.allows(height, 0));
    }

    #[test]
    fn forgets_a_floor_after_its_ttl() {
        let limits = SubmissionLimits::new();
        let height = 998_828;

        assert_eq!(limits.on_rejected(height, 0), Some(Learned::Floor(0)));
        assert!(!limits.allows(height, 0));

        limits.backdate_floor(height, FLOOR_TTL.as_secs() + 1);
        assert!(limits.allows(height, 0));

        // A pool refusing it again is news once more.
        assert_eq!(limits.on_rejected(height, 0), Some(Learned::Floor(0)));
    }

    #[test]
    fn recognises_the_pool_message() {
        assert!(is_deadline_too_high("Deadline exceeds maximum allowed deadline"));
        assert!(is_deadline_too_high("deadline too high"));
        assert!(!is_deadline_too_high("limit exceeded"));
        assert!(!is_deadline_too_high("success"));
        assert!(!is_deadline_too_high(""));
    }

    fn temporary_file(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("signum-miner-test-{}-{}.yaml", name, unique))
    }

    #[test]
    fn survives_a_restart_through_the_file() {
        let config = temporary_file("limits");
        let height = 998_828;

        let first_run = SubmissionLimits::load_for_config(&config);
        assert!(first_run.allows(height, 5)); // nothing to load yet
        assert_eq!(first_run.on_rejected(height, 35), Some(Learned::Ceiling(34)));
        assert_eq!(first_run.on_rejected(height, 0), Some(Learned::Floor(0)));
        // Any other chain is untouched.
        assert_eq!(first_run.on_rejected(1_595_708, 79), Some(Learned::Ceiling(78)));

        let second_run = SubmissionLimits::load_for_config(&config);
        assert!(second_run.allows(height, 33));
        assert!(second_run.allows(height, 34));
        assert!(!second_run.allows(height, 35));
        assert!(!second_run.allows(height, 0));
        assert!(second_run.allows(height, 1));
        assert!(second_run.allows(1_595_708, 78));
        assert!(!second_run.allows(1_595_708, 79));
        assert_eq!(
            second_run.describe(height).as_deref(),
            Some("no deadline above 34s, no deadline at or below 0s")
        );

        // The file lives beside the config, under a name of its own.
        let file = limits_file_for(&config);
        assert_eq!(
            file.file_name().unwrap(),
            config.with_extension("limits.json").file_name().unwrap()
        );
        assert!(file.exists());
        fs::remove_file(&file).unwrap();
        assert!(!file.exists());
    }

    #[test]
    fn reads_nothing_from_a_corrupt_file() {
        let config = temporary_file("corrupt");
        let file = limits_file_for(&config);
        fs::write(&file, "{ this is not the limits }").unwrap();

        let limits = SubmissionLimits::load_for_config(&config);
        assert!(limits.allows(998_828, 100_000));
        assert_eq!(limits.max_deadline(998_828), NO_LIMIT);

        fs::remove_file(&file).unwrap();
    }

    #[test]
    fn reads_nothing_from_the_previous_layout() {
        // Layout 1 kept one timestamp for both bounds. It is ignored rather
        // than guessed at, so a chain is learned again once.
        let config = temporary_file("old-layout");
        let file = limits_file_for(&config);
        fs::write(
            &file,
            r#"{"version": 1, "limits": {"15": {"max_deadline": 288, "learned_at": 1793181581}}}"#,
        )
        .unwrap();

        let limits = SubmissionLimits::load_for_config(&config);
        assert!(limits.allows(998_828, 100_000));
        assert_eq!(limits.max_deadline(998_828), NO_LIMIT);
        assert_eq!(limits.describe(998_828), None);

        fs::remove_file(&file).unwrap();
    }
}
