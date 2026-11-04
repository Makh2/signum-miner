# Changelog

All notable changes in this fork are documented here. Upstream does not keep a
changelog of its own; its releases are listed at
[signum-network/signum-miner/releases](https://github.com/signum-network/signum-miner/releases).

## v2.1.0 — first release of the fork (2026-11-04)

First tagged release of the maintenance fork maintained from a production mining
rig ([Makh2/signum-miner](https://github.com/Makh2/signum-miner)). It is based on
upstream `master` at `4a16d45` (release 2.0.0, 2025-05-29) plus a stack of fixes
accumulated while running the miner against the mainnet pool.

### Added

* The `opencl` feature now compiles, and the GPU path works end to end:
  `src/gpu_worker_async.rs` (dead code from before the tokio migration) is gone,
  the sync GPU worker is used, and `CpuBuffer` owns its allocation layout instead
  of handing a misaligned `Vec<u8>` to the buffer pipeline (which corrupted the
  heap on the first round with GPU memory mapping enabled).
* GPU worker rebuild: a worker that fails repeatedly or goes silent is replaced by a
  fresh one on its own thread; rounds keep running meanwhile.
* Round watchdog: a round still open after 90 s is closed by the miner with a
  one-line summary (`N of 5 …, M buffer(s) free`).
* Preemption grace period of 1.5 s so a new round lets an interrupted one leave
  before scanning without it.
* Priority retry (`PrioRetry`): send a found share at once; keep learned
  submission limits next to the config with a one-day TTL; print pool refusals as
  one short line instead of a raw JSON dump.
* Unit tests pinning the watchdog timeout, the preemption grace period, the
  aligned buffer and the panic-recovery paths.

### Fixed

* A bad generation signature (wrong length, non-hex) from the pool no longer panics
  the round setup - the block is ignored, once per distinct value.
* Failed hashing windows are retried once and then dropped; failed driver calls
  return an error with the driver answer instead of unwrapping.
* A poisoned mutex no longer stops the miner (recovery is only reachable once
  panics unwind).
* The process no longer aborts on the first panic: the build uses `panic = 'unwind'`
  and the critical tasks (round start, share submit, mining-info fetch) run behind
  `catch_unwind`.

### Changed

* `rand` 0.9 → 0.10 and `reqwest` 0.12 → 0.13, with the API and feature renames they
  require. The full test suite passes in all three configurations, including the
  live HTTPS round trip against the pool.

## v2.0.0 — upstream release this fork is based on

Upstream `master` at `4a16d45`. See the
[upstream releases](https://github.com/signum-network/signum-miner/releases).

## Not in this release, deliberately

* The learned deadline filter: written, measured live and removed again — refused
  shares are free and a ceiling learned from one refusal can drop shares the pool
  would have taken. See `FORK.md`.
* Submission-side filtering. Never added.
* Anything that was not validated on real hardware.
