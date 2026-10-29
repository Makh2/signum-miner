# Notes on this fork

This branch is a build of `signum-miner` whose **OpenCL path actually works**, plus
the fixes needed to keep it running for days. It is based on upstream `master`
(`4a16d45`, 2025-05-29) and is meant to be run through
[foxy-miner](https://github.com/foxypool/foxy-miner).

Read upstream's `README.md` for the miner itself; this file records what is
different here and what was learned by running it.

## Why a fork at all

* Upstream **never builds the `opencl` feature**: the release matrix in
  `.github/workflows/release.yml` lists only `simd_*` and `neon`, and the PR
  workflow runs `cargo build`/`cargo test` with default features (CPU only).
* On upstream `master` the feature does not even compile: `src/gpu_worker_async.rs`
  uses `futures::sync` from futures 0.1, which the 2025 migration to tokio
  removed. Nobody noticed, because nothing builds it.
* The OpenCL code is the oldest part of the project: `src/ocl.rs` was last
  touched in **January 2019** and contains ~57 `unwrap()` calls on OpenCL
  results. The buffer pipeline (`src/reader.rs`) dates from 2021 and assumes a
  buffer always comes back and a round always closes.

Running that on Windows, for days, with `useGpuMemMapping: true`, is what
exposed the rest.

## Building

```sh
cargo build --release --no-default-features --features opencl,simd_avx2
cargo test  --release --no-default-features --features opencl,simd_avx2
```

The tests are 18 with the GPU worker compiled in, 17 without. There is no test
coverage for the OpenCL paths themselves - they need a device - so changes there
rest on the build and on watching a live run.

Useful foxy-miner settings (`foxy-miner.yaml`), the ones that matter for this
build:

```yaml
minerConfig:
  cpuWorkers: 0            # GPU only
  gpuThreads: 1
  gpuWorkers: 5            # one worker per buffer
  useGpuMemMapping: true
  useGpuAsyncCompute: false
logToFile: true            # logs land in %USERPROFILE%\.config\foxy-miner\logs\foxy-miner.log
```

## What is different from upstream

Build and stability:

* the dead `gpu_worker_async.rs` is gone and `main.rs` uses `matches.get_flag`,
  so the `opencl` feature compiles at all;
* `CpuBuffer` owns its allocation layout instead of handing page-aligned memory
  to a `Vec<u8>` (which freed it with the wrong alignment and corrupted the heap
  on the first round);
* a share is sent as soon as it is found instead of being held back three
  seconds (`PrioRetry`; upstream lost the instant yield in the tokio migration);
* the pool's answer to a refused share is parsed and printed as one short line
  instead of a raw JSON dump:
  `refused: height=1599153, deadline=00:06:10: Deadline exceeds maximum allowed deadline`
  (the old line was red, wrapped over three, and repeated the account on every
  round).

Resilience - all of it from watching the miner go silent for hours:

* a drive task waits for a free buffer with a timeout, notices a replaced round
  while waiting and gives up after a minute with a line naming the drive, instead
  of blocking its pool thread for good;
* a send to a worker that is gone returns the buffer to the pool and logs it
  instead of panicking the drive task;
* the GPU worker hashes under `catch_unwind`, retries a window once, and always
  hands the buffer back, so a failed OpenCL call costs a window instead of a
  worker, and a dead worker no longer stops rounds from closing;
* a round that is still open after 90 seconds is closed by the watchdog with a
  line saying how many reader tasks reported - the round submits what it found
  instead of the miner going quiet.

Deliberately **not** here:

* the learned deadline limits (they were built, measured and removed, see below);
* anything that filters submissions on our side.

## What the log says

| line | meaning |
|---|---|
| `Starting scan of new block: height=…, scoop=… (chain)` | foxy handed the miner a round |
| `round finished: roundtime=…ms, speed=…` | the scan completed and the best share (if any) was submitted |
| `accepted: height=…, deadline=…` | the pool took the share |
| `refused: height=…, deadline=…: <pool text>` | the pool's answer; costs nothing |
| `reader …: no free buffer for 60s while scanning height=…` | the hashing side stopped returning buffers |
| `reader …: GPU worker N is gone` | that worker thread died; buffers are still returned |
| `GPU worker: hashing failed at height=… trying it again` | an OpenCL call panicked once |
| `GPU worker: giving up hashing after 5 failures` | that worker will not hash again until the miner is restarted |
| `round for height=… has not closed for 90s (N of 5 …)` | the watchdog closed a stuck round |

Deadlines in the log are humanized by foxy-miner with
`display = floor(ln(raw_seconds) * 43.79)`, so `00:06:10` is really about 1.3
hours. Without that conversion the numbers look like they contradict each other.

## What was removed, and why it matters

A learned deadline filter (ceiling from refusals, floor from refused extreme
values, kept in `*.limits.json`) was written, measured live and then deleted:

* refusals are **free** - the pool does not penalise them - so the filter bought
  nothing but risk;
* a ceiling learned from one refusal can be wrong (a refusal can be a round race
  rather than a limit), and everything above it is then dropped without a word.
  That is the only place in the miner where a share the pool would have taken can
  be lost;
* measured: accepted deadlines on `nam Signa` were 1.7 h and 10.6 h while
  refused ones ranged 1.0-23.1 h. The ranges overlap, so the value we print is
  not what the pool judges: with a secret phrase configured, foxy-miner forwards
  `secretPhrase` instead of the deadline (`lib/submission.js`), and the pool
  recomputes it from the nonce.

Consequence: the miner offers every share it finds, the pool answers, and on a
60 TiB farm against a 3.5 PiB network most answers are `Deadline exceeds maximum
allowed deadline` - that is a matter of scale, not of configuration.

## foxy-miner's schedule, for the record

From foxy-miner 2.7.0's sources (`lib/currentRoundManager.js`, `lib/proxy.js`):

* a round is only preempted by a pending round with a **higher `weight`**; with
  `weight: 8` (mainnet) against `5` (testnet), the testnet can never take the
  mainnet's place - it waits, and it runs in the mainnet's idle time;
* `doNotInterruptAbovePercent` protects the *current* round: a scan past that
  percentage is not abandoned, so on a lower-weight upstream it can only delay
  the higher-weight one (measured: a mainnet block waited up to ~5 s in 95
  minutes);
* `maxNumberOfChains` / `minWeight` drop a chain's blocks before they are queued
  at all, with an explicit line in the log.

## Still open

* The root cause of the OpenCL failures is not fixed, only survived: `ocl.rs`
  still unwraps every call. The next step, if the log shows failures, is to make
  a window's failure recoverable - retry with a bound, and rebuild the GPU
  context when the device itself is in trouble.
* The remaining `unwrap()`s on locks are no longer fatal (`utils::lock` recovers
  from a poisoned mutex), but the two startup-only ones in `miner.rs` and
  `reader.rs` are still plain `unwrap()`.
* No test covers the watchdog or the buffer timeout; both are pure enough to
  extract and test if that ever bites.
