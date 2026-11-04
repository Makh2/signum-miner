# Notes on this fork

The `main` branch of this fork is a build of `signum-miner` whose **OpenCL path actually works**, plus
the fixes needed to keep it running for days. It is based on upstream `master`
(`4a16d45`, 2025-05-29 — upstream release 2.0.0) and is meant to be run through
[foxy-miner](https://github.com/foxypool/foxy-miner).

This branch is what is tagged as **v2.1.0** and what the release binaries in
[Releases](https://github.com/Makh2/signum-miner/releases) are built from; the
`CHANGELOG.md` at the root of this branch records the changes against upstream.

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

The tests pass in three configurations:

* `cargo test` (default features, AVX2 CPU),
* `cargo test --no-default-features`,
* `cargo test --no-default-features --features opencl,simd_avx2`.

The last one compiles the GPU worker too and passes **24 tests**; the first two pass
**23** each. There is no test coverage for the OpenCL paths themselves - they need a device - so changes there
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
  round);
* the release profile **unwinds** on panic instead of aborting. `panic = 'abort'`
  is upstream's setting (from `b813a5b`, "Optimize build for size") and it means
  one failed OpenCL call ends the whole miner - every round in flight with it.
  It also deletes the recovery code silently: under that strategy
  `catch_unwind` can never return `Err`, so the compiler folds the `Err` arms
  away as unreachable. That is what happened here - the built binary did not
  contain a single string those arms log, which is how it was found.

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
  instead of the miner going quiet;
* the two tasks the miner cannot lose - the one that starts rounds and the one
  that submits shares - run their work behind `catch_panic`, and fetching the
  mining info is wrapped in `catch_unwind` as well. Those are the only places
  where a single failure used to end the whole job: the HTTP path parses headers
  and answers, and the round setup touches the reader tasks and the thread pool.
  Now such a panic costs one round instead of every round from then on;
* the generation signature the pool sends is decoded as a value, not unwrapped.
  It is a string from outside, so it can be the wrong length or not hex at all,
  and it used to be a panic in the round setup - with `panic = 'abort'`, the end
  of the miner. A block with a refused signature is ignored, once per distinct
  value;
* a drive is taken for a round and only gives itself back, so a round never starts
  a second task on a drive whose previous task is still inside. That second task
  used to block on the plot file forever, and the round it belonged to could never
  close: one stuck drive was enough to stop every share until a restart. Now the
  round scans without that drive and says which one, for how long it has been
  busy, and how long since it last showed any progress;
* the reader task itself runs under `catch_unwind`. This is not cosmetic: an
  unexpected panic in a rayon task does not kill that thread, it aborts the whole
  process (`Rayon: detected unexpected panic; aborting`, reproduced on purpose).
  A panic now costs one round and one line;
* a new round lets the interrupted one leave first - up to 1.5 s - before it calls
  a drive stuck. Without that, a block arriving mid-round found all five drives
  still inside the previous round, skipped every one of them and closed the new
  round at `roundtime=0ms`: the block was scanned by nobody. Measured in the live
  log at 13:26:51, and the reason the wait exists;
* the hashing path of `ocl.rs` returns `Result<_, GpuError>` instead of unwrapping
  every driver call: `gpu_transfer`, `upload_gensig`, `transfer_buffer_to_gpu`,
  `gpu_hash`, `hash` and `get_result` all say which call failed and what the
  driver answered. A failed window is a value, not a panic, so it is retried once
  and then dropped - and a driver that keeps failing no longer turns into an
  aborted process;
* the GPU worker is rebuilt instead of given up on. A worker that has failed five
  windows in a row, or has hashed nothing for 60 s while not one buffer came back,
  is replaced by a new one - new contexts, new queues, new buffers - on a thread of
  its own, because the driver call the old worker sits in may block the rebuild
  too and that must not reach the task that starts rounds. The buffers of the old
  generation are retired by the readers instead of being filled for a worker that
  cannot use them (a buffer belongs to its context, and the driver refuses one
  from another). The pool is made one generation larger for exactly that.

`useGpuMemMapping: false` is now the recommended setting, and the reason is in
the log history: with mapping on, every reader thread calls into the driver
itself - `map`/`unmap` around each 16 MB chunk, about 950 driver calls per round
from five threads on one queue. After a round was preempted by a new block, all
five threads were found stuck inside those calls: no disk I/O at all (0 MB/s, 0
IOPS, 0% busy on all five drives, while reading the same files outside the miner
took 70-450 ms), no line in the log, and nothing but a restart brought it back.
With mapping off the reader threads make no OpenCL calls at all; the single GPU
worker does the transfer. Measured: 13 rounds out of 13 closed, 18.6-21.8 s each,
13 shares accepted, no watchdog line - against a wedge on the first or second
round with mapping on.

None of this is specific to the GPU build: the buffer timeout, the worker-gone
lines, the watchdog and `catch_panic` are in the CPU-only build too, and it was
checked there. Only the GPU worker's own lines are absent from it, because the
OpenCL code is.

Deliberately **not** here:

* the learned deadline limits (they were built, measured and removed, see below);
* anything that filters submissions on our side.

## What the log says

| line | meaning |
|---|---|
| `Starting scan of new block: height=…, scoop=… (chain)` | foxy handed the miner a round |
| `round finished: roundtime=…ms, speed=…` | the scan completed and the best share (if any) was submitted; `speed=nothing was scanned` means it closed with no drive able to start |
| `accepted: height=…, deadline=…` | the pool took the share |
| `refused: height=…, deadline=…: <pool text>` | the pool's answer; costs nothing |
| `miner: … panicked -> carrying on` | a panic was caught in a long-lived task; nothing else changed |
| `pool sent a generationSignature that is not 32 bytes of hex: "…"` | that block is ignored; the pool's string is not to be trusted |
| `reader …: no free buffer for 60s while scanning height=…` | the hashing side stopped returning buffers |
| `reader …: GPU worker N is gone` | that worker thread died; buffers are still returned |
| `GPU worker: hashing failed at height=… (set_kernel_arg(kernel2, 2) failed: …) -> trying it again (attempt 1 of 2)` | a driver call returned an error; the window gets a second try |
| `GPU worker: window at height=… failed 2 times in a row (…) -> dropping it` | that window is lost; the round keeps going |
| `GPU worker: giving up hashing after 5 failures in a row` | that worker stops hashing but still returns buffers, so rounds still close - and the miner rebuilds it |
| `GPU worker: still not hashing (N windows dropped since it gave up …)` | reminder every 50 windows while a rebuild has not happened yet |
| `GPU worker: … -> building a new one in the background (generation N)` | the old worker is being replaced; the drives keep scanning |
| `GPU worker: generation N is running; the round carries on with it` | the new worker is in place and hashing |
| `GPU worker: rebuilding failed - the device or its driver is in trouble` | the new context could not be built either; rounds still close, nothing is hashed |
| `round for height=… has not closed for 90s (N of 5 …, M buffer(s) free)` | the watchdog closed a stuck round; `0 free` points at the hashing side, `5 free` at the drives |
| `reader G:\: still busy with height=… for …s (no progress for …s) -> scanning height=… without this drive` | that drive's previous task never came back; the round goes on without it |
| `no drive could start height=…: all 5 drives are busy from earlier rounds` | every drive is stuck: a restart is the only cure, and now it says so at once |
| `reader G:\: panicked while scanning height=… -> drive gives up this round` | an unexpected panic was caught in a reader task instead of aborting the process |

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

* The hang itself is not repaired where it happens. `ocl.rs` reports failed calls
  now, and a worker that stops answering is rebuilt, but a driver call that never
  returns still costs the window it was in and the worker for as long as the
  rebuild takes. What is fixed is the silence: the drives name themselves, the
  round closes, the log says a new worker is being built, and mining resumes
  without a restart.
* A rebuild can fail the same way - building a context talks to the same driver
  that is stuck. That is why it runs on its own thread, behind a backoff and a
  "one rebuild at a time" flag: if it blocks, the miner keeps starting rounds and
  the log says no new worker appeared. The rebuild itself has not been tested
  against a real wedge, only against the failure paths.
* The startup path still unwraps: building the first context, the program and the
  buffers. A machine that cannot create them has nothing to mine with, so it exits
  with the driver's message - that is intended, but it is a plain panic.
* Mutex poisoning, and with it `utils::lock`, only exists because panics unwind -
  under `panic = 'abort'` there is nothing to recover.
* No test covers the watchdog, the preemption grace or the rebuild; the drive slot
  has two tests, because losing a drive silently is what cost the most so far.
