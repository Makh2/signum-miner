# Signum Miner — community maintenance fork

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Telegram](https://img.shields.io/badge/chat-telegram-blue.svg)](https://t.me/signumnetwork)
[![Get Support](https://img.shields.io/badge/join-discord-blue.svg)](https://discord.gg/9rH2bbCNpe)

---

> **This is an unofficial community fork.** It is based on, and includes code from,
> [`signum-network/signum-miner`](https://github.com/signum-network/signum-miner),
> but it is not maintained by the Signum network team and is not affiliated with it.
> The goal of this fork: make the miner stable enough to run unattended for weeks.

## ⚡ Why this fork?

The official miner has been in maintenance mode for a while, and running it
unattended against the mainnet pool has real, measured problems:

* the **`opencl` feature never compiled** — the async GPU worker referenced code
  that the tokio migration had already deleted, so nothing but CPU builds ever
  built;
* the process could **die completely** on a bad block, a failed driver call or a
  poisoned lock — the build used `panic = 'abort'`, which turns a recoverable
  panic into a process-wide abort that also deletes the recovery code;
* a drive or a GPU worker going silent was only noticed by a human watching the
  log, and the miner kept going quiet for minutes to hours instead of closing the
  round.

## 📦 Downloads

Prebuilt binaries for the most common cases live in the
[Releases](https://github.com/Makh2/signum-miner/releases) of this fork.

| Binary | What it does |
|---|---|
| `signum-miner-avx2-<os>-x86_64` | CPU mining with AVX2 |
| `signum-miner-sse2-<os>-x86_64` | CPU mining with the SSE2 baseline |
| `signum-miner-avx2-opencl-<os>-x86_64` | CPU + OpenCL GPU mining |

Currently built for Linux and Windows, x86-64. GPU binaries need an OpenCL runtime
(`OpenCL.dll` on Windows — shipped with NVIDIA drivers — or `libOpenCL.so.1` on Linux
from your vendor's driver/ICD).

> The fork's binaries are built with the fork's CI; nothing here was smoke-tested
> on hardware beyond the author's own rig beyond what is described in `FORK.md`.

## 🛠 What v2.1.0 changes compared to upstream 2.0.0

Everything listed below is relative to upstream release 2.0.0 (`master` @ `4a16d45`).
The full technical notes, log-line reference and the reasoning behind each fix are
in [`FORK.md`](./FORK.md).

### The miner stops dying

* panics no longer abort the process: `panic = 'unwind'`, critical tasks run behind
  `catch_unwind`;
* a bad generation signature from the pool is ignored instead of panicking the round;
* a failed hashing window is retried once and dropped instead of killing the worker;
* a poisoned lock is recovered instead of stopping the miner.

### The miner stops going silent

* a round still open after 90 s is closed by a watchdog with a one-line summary;
* interrupted rounds are given a 1.5 s preemption grace period before a drive is
  skipped;
* a worker that fails repeatedly or goes silent is rebuilt on its own thread.

### The miner keeps shares

* shares found are submitted at once again (priority retry);
* pool refusals are printed as a single short line, and a refused extreme deadline
  is remembered instead of being offered back forever.

### The GPU path

* the `opencl` feature compiles again, and the release binaries include an OpenCL
  variant (`signum-miner-avx2-opencl-*`).

### The dependencies

* `rand` 0.10 and `reqwest` 0.13, with the code adjustments they require; the full
  test suite passes in the default, no-default and OpenCL configurations, including
  the live HTTPS pool round trip.

## 🔧 Building from source

```sh
# Rust stable toolchain from https://rust-lang.org/tools/install
cargo build --release --no-default-features --features simd_avx2            # CPU only
cargo build --release --no-default-features --features opencl,simd_avx2     # CPU + OpenCL
```

Linux: for the OpenCL build install the OpenCL ICD loader
(`sudo apt install ocl-icd-opencl-dev` or your vendor's equivalent).

Windows: for the OpenCL build install the vendor OpenCL SDK / CUDA toolkit so that
`OpenCL.lib` is available.

## 🏃 Running

The miner reads `config.yaml` from its working directory (see the release for a
sample) and connects to a Signum pool or a solo mining endpoint. On Windows it is
usually started through [foxy-miner](https://github.com/foxypool/foxy-miner);
on Linux run the binary directly.

GPU configuration lives in `config.yaml` (`gpu_threads`, `gpu_platform`,
`gpu_mem_mapping`, `gpu_workers`) and is enabled with the `opencl` feature build.

## ✅ Verifying a release

Each release carries `SHA256SUMS` and a source tarball (GPL-3.0). Check the
signature/hashes before running a binary from the internet.

## 📜 License

GPL-3.0 — see [LICENSE](./LICENSE). Source of this release is the tagged `v2.1.0`
source archive.

## 🔗 Upstream

Forked from [`signum-network/signum-miner`](https://github.com/signum-network/signum-miner).
Upstream's README describes the miner itself; this fork's docs describe the fork.
