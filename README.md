# Signum Miner

[![Build Status](https://github.com/signum-network/signum-miner/actions/workflows/release.yml/badge.svg)](https://github.com/signum-network/signum-miner/actions)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Telegram](https://img.shields.io/badge/chat-telegram-blue.svg)](https://t.me/signumnetwork)
[![Get Support](https://img.shields.io/badge/join-discord-blue.svg)](https://discord.gg/9rH2bbCNpe) 
</br>
[![Website](https://img.shields.io/badge/Website-signum.network-green?logo=Firefox&logoColor=white)](https://signum.network)
[![Docs](https://img.shields.io/badge/Docs-Mining%20Guide-blue?logo=Book&logoColor=white)](https://docs.signum.network/signum/starting-mining-signa)

---

## ⚡ **Why Signum Miner?**

Signum is the world's first truly sustainable blockchain — and the Signum Miner is your gateway to mining it efficiently and eco-consciously. Whether you're using modern AVX512-capable CPUs, an ARM board, or legacy hardware, the Signum Miner is optimized to deliver fast results with minimal energy.

> Join the green revolution of decentralized computing — mine with purpose. Mine with Signum.

---

## 🔧 **Features**

- Ultra-low energy-power mining algorithm (**Proof of Commitment (PoC+)**)
- AVX512, AVX2, AVX, SSE2 and NEON **SIMD optimizations**
- Multi-threaded & high performance I/O
- Integrated plot reader, CPU miner & buffer pipeline
- Full async + **Tokio** + **crossbeam** architecture

## 🧰 **Feature Overview**

| Architecture     | Feature(s)               | Description                            |
|------------------|--------------------------|----------------------------------------|
| x86 (Intel/AMD)  | sse2, avx, avx2, avx512f | SIMD extensions for Desktop/Server CPUs |
| ARM (e.g. Pi)    | neon                    | SIMD for ARM CPUs                      |




| Variant  | Description                                              | CPU Availability                            |
|----------|----------------------------------------------------------|---------------------------------------------|
| sse2     | Older, widely supported SIMD extension                   | Almost all x86 CPUs since 2001              |
| avx      | Advanced Vector Extensions – larger 256-bit registers    | Intel: Sandy Bridge (2011), AMD: Bulldozer  |
| avx2     | Improved AVX version with integer support                | Intel: Haswell (2013), AMD: Excavator       |
| avx512f  | Even wider 512-bit SIMD registers – very powerful        | Intel: Skylake-X (rare in consumer CPUs)    |
| neon     | SIMD extension for ARM architecture                      | ARMv7 (32-bit) and ARMv8 (64-bit, e.g. Raspberry Pi 4) |

---



## 📦 **Installation**

### 🖥️ Precompiled Binaries

➡️ [Go to Releases](https://github.com/signum-network/signum-miner/releases)

Download the binary matching your system:

For Linux distributions:
- `signum-miner-avx` 
- `signum-miner-avx2`
- `signum-miner-avx512f` (modern CPUs)
- `signum-miner-sse2` (legacy CPUs)
- ` signum-miner-aarch64-neon` (ARM)

For Windows distributions one of the `.exe` versions:
- `signum-miner-avx` 
- `signum-miner-avx2`
- `signum-miner-avx512f` (modern CPUs)
- `signum-miner-sse2` (legacy CPUs)

## ⚙️ **Running the binaries**

### Config

The miner needs a **config.yaml** file.</br>
Please download from the corresponding release.

### Running
Be sure to have the config file on the same folder of your binary.</br>

For windows, double click on the executable file.</br>
If it refuses to run, start the executable from a command prompt to check for error messages.</br>
For Linux run it with the folliwing command:</br>
```shell
./signum-miner
```

---

## 💻 Build from Source
 - First you need to install a Rust stable toolchain, check https://www.rust-lang.org/tools/install.
 - Binaries are in **target/debug** or **target/release** depending on optimization.

```bash
# Install Rust
curl https://sh.rustup.rs -sSf | sh

# Clone the repository
git clone https://github.com/signum-network/signum-miner
cd signum-miner

# decide on features to run/build:
simd: support for SSE2, AVX, AVX2 and AVX512F (x86_cpu)
neon: support for Arm NEON (arm_cpu)

# Build with desired features (choose one!)
cargo build --release --no-default-features --features simd_avx

# Default Build with avx2 features 
cargo build --release 
```

## 📜 License

This project is licensed under the GPLv3 — see the LICENSE file for details.</br>
Made with ❤️ for the future of decentralized sustainability — Signum.

### Forked from
This is a code fork from https://github.com/PoC-Consortium/scavenger