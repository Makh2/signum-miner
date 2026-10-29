use crate::miner::Buffer;
#[cfg(feature = "opencl")]
use crate::miner::CpuBuffer;
use crate::plot::{Meta, Plot};
use crate::utils::{lock, new_thread_pool};
use crossbeam_channel;
use crossbeam_channel::{Receiver, Sender};
use pbr::{ProgressBar, Units};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Stdout;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use stopwatch::Stopwatch;

/// How long a drive task waits for a free buffer before looking around again.
const BUFFER_WAIT_MS: u64 = 500;

/// How long a drive task keeps waiting for a buffer before it gives up on the
/// round. A buffer that never comes back means the hashing side is stuck, and
/// waiting for it forever used to hang the round for good.
const BUFFER_GIVE_UP_MS: u64 = 60_000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// What one drive's reader task is doing right now.
///
/// A task that never comes back from a disk read - or from a driver call it
/// makes on the way - used to be invisible: no line, no report, and the round
/// it belonged to simply never closed. Worse, the next round started another
/// task for the same drive, which blocked on the plot file the stuck task still
/// held, so one stuck drive could cost the round its completion for good. The
/// slot makes a stuck drive nameable, and lets a round go ahead without it.
pub struct DriveSlot {
    busy: AtomicBool,
    started_ms: AtomicU64,
    progress_ms: AtomicU64,
    height: AtomicU64,
}

impl DriveSlot {
    fn new() -> Self {
        Self {
            busy: AtomicBool::new(false),
            started_ms: AtomicU64::new(0),
            progress_ms: AtomicU64::new(0),
            height: AtomicU64::new(0),
        }
    }

    /// Take the drive for a round. False means its previous task is still in
    /// there, so this round must go on without it.
    fn try_start(&self, height: u64) -> bool {
        if self
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let now = now_ms();
            self.started_ms.store(now, Ordering::SeqCst);
            self.progress_ms.store(now, Ordering::SeqCst);
            self.height.store(height, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// A chunk was read or handed over: the drive is alive.
    fn progress(&self) {
        self.progress_ms.store(now_ms(), Ordering::SeqCst);
    }

    /// The task has left, whatever the reason. Only the task that took the slot
    /// releases it, so a stale task cannot free a drive that a newer one holds.
    fn release(&self) {
        self.busy.store(false, Ordering::SeqCst);
    }

    /// How long the previous task has been in this drive, and how long ago it
    /// last showed any progress.
    fn stuck_for(&self) -> (u64, u64, u64) {
        let now = now_ms();
        (
            now.saturating_sub(self.started_ms.load(Ordering::SeqCst)) / 1000,
            now.saturating_sub(self.progress_ms.load(Ordering::SeqCst)) / 1000,
            self.height.load(Ordering::SeqCst),
        )
    }
}

pub struct BufferInfo {
    pub len: usize,
    pub height: u64,
    pub block: u64,
    pub base_target: u64,
    pub gensig: Arc<[u8; 32]>,
    pub start_nonce: u64,
    pub finished: bool,
    pub account_id: u64,
    pub gpu_signal: u64,
}
pub struct ReadReply {
    pub buffer: Box<dyn Buffer + Send>,
    pub info: BufferInfo,
}

#[allow(dead_code)]
pub struct Reader {
    drive_id_to_plots: HashMap<String, Arc<Vec<Mutex<Plot>>>>,
    pub total_size: u64,
    pool: rayon::ThreadPool,
    rx_empty_buffers: Receiver<Box<dyn Buffer + Send>>,
    tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
    tx_read_replies_cpu: Sender<ReadReply>,
    tx_read_replies_gpu: Option<Vec<Sender<ReadReply>>>,
    interupts: Vec<Sender<()>>,
    /// One slot per drive, so a task that never returns can be named and the
    /// round can be scanned without that drive instead of never closing.
    drive_slots: Arc<HashMap<String, Arc<DriveSlot>>>,
    show_progress: bool,
    show_drive_stats: bool,
}

impl Reader {
    pub fn new(
        drive_id_to_plots: HashMap<String, Arc<Vec<Mutex<Plot>>>>,
        total_size: u64,
        num_threads: usize,
        rx_empty_buffers: Receiver<Box<dyn Buffer + Send>>,
        tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
        tx_read_replies_cpu: Sender<ReadReply>,
        tx_read_replies_gpu: Option<Vec<Sender<ReadReply>>>,
        show_progress: bool,
        show_drive_stats: bool,
        thread_pinning: bool,
        benchmark: bool,
    ) -> Reader {
        if !benchmark {
            check_overlap(&drive_id_to_plots);
        }

        let drive_slots: Arc<HashMap<String, Arc<DriveSlot>>> = Arc::new(
            drive_id_to_plots
                .keys()
                .map(|drive| (drive.clone(), Arc::new(DriveSlot::new())))
                .collect(),
        );

        Reader {
            drive_id_to_plots,
            total_size,
            pool: new_thread_pool(num_threads, thread_pinning),
            rx_empty_buffers,
            tx_empty_buffers,
            tx_read_replies_cpu,
            tx_read_replies_gpu,
            interupts: Vec::new(),
            drive_slots,
            show_progress,
            show_drive_stats,
        }
    }

    /// Starts one reader task per drive and returns how many drives could not be
    /// started because their previous task never came back. A round with a stuck
    /// drive has to be able to close anyway - the caller counts the skipped
    /// drives as reported - otherwise one bad read stops every share until
    /// someone restarts the miner.
    pub fn start_reading(
        &mut self,
        height: u64,
        block: u64,
        base_target: u64,
        scoop: u32,
        gensig: &Arc<[u8; 32]>,
    ) -> usize {
        for interupt in &self.interupts {
            interupt.send(()).ok();
        }
        let mut pb = ProgressBar::new(self.total_size);
        pb.format("│██░│");
        pb.set_width(Some(80));
        pb.set_units(Units::Bytes);
        pb.message("Searching your hashes: ");
        let pb = Arc::new(Mutex::new(pb));

        // send start signals (dummy buffer) to gpu threads. A worker that is
        // gone must not panic this caller: it runs on the round's critical path,
        // holding the round state and the reader, and a panic here would poison
        // both locks and stop the miner.
        #[cfg(feature = "opencl")]
        for i in 0..self.tx_read_replies_gpu.as_ref().unwrap().len() {
            let sent = self.tx_read_replies_gpu.as_ref().unwrap()[i].send(ReadReply {
                buffer: Box::new(CpuBuffer::new(0)) as Box<dyn Buffer + Send>,
                info: BufferInfo {
                    len: 1,
                    height,
                    block,
                    base_target,
                    gensig: gensig.clone(),
                    start_nonce: 0,
                    finished: false,
                    account_id: 0,
                    gpu_signal: 1,
                },
            });
            if sent.is_err() {
                error!(
                    "reader: GPU worker {} is gone, the round for height={} will get no \
                     hashes from it",
                    i + 1,
                    height
                );
            }
        }

        let mut interupts = Vec::new();
        let mut skipped = 0usize;
        for (drive, plots) in self.drive_id_to_plots.iter() {
            let slot = match self.drive_slots.get(drive) {
                Some(slot) => slot.clone(),
                None => continue,
            };
            if !slot.try_start(height) {
                let (busy_s, silent_s, stuck_height) = slot.stuck_for();
                error!(
                    "reader {}: still busy with height={} for {}s (no progress for {}s) -> \
                     scanning height={} without this drive, the round will close without it",
                    drive, stuck_height, busy_s, silent_s, height
                );
                skipped += 1;
                continue;
            }
            let (interupt, task) = if self.show_progress {
                self.create_read_task(
                    Some(pb.clone()),
                    drive.clone(),
                    plots.clone(),
                    height,
                    block,
                    base_target,
                    scoop,
                    gensig.clone(),
                    self.show_drive_stats,
                    slot,
                )
            } else {
                self.create_read_task(
                    None,
                    drive.clone(),
                    plots.clone(),
                    height,
                    block,
                    base_target,
                    scoop,
                    gensig.clone(),
                    self.show_drive_stats,
                    slot,
                )
            };

            self.pool.spawn(task);
            interupts.push(interupt);
        }
        self.interupts = interupts;
        skipped
    }

    /// How many buffers are waiting to be filled right now. Zero means the
    /// hashing side holds them all, which is the one state in which no drive can
    /// read anything - worth saying out loud in the round watchdog.
    pub fn free_buffers(&self) -> usize {
        self.rx_empty_buffers.len()
    }

    pub fn wakeup(&mut self) {
        for plots in self.drive_id_to_plots.values() {
            let plots = plots.clone();
            self.pool.spawn(move || {
                let mut p = lock(&plots[0]);

                if let Err(e) = p.seek_random() {
                    error!(
                        "wakeup: error during wakeup {}: {} -> skip one round",
                        p.meta.name, e
                    );
                }
            });
        }
    }

    fn create_read_task(
        &self,
        pb: Option<Arc<Mutex<pbr::ProgressBar<Stdout>>>>,
        drive: String,
        plots: Arc<Vec<Mutex<Plot>>>,
        height: u64,
        block: u64,
        base_target: u64,
        scoop: u32,
        gensig: Arc<[u8; 32]>,
        show_drive_stats: bool,
        slot: Arc<DriveSlot>,
    ) -> (Sender<()>, impl FnOnce()) {
        let (tx_interupt, rx_interupt) = crossbeam_channel::unbounded();
        let rx_empty_buffers = self.rx_empty_buffers.clone();
        let tx_empty_buffers = self.tx_empty_buffers.clone();
        let tx_read_replies_cpu = self.tx_read_replies_cpu.clone();
        #[cfg(feature = "opencl")]
        let tx_read_replies_gpu = self.tx_read_replies_gpu.clone();

        // The task itself runs under `catch_unwind`: an unexpected panic in a
        // rayon task does not kill this thread only, it aborts the whole process
        // ("Rayon: detected unexpected panic; aborting"), and the drive's slot
        // would stay taken forever. The whole task costs one round if it panics,
        // and the drive is given back so the next round can use it.
        let release_slot = slot.clone();
        let drive_name = drive.clone();
        let body = move || {
            let mut sw = Stopwatch::new();
            let mut elapsed = 0i64;
            let mut nonces_processed = 0u64;
            let mut waited_ms = 0u64;
            let plot_count = plots.len();
            'outer: for (i_p, p) in plots.iter().enumerate() {
                let mut p = lock(p);
                if let Err(e) = p.prepare(scoop) {
                    error!(
                        "reader: error preparing {} for reading: {} -> skip one round",
                        p.meta.name, e
                    );
                    continue 'outer;
                }

                // The buffers come back when the hashing side is done with them.
                // Waiting for one forever - because a hashing worker is stuck or
                // gone - used to block this task and its pool thread for good,
                // and then the round could never close: the miner went silent
                // until someone restarted it. So wait with a timeout, notice a
                // replaced round while waiting, and give up on this drive after
                // a minute, saying why.
                'inner: loop {
                    let mut buffer = match rx_empty_buffers.recv_timeout(Duration::from_millis(BUFFER_WAIT_MS)) {
                        Ok(buffer) => buffer,
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break 'outer,
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if rx_interupt.try_recv().is_ok() {
                                break 'outer;
                            }
                            waited_ms += BUFFER_WAIT_MS;
                            if waited_ms >= BUFFER_GIVE_UP_MS {
                                error!(
                                    "reader {}: no free buffer for {}s while scanning height={}, \
                                     scoop={} -> giving up on this drive for this round",
                                    drive,
                                    waited_ms / 1000,
                                    height,
                                    scoop
                                );
                                break 'outer;
                            }
                            continue;
                        }
                    };
                    waited_ms = 0;
                    slot.progress();
                    if show_drive_stats {
                        sw.restart();
                    }
                    let mut_bs = buffer.get_buffer_for_writing();
                    let mut bs = lock(&mut_bs);
                    let (bytes_read, start_nonce, next_plot) = match p.read(&mut bs, scoop) {
                        Ok(x) => x,
                        Err(e) => {
                            error!(
                                "reader: error reading chunk from {}: {} -> skip one round",
                                p.meta.name, e
                            );
                            buffer.unmap();
                            (0, 0, true)
                        }
                    };
                    slot.progress();

                    if rx_interupt.try_recv().is_ok() {
                        buffer.unmap();
                        let _ = tx_empty_buffers.send(buffer);
                        break 'outer;
                    }

                    let finished = i_p == (plot_count - 1) && next_plot;
                    // buffer routing. A send to a worker that died (a panicked
                    // OpenCL call, say) used to panic this task too, which cost
                    // the round its completion and blocked a pool thread; now the
                    // buffer comes back to the pool and the drive says so.
                    #[cfg(feature = "opencl")]
                    match buffer.get_id() {
                        0 => {
                            let reply = ReadReply {
                                buffer,
                                info: BufferInfo {
                                    len: bytes_read,
                                    height,
                                    block,
                                    base_target,
                                    gensig: gensig.clone(),
                                    start_nonce,
                                    finished,
                                    account_id: p.meta.account_id,
                                    gpu_signal: 0,
                                },
                            };
                            if let Err(err) = tx_read_replies_cpu.send(reply) {
                                error!(
                                    "reader {}: CPU worker is gone (height={}, scoop={}) -> \
                                     returning the buffer",
                                    drive, height, scoop
                                );
                                let _ = tx_empty_buffers.send(err.0.buffer);
                                break 'outer;
                            }
                        }
                        i => {
                            let reply = ReadReply {
                                buffer,
                                info: BufferInfo {
                                    len: bytes_read,
                                    height,
                                    block,
                                    base_target,
                                    gensig: gensig.clone(),
                                    start_nonce,
                                    finished,
                                    account_id: p.meta.account_id,
                                    gpu_signal: 0,
                                },
                            };
                            if let Err(err) = tx_read_replies_gpu.as_ref().unwrap()[i - 1].send(reply) {
                                error!(
                                    "reader {}: GPU worker {} is gone (height={}, scoop={}) -> \
                                     returning the buffer",
                                    drive, i, height, scoop
                                );
                                let _ = tx_empty_buffers.send(err.0.buffer);
                                break 'outer;
                            }
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        let reply = ReadReply {
                            buffer,
                            info: BufferInfo {
                                len: bytes_read,
                                height,
                                block,
                                base_target,
                                gensig: gensig.clone(),
                                start_nonce,
                                finished,
                                account_id: p.meta.account_id,
                                gpu_signal: 0,
                            },
                        };
                        if let Err(err) = tx_read_replies_cpu.send(reply) {
                            error!(
                                "reader {}: CPU worker is gone (height={}, scoop={}) -> \
                                 returning the buffer",
                                drive, height, scoop
                            );
                            let _ = tx_empty_buffers.send(err.0.buffer);
                            break 'outer;
                        }
                    }

                    slot.progress();
                    nonces_processed += bytes_read as u64 / 64;

                    match &pb {
                        Some(pb) => {
                            let mut pb = lock(pb);
                            pb.add(bytes_read as u64);
                        }
                        None => (),
                    }

                    if show_drive_stats {
                        elapsed += sw.elapsed_ms();
                    }

                    // send termination signal (dummy buffer) to gpu
                    if finished {
                        #[cfg(feature = "opencl")]
                        for i in 0..tx_read_replies_gpu.as_ref().unwrap().len() {
                            // A signal for a worker that is gone is worth no more
                            // than its buffer - never worth a panic here.
                            let _ = tx_read_replies_gpu.as_ref().unwrap()[i].send(ReadReply {
                                buffer: Box::new(CpuBuffer::new(0)) as Box<dyn Buffer + Send>,
                                info: BufferInfo {
                                    len: 1,
                                    height,
                                    block,
                                    base_target,
                                    gensig: gensig.clone(),
                                    start_nonce: 0,
                                    finished: false,
                                    account_id: 0,
                                    gpu_signal: 2,
                                },
                            });
                        }
                    }

                    if finished && show_drive_stats {
                        info!(
                            "{: <80}",
                            format!(
                                "drive {} finished, speed={} MiB/s",
                                drive,
                                nonces_processed * 1000 / (elapsed + 1) as u64 * 64 / 1024 / 1024,
                            )
                        );
                    }

                    if next_plot {
                        break 'inner;
                    }
                }
            }
        };

        (tx_interupt, move || {
            let caught = panic::catch_unwind(AssertUnwindSafe(body));
            release_slot.release();
            if caught.is_err() {
                error!(
                    "reader {}: panicked while scanning height={}, scoop={} -> drive gives up \
                     this round, the miner carries on",
                    drive_name, height, scoop
                );
            }
        })
    }
}

// Don't waste your time striving for perfection; instead, strive for excellence - doing your best.
// let my_best = perfection;
pub fn check_overlap(drive_id_to_plots: &HashMap<String, Arc<Vec<Mutex<Plot>>>>) -> bool {
    let plots: Vec<Meta> = drive_id_to_plots
        .values()
        .map(|a| a.iter())
        .flatten()
        .map(|plot| plot.lock().unwrap().meta.clone())
        .collect();
    plots
        .par_iter()
        .enumerate()
        .filter(|(i, plot_a)| {
            plots[i + 1..]
                .par_iter()
                .filter(|plot_b| {
                    plot_a.account_id == plot_b.account_id && plot_b.overlaps_with(&plot_a)
                })
                .count()
                > 0
        })
        .count()
        > 0
}

#[cfg(test)]
mod tests {
    use super::DriveSlot;

    /// A drive is taken for a round, and the next round must not start a second
    /// task on it: that task would block on the plot file the first one holds,
    /// and the round could never close.
    #[test]
    fn a_busy_drive_is_not_started_twice() {
        let slot = DriveSlot::new();
        assert!(slot.try_start(1), "first round takes the drive");
        assert!(!slot.try_start(2), "a second round must not");
        slot.release();
        assert!(slot.try_start(3), "and takes it again once the task left");
    }

    /// A stuck task is worth a line that says how long it has been silent.
    #[test]
    fn a_stuck_drive_reports_its_silence() {
        let slot = DriveSlot::new();
        assert!(slot.try_start(42));
        let (busy_s, silent_s, height) = slot.stuck_for();
        assert_eq!(height, 42);
        assert!(busy_s < 5);
        assert!(silent_s < 5);
        slot.progress();
        let (_, silent_after, _) = slot.stuck_for();
        assert!(silent_after < 5, "progress resets the silence");
    }
}
