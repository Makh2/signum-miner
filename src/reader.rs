use crate::miner::Buffer;
#[cfg(feature = "opencl")]
use crate::miner::CpuBuffer;
use crate::miner::NonceData;
use crate::plot::{Meta, Plot};
use crate::utils::{lock, new_thread_pool};
use crossbeam_channel;
use crossbeam_channel::{Receiver, Sender};
use pbr::{ProgressBar, Units};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io;
use std::io::Stdout;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use stopwatch::Stopwatch;
use tokio::sync::mpsc;

/// How long a drive task waits for a free buffer before looking around again.
const BUFFER_WAIT_MS: u64 = 500;

/// How long a drive task keeps waiting for a buffer before it gives up on the
/// round. A buffer that never comes back means the hashing side is stuck, and
/// waiting for it forever used to hang the round for good.
const BUFFER_GIVE_UP_MS: u64 = 60_000;

/// How long a new round waits for the previous round's tasks to leave before it
/// judges their drives stuck.
///
/// A new block interrupts the round that is running, and its tasks leave on the
/// next chunk they read - a fraction of a second, but not instantly. Starting
/// the new round at once found every drive still busy, skipped the whole round
/// and closed it at zero milliseconds: the block was then scanned by nobody.
const PREEMPT_GRACE_MS: u64 = 1_500;

/// How long a GPU worker may stay silent before the miner builds a new one.
/// A window takes a second or two, so a minute of silence is not slowness.
#[cfg(feature = "opencl")]
pub const GPU_SILENT_MS: u64 = 60_000;

/// How many windows in a row may fail before the worker is replaced instead of
/// waiting for it to give up on its own.
#[cfg(feature = "opencl")]
pub const GPU_FAILURES_IN_A_ROW: u64 = 5;

/// How long to wait after a rebuild before trying another one, so a device that
/// is failing hard does not get a new context every second.
#[cfg(feature = "opencl")]
pub const GPU_REBUILD_BACKOFF_MS: u64 = 20_000;

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

    fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
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

/// The GPU worker that is running now.
///
/// Reads take the current sender from here for every chunk, so a worker that had
/// to be replaced - because the device wedged, or kept failing - is used from the
/// next chunk on without touching the reader tasks. The counters are what the
/// stall check in the miner looks at.
#[allow(dead_code)]
pub struct GpuWorkers {
    txs: Mutex<Vec<Sender<ReadReply>>>,
    /// When the running worker last finished (or failed) a window.
    last_window_ms: AtomicU64,
    windows: AtomicU64,
    failures_in_a_row: AtomicU64,
    generation: AtomicU64,
    last_rebuild_ms: AtomicU64,
}

#[allow(dead_code)]
impl GpuWorkers {
    pub fn new(txs: Vec<Sender<ReadReply>>) -> Arc<Self> {
        Arc::new(Self {
            txs: Mutex::new(txs),
            last_window_ms: AtomicU64::new(0),
            windows: AtomicU64::new(0),
            failures_in_a_row: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            last_rebuild_ms: AtomicU64::new(0),
        })
    }

    /// Hands a chunk to the worker that is running now. On `Err` the caller gets
    /// its reply - and the buffer in it - back.
    pub fn send(&self, index: usize, reply: ReadReply) -> Result<(), ReadReply> {
        let tx = match lock(&self.txs).get(index) {
            Some(tx) => tx.clone(),
            None => return Err(reply),
        };
        tx.send(reply).map_err(|err| err.0)
    }

    pub fn count(&self) -> usize {
        lock(&self.txs).len()
    }

    /// A window came back: hashed, or failed. Both are signs of a worker that is
    /// still alive; only silence is not.
    pub fn note_window(&self, hashed: bool) {
        self.last_window_ms.store(now_ms(), Ordering::SeqCst);
        if hashed {
            self.windows.fetch_add(1, Ordering::SeqCst);
            self.failures_in_a_row.store(0, Ordering::SeqCst);
        } else {
            self.failures_in_a_row.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub fn failures_in_a_row(&self) -> u64 {
        self.failures_in_a_row.load(Ordering::SeqCst)
    }

    /// How long the running worker has been silent. Zero means it has not been
    /// used yet in this process.
    pub fn silent_for_ms(&self) -> u64 {
        let last = self.last_window_ms.load(Ordering::SeqCst);
        if last == 0 {
            return 0;
        }
        now_ms().saturating_sub(last)
    }

    pub fn windows(&self) -> u64 {
        self.windows.load(Ordering::SeqCst)
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    pub fn last_rebuild_ms(&self) -> u64 {
        self.last_rebuild_ms.load(Ordering::SeqCst)
    }

    /// How long since the running generation was installed; a rebuild is not
    /// worth trying before the backoff has passed.
    pub fn since_rebuild_ms(&self) -> u64 {
        let last = self.last_rebuild_ms.load(Ordering::SeqCst);
        if last == 0 {
            return u64::MAX;
        }
        now_ms().saturating_sub(last)
    }

    /// Puts a freshly created set of workers in place. Reads pick the new
    /// senders up on their next chunk; the old ones are abandoned with their
    /// receiver, which is the only thing a wedged driver call leaves behind.
    pub fn install(&self, txs: Vec<Sender<ReadReply>>) {
        *lock(&self.txs) = txs;
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.last_rebuild_ms.store(now_ms(), Ordering::SeqCst);
        self.last_window_ms.store(0, Ordering::SeqCst);
        self.failures_in_a_row.store(0, Ordering::SeqCst);
    }
}

/// Hands a buffer back to the pool without ever blocking the caller.
///
/// The pool is bounded, and after a GPU worker was rebuilt there can be more
/// buffers in circulation than the pool was made for: the abandoned worker may
/// still be holding some, and the new one got its own. A blocking send would
/// then wedge the reader task - and its pool thread - on a full channel, which
/// is exactly the kind of silence this fork is trying to remove. A full pool
/// means the buffers it dropped are surplus, so dropping them costs nothing.
fn return_buffer(tx_empty_buffers: &Sender<Box<dyn Buffer + Send>>, buffer: Box<dyn Buffer + Send>) {
    match tx_empty_buffers.try_send(buffer) {
        Ok(()) => (),
        Err(crossbeam_channel::TrySendError::Full(_)) => (),
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => (),
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
    tx_read_replies_gpu: Option<Arc<GpuWorkers>>,
    interupts: Vec<Sender<()>>,
    /// One slot per drive, so a task that never returns can be named and the
    /// round can be scanned without that drive instead of never closing.
    drive_slots: Arc<HashMap<String, Arc<DriveSlot>>>,
    show_progress: bool,
    show_drive_stats: bool,
    /// How long a single chunk read may take before the drive is given up on
    /// for the round.
    read_timeout: Duration,
    /// Lets a drive that gave up on a hung read tell the round it is done, so
    /// the round closes when the other drives finish instead of waiting for the
    /// round watchdog.
    tx_nonce_data: mpsc::Sender<NonceData>,
    /// Plots whose reads hung during a round, so later rounds skip them
    /// instead of losing a buffer on them again.
    bad_plots: Arc<Mutex<HashSet<String>>>,
}

impl Reader {
    pub fn new(
        drive_id_to_plots: HashMap<String, Arc<Vec<Mutex<Plot>>>>,
        total_size: u64,
        num_threads: usize,
        rx_empty_buffers: Receiver<Box<dyn Buffer + Send>>,
        tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
        tx_read_replies_cpu: Sender<ReadReply>,
        tx_read_replies_gpu: Option<Arc<GpuWorkers>>,
        show_progress: bool,
        show_drive_stats: bool,
        thread_pinning: bool,
        benchmark: bool,
        read_timeout: Duration,
        tx_nonce_data: mpsc::Sender<NonceData>,
        bad_plots: Arc<Mutex<HashSet<String>>>,
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
            read_timeout,
            tx_nonce_data,
            bad_plots,
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

        // Let the interrupted round's tasks leave before deciding a drive is
        // stuck. They check for the interrupt after the chunk they are reading,
        // so this usually takes tens of milliseconds; a drive that is still busy
        // after the grace is the one worth naming and skipping.
        let grace_until = now_ms() + PREEMPT_GRACE_MS;
        while now_ms() < grace_until
            && self.drive_slots.values().any(|slot| slot.is_busy())
        {
            std::thread::sleep(Duration::from_millis(10));
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
        match self.tx_read_replies_gpu.as_ref() {
            Some(workers) => {
                for i in 0..workers.count() {
                    let sent = workers.send(
                        i,
                        ReadReply {
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
                        },
                    );
                    if sent.is_err() {
                        error!(
                            "reader: GPU worker {} is gone, the round for height={} will get no \
                             hashes from it",
                            i + 1,
                            height
                        );
                    }
                }
            }
            None => (),
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
                    self.read_timeout,
                    self.tx_nonce_data.clone(),
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
                    self.read_timeout,
                    self.tx_nonce_data.clone(),
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
        read_timeout: Duration,
        tx_nonce_data: mpsc::Sender<NonceData>,
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
        let bad_plots = self.bad_plots.clone();
        let body = move || {
            let mut sw = Stopwatch::new();
            let mut elapsed = 0i64;
            let mut nonces_processed = 0u64;
            let mut waited_ms = 0u64;
            let plot_count = plots.len();
            'outer: for (i_p, p) in plots.iter().enumerate() {
                let mut p = lock(p);
                if bad_plots.lock().unwrap().contains(&p.meta.name) {
                    // This plot already hung once; drop it and keep the drive
                    // scanning its remaining plots instead of dropping the drive.
                    continue 'outer;
                }
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

                    // A buffer made for a GPU worker that has since been
                    // replaced belongs to a device context the new worker does
                    // not use: every window filled with it would be refused by
                    // the driver, and it would keep failing forever. Retire it
                    // and wait for the next buffer instead.
                    #[cfg(feature = "opencl")]
                    if let Some(workers) = tx_read_replies_gpu.as_ref() {
                        let generation = buffer.generation();
                        if generation != 0 && generation != workers.generation() {
                            debug!(
                                "reader {}: retiring a buffer from GPU worker generation {} \
                                 (now {})",
                                drive,
                                generation,
                                workers.generation()
                            );
                            drop(buffer);
                            continue 'inner;
                        }
                    }

                    waited_ms = 0;
                    slot.progress();
                    if show_drive_stats {
                        sw.restart();
                    }
                    let mut_bs = buffer.get_buffer_for_writing();
                    let (bytes_read, start_nonce, next_plot) =
                        match p.read_timed(mut_bs.clone(), scoop, read_timeout) {
                            Ok(x) => x,
                            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                                error!(
                                    "reader {}: read on {} did not return within {}s -> \
                                     skipping its plot for this round",
                                    drive,
                                    p.meta.name,
                                    read_timeout.as_secs()
                                );
                                // The read thread is stuck in the driver and holds the
                                // buffer; it is lost with it. Skip this plot - the
                                // corrupted nonce range is dropped - and keep the drive
                                // on the rest of its plots.
                                bad_plots.lock().unwrap().insert(p.meta.name.clone());
                                continue 'outer;
                            }
                            Err(e) => {
                                error!(
                                    "reader: error reading chunk from {}: {} -> skipping this plot for the round",
                                    p.meta.name, e
                                );
                                buffer.unmap();
                                // Behavioral skip: a bad chunk means this plot's tail for
                                // this round cannot be trusted, so skip to the next plot
                                // instead of letting the round stall into the watchdog.
                                continue 'outer;
                            }
                        };
                    slot.progress();

                    if rx_interupt.try_recv().is_ok() {
                        buffer.unmap();
                        return_buffer(&tx_empty_buffers, buffer);
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
                                return_buffer(&tx_empty_buffers, err.0.buffer);
                                break 'outer;
                            }
                        }
                        i => {
                            // A chunk filled into a buffer from a replaced
                            // worker belongs to a context the new worker does
                            // not use, and the driver would refuse it: drop it
                            // rather than spend a failed window on it. The last
                            // chunk of a drive is sent anyway, so its "drive
                            // finished" report is not lost with it.
                            let stale = match tx_read_replies_gpu.as_ref() {
                                Some(workers) => buffer.generation() != workers.generation(),
                                None => false,
                            };
                            if stale && !finished {
                                debug!(
                                    "reader {}: dropping a chunk filled for GPU worker \
                                     generation {} (now {})",
                                    drive,
                                    buffer.generation(),
                                    tx_read_replies_gpu
                                        .as_ref()
                                        .map(|workers| workers.generation())
                                        .unwrap_or(0)
                                );
                                drop(buffer);
                                continue 'inner;
                            }
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
                            let sent = match tx_read_replies_gpu.as_ref() {
                                Some(workers) => workers.send(i - 1, reply),
                                None => Err(reply),
                            };
                            if let Err(err) = sent {
                                error!(
                                    "reader {}: GPU worker {} is gone (height={}, scoop={}) -> \
                                     returning the buffer",
                                    drive, i, height, scoop
                                );
                                return_buffer(&tx_empty_buffers, err.buffer);
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
                            return_buffer(&tx_empty_buffers, err.0.buffer);
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
                        if let Some(workers) = tx_read_replies_gpu.as_ref() {
                            for i in 0..workers.count() {
                                // A signal for a worker that is gone is worth no
                                // more than its buffer - never worth a panic here.
                                let _ = workers.send(
                                    i,
                                    ReadReply {
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
                                    },
                                );
                            }
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
    use super::GpuWorkers;

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

    #[test]
    fn a_taken_drive_says_it_is_busy() {
        let slot = DriveSlot::new();
        assert!(!slot.is_busy());
        assert!(slot.try_start(1));
        assert!(slot.is_busy());
        slot.release();
        assert!(!slot.is_busy());
    }

    /// A replaced worker starts with a new generation and clean counters: the
    /// reader uses the generation to retire the old worker's buffers, and the
    /// miner uses the counters to decide whether this one is alive.
    #[test]
    fn a_rebuilt_worker_starts_a_new_generation() {
        let workers = GpuWorkers::new(vec![crossbeam_channel::unbounded().0]);
        assert_eq!(workers.generation(), 0);
        workers.install(vec![crossbeam_channel::unbounded().0]);
        assert_eq!(workers.generation(), 1);
        assert_eq!(workers.count(), 1);
        workers.note_window(false);
        assert_eq!(workers.failures_in_a_row(), 1);
        workers.install(vec![crossbeam_channel::unbounded().0]);
        assert_eq!(workers.generation(), 2);
        assert_eq!(
            workers.failures_in_a_row(),
            0,
            "a new device starts with no failures"
        );
    }

    /// When a higher-weight block preempts a round, the interrupted drive tasks
    /// are given up to 1.5 seconds to leave before the new round marks them as
    /// stuck and scans without them.  A drive that finishes within the grace
    /// period is not named or skipped.  The constant is the only thing a unit
    /// test can verify without a real 1.5-second wait.
    #[test]
    fn preemption_grace_period_is_1500_ms() {
        assert_eq!(
            super::PREEMPT_GRACE_MS, 1_500,
            "1.5 s grace lets most interrupted tasks finish; a slower one is then skipped"
        );
    }
}
