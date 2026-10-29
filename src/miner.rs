use crate::com::api::MiningInfoResponse as MiningInfo;
use crate::config::Cfg;
use crate::cpu_worker::create_cpu_worker_task;
use crate::future::interval::Interval;
#[cfg(feature = "opencl")]
use crate::gpu_worker::create_gpu_worker_task;
#[cfg(feature = "opencl")]
use crate::ocl::GpuBuffer;
#[cfg(feature = "opencl")]
use crate::ocl::GpuContext;
use crate::plot::{Plot, SCOOP_SIZE};
use crate::poc_hashing;
use crate::reader::Reader;
#[cfg(feature = "opencl")]
use crate::reader::GpuWorkers;
#[cfg(feature = "opencl")]
use crate::reader::{GPU_FAILURES_IN_A_ROW, GPU_REBUILD_BACKOFF_MS, GPU_SILENT_MS};
use crate::requests::RequestHandler;
use crate::utils::{catch_panic, get_device_id, lock, new_thread_pool};
use crossbeam_channel;
use crossbeam_channel::Sender;
use filetime::FileTime;
use futures_util::{stream::StreamExt, FutureExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
#[cfg(feature = "opencl")]
use ocl_core::Mem;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs::read_dir;
use std::ops::{Deref, DerefMut};
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
#[cfg(feature = "opencl")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::u64;
use stopwatch::Stopwatch;
use tokio::runtime::Handle;
use page_size;


/// How long a round may stay open before the watchdog closes it. A round takes
/// about twenty seconds, so this only fires when something is really stuck.
const ROUND_STALL_MS: i64 = 90_000;

pub struct Miner {
    reader: Reader,
    request_handler: RequestHandler,
    rx_nonce_data: mpsc::Receiver<NonceData>,
    /// Kept so the round watchdog can close a round that a stuck drive or a
    /// dead hashing worker would otherwise leave open for good.
    tx_nonce_data: mpsc::Sender<NonceData>,
    target_deadline: u64,
    account_id_to_target_deadline: HashMap<u64, u64>,
    state: Arc<Mutex<State>>,
    reader_task_count: usize,
    get_mining_info_interval: u64,
    executor: Handle,
    wakeup_after: i64,
    submit_only_best: bool,
    /// The running GPU workers and the way to replace one that stopped
    /// answering. `None` when this build has no OpenCL or no GPU worker runs.
    #[cfg(feature = "opencl")]
    gpu_farm: Option<Arc<GpuFarm>>,
}

pub struct State {
    generation_signature: String,
    generation_signature_bytes: [u8; 32],
    height: u64,
    block: u64,
    account_id_to_best_deadline: HashMap<u64, u64>,
    server_target_deadline: u64,
    base_target: u64,
    sw: Stopwatch,
    scanning: bool,
    processed_reader_tasks: usize,
    scoop: u32,
    first: bool,
    outage: bool,
    /// How long the round being scanned has been running, and whether the
    /// watchdog has already said that it is not closing. A round that never
    /// closes used to be silent: no submissions, no lines, nothing to see.
    round_sw: Stopwatch,
    stalled_warned: bool,
    /// The last generation signature refused for not being 32 bytes of hex, so
    /// a pool that keeps sending the same bad one is reported once, not every
    /// tick.
    bad_gensig: String,
}

impl State {
    fn new() -> Self {
        Self {
            generation_signature: "".to_owned(),
            height: 0,
            block: 0,
            scoop: 0,
            account_id_to_best_deadline: HashMap::new(),
            server_target_deadline: u64::MAX,
            base_target: 1,
            processed_reader_tasks: 0,
            sw: Stopwatch::new(),
            generation_signature_bytes: [0; 32],
            scanning: false,
            first: true,
            outage: false,
            round_sw: Stopwatch::new(),
            stalled_warned: false,
            bad_gensig: "".to_owned(),
        }
    }

    fn update_mining_info(&mut self, mining_info: &MiningInfo) -> bool {
        // The signature is a string from the pool, so it can be anything; it
        // used to be decoded with `unwrap()`. Decode it first, before touching
        // the round state, so a refusal leaves the miner where it was.
        let Some(generation_signature_bytes) =
            poc_hashing::decode_gensig(&mining_info.generation_signature)
        else {
            if self.bad_gensig != mining_info.generation_signature {
                error!(
                    "{: <80}",
                    format!(
                        "pool sent a generationSignature that is not 32 bytes of hex: {:?} \
                         -> ignoring this block",
                        mining_info.generation_signature
                    )
                );
                self.bad_gensig = mining_info.generation_signature.clone();
            }
            return false;
        };

        for best_deadlines in self.account_id_to_best_deadline.values_mut() {
            *best_deadlines = u64::MAX;
        }
        self.round_sw.restart();
        self.stalled_warned = false;
        self.height = mining_info.height;
        self.block += 1;
        self.base_target = mining_info.base_target;
        self.server_target_deadline = mining_info.target_deadline;

        self.generation_signature_bytes = generation_signature_bytes;
        self.generation_signature = mining_info.generation_signature.clone();

        let scoop =
            poc_hashing::calculate_scoop(mining_info.height, &self.generation_signature_bytes);
        info!(
            "{: <80}",
            format!("new block: height={}, scoop={}", mining_info.height, scoop)
        );
        self.scoop = scoop;

        self.sw.restart();
        self.processed_reader_tasks = 0;
        self.scanning = true;
        true
    }
}

#[derive(Copy, Clone)]
pub struct NonceData {
    pub height: u64,
    pub block: u64,
    pub base_target: u64,
    pub deadline: u64,
    pub nonce: u64,
    pub reader_task_processed: bool,
    pub account_id: u64,
}

/// Backing store for the plot data a [`Buffer`] hands out.
///
/// The miner reads scoop data into memory that has to be aligned to the page
/// size (the SIMD kernels read whole pages, and the OpenCL host mappings are
/// page aligned anyway), while the GPU buffers are only *views* onto memory
/// owned by the OpenCL driver. A `Vec<u8>` can express neither property: it
/// always releases its block with `align == 1`, and it always assumes
/// ownership of it.
///
/// `AlignedBuffer` keeps the layout it allocated with, so dropping it frees the
/// very same block it allocated - and it can also borrow foreign memory it must
/// not free at all.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    /// `Some` only for memory that was allocated by [`AlignedBuffer::zeroed`]
    /// and therefore has to be released by us.
    layout: Option<Layout>,
}

// The pointer either belongs to us (`layout: Some`, page-aligned, exclusively
// owned) or to the OpenCL driver, which the miner only ever touches under the
// buffer's `Mutex`.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// A buffer that holds no memory and frees nothing.
    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            layout: None,
        }
    }

    /// `len` zeroed bytes, page aligned, released with the matching layout on
    /// drop.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }

        let layout =
            Layout::from_size_align(len, page_size::get()).expect("Invalid layout");

        // SAFETY: `layout` has a non-zero size (checked above).
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("Aligned allocation failed");

        Self {
            ptr,
            len,
            layout: Some(layout),
        }
    }

    /// Views foreign memory without taking ownership of it.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `len` readable and writable bytes for as long as
    /// the view is used, and dropping the view must not invalidate them (the
    /// memory is not freed by us).
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> Self {
        match NonNull::new(ptr) {
            Some(ptr) => Self {
                ptr,
                len,
                layout: None,
            },
            None => Self::empty(),
        }
    }

    /// Repoints a borrowed view at another address, e.g. because the driver
    /// handed out a new host mapping.
    ///
    /// # Safety
    ///
    /// Same requirements as [`AlignedBuffer::from_raw`].
    pub unsafe fn set_raw(&mut self, ptr: *mut u8, len: usize) {
        debug_assert!(self.layout.is_none(), "cannot repoint an owned buffer");
        *self = Self::from_raw(ptr, len);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes in every variant of the type.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: see `deref`; the buffer is exclusively borrowed here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            // SAFETY: `ptr` was allocated by `alloc_zeroed` with exactly this
            // `layout`, and `layout` is only ever `Some` for such buffers.
            unsafe { dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

pub trait Buffer {
    fn get_buffer(&mut self) -> Arc<Mutex<AlignedBuffer>>;
    fn get_buffer_for_writing(&mut self) -> Arc<Mutex<AlignedBuffer>>;
    #[cfg(feature = "opencl")]
    fn get_gpu_buffers(&self) -> Option<&GpuBuffer>;
    #[cfg(feature = "opencl")]
    fn get_gpu_data(&self) -> Option<Mem>;
    fn unmap(&mut self);
    fn get_id(&self) -> usize;
    /// Which GPU worker generation this buffer belongs to; zero for buffers that
    /// are not tied to a device context.
    fn generation(&self) -> u64 {
        0
    }
}

pub struct CpuBuffer {
    data: Arc<Mutex<AlignedBuffer>>,
}

impl CpuBuffer {
    pub fn new(buffer_size: usize) -> Self {
        // Zero-length buffers are only used as `gpu_signal` carriers: no worker
        // ever reads from or writes to them. `AlignedBuffer::zeroed` returns an
        // empty, allocation-free buffer for them.
        CpuBuffer {
            data: Arc::new(Mutex::new(AlignedBuffer::zeroed(buffer_size))),
        }
    }
}

impl Buffer for CpuBuffer {
    fn get_buffer(&mut self) -> Arc<Mutex<AlignedBuffer>> {
        self.data.clone()
    }
    fn get_buffer_for_writing(&mut self) -> Arc<Mutex<AlignedBuffer>> {
        self.data.clone()
    }
    #[cfg(feature = "opencl")]
    fn get_gpu_buffers(&self) -> Option<&GpuBuffer> {
        None
    }
    #[cfg(feature = "opencl")]
    fn get_gpu_data(&self) -> Option<Mem> {
        None
    }
    fn unmap(&mut self) {}
    fn get_id(&self) -> usize {
        0
    }
}

fn scan_plots(
    plot_dirs: &[PathBuf],
    use_direct_io: bool,
    dummy: bool,
) -> (HashMap<String, Arc<Vec<Mutex<Plot>>>>, u64) {
    let mut drive_id_to_plots: HashMap<String, Vec<Mutex<Plot>>> = HashMap::new();
    let mut global_capacity: u64 = 0;

    for plot_dir in plot_dirs {
        let mut num_plots = 0;
        let mut local_capacity: u64 = 0;
        for file in read_dir(plot_dir).unwrap() {
            let file = &file.unwrap().path();

            if let Ok(p) = Plot::new(file, use_direct_io, dummy) {
                let drive_id = get_device_id(&file.to_str().unwrap().to_string());
                let plots = drive_id_to_plots.entry(drive_id).or_insert(Vec::new());

                local_capacity += p.meta.nonces as u64;
                plots.push(Mutex::new(p));
                num_plots += 1;
            }
        }

        info!(
            "path={}, files={}, size={:.4} TiB",
            plot_dir.to_str().unwrap(),
            num_plots,
            local_capacity as f64 / 4.0 / 1024.0 / 1024.0
        );

        global_capacity += local_capacity;
        if num_plots == 0 {
            warn!("no plots in {}", plot_dir.to_str().unwrap());
        }
    }

    // sort plots by filetime and get them into an arc
    let drive_id_to_plots: HashMap<String, Arc<Vec<Mutex<Plot>>>> = drive_id_to_plots
        .drain()
        .map(|(drive_id, mut plots)| {
            plots.sort_by_key(|p| {
                let m = p.lock().unwrap().fh.metadata().unwrap();
                -FileTime::from_last_modification_time(&m).unix_seconds()
            });
            (drive_id, Arc::new(plots))
        })
        .collect();

    info!(
        "plot files loaded: total drives={}, total capacity={:.4} TiB",
        drive_id_to_plots.len(),
        global_capacity as f64 / 4.0 / 1024.0 / 1024.0
    );

    (drive_id_to_plots, global_capacity * 64)
}

/// Everything needed to build the GPU workers, kept so a worker that stopped
/// answering can be built again without restarting the miner.
#[cfg(feature = "opencl")]
struct GpuCfg {
    benchmark: bool,
    platform: usize,
    device: usize,
    nonces_per_cache: usize,
    mapping: bool,
    /// How many buffers each worker should own; index is the worker's.
    buffers_per_worker: Vec<usize>,
}

/// The GPU side of the miner: the workers that are running, and the way to
/// replace them.
#[cfg(feature = "opencl")]
struct GpuFarm {
    cfg: GpuCfg,
    workers: Arc<GpuWorkers>,
    tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
    /// Only one rebuild at a time. A driver call that never returns would
    /// otherwise leave a new rebuild thread behind every time the stall check
    /// looks at the farm.
    rebuilding: Arc<AtomicBool>,
}

#[cfg(feature = "opencl")]
impl GpuFarm {
    /// Replaces the workers on a thread of their own.
    ///
    /// The old worker is stuck inside a driver call that may never return, and
    /// building a new context talks to the same driver: if that call blocks too,
    /// it must not block the task that starts rounds - that is how the miner
    /// went silent for hours in the first place.
    fn rebuild(&self, reason: &str) {
        if self.rebuilding.swap(true, Ordering::SeqCst) {
            return;
        }
        if self.workers.since_rebuild_ms() < GPU_REBUILD_BACKOFF_MS {
            self.rebuilding.store(false, Ordering::SeqCst);
            return;
        }
        error!(
            "GPU worker: {} -> building a new one in the background (generation {}); the drives \
             keep scanning, chunks already handed to the old worker are dropped",
            reason,
            self.workers.generation()
        );

        let cfg = GpuCfg {
            benchmark: self.cfg.benchmark,
            platform: self.cfg.platform,
            device: self.cfg.device,
            nonces_per_cache: self.cfg.nonces_per_cache,
            mapping: self.cfg.mapping,
            buffers_per_worker: self.cfg.buffers_per_worker.clone(),
        };
        let workers = self.workers.clone();
        let tx_empty_buffers = self.tx_empty_buffers.clone();
        let tx_nonce_data = self.tx_nonce_data.clone();
        let rebuilding = self.rebuilding.clone();

        thread::spawn(move || {
            let built = std::panic::catch_unwind(AssertUnwindSafe(|| {
                spawn_gpu_workers(&cfg, &workers, &tx_empty_buffers, &tx_nonce_data);
            }));
            rebuilding.store(false, Ordering::SeqCst);
            match built {
                Ok(()) => error!(
                    "GPU worker: generation {} is running; the round carries on with it",
                    workers.generation()
                ),
                Err(_) => error!(
                    "GPU worker: rebuilding failed - the device or its driver is in trouble; \
                     rounds still close but no shares are hashed until the next attempt"
                ),
            }
        });
    }
}

/// Builds the GPU workers, their buffers and their channels, and puts them in
/// place. Used at startup and again whenever a worker has to be replaced.
#[cfg(feature = "opencl")]
fn spawn_gpu_workers(
    cfg: &GpuCfg,
    workers: &Arc<GpuWorkers>,
    tx_empty_buffers: &Sender<Box<dyn Buffer + Send>>,
    tx_nonce_data: &mpsc::Sender<NonceData>,
) {
    let mut txs = Vec::with_capacity(cfg.buffers_per_worker.len());
    let mut rxs = Vec::with_capacity(cfg.buffers_per_worker.len());
    for _ in 0..cfg.buffers_per_worker.len() {
        let (tx, rx) = crossbeam_channel::unbounded();
        txs.push(tx);
        rxs.push(rx);
    }

    // A fresh context per worker: a wedged one is exactly what is being replaced.
    let contexts: Vec<Arc<GpuContext>> = (0..cfg.buffers_per_worker.len())
        .map(|_| {
            Arc::new(GpuContext::new(
                cfg.platform,
                cfg.device,
                cfg.nonces_per_cache,
                cfg.mapping,
            ))
        })
        .collect();

    workers.install(txs);
    let generation = workers.generation();

    // The buffers are allocated in the context of the worker that will use
    // them, and carry its generation so a buffer left over from a replaced
    // worker is retired by the reader instead of being filled for a worker that
    // cannot use it.
    for (i, context) in contexts.iter().enumerate() {
        for _ in 0..cfg.buffers_per_worker[i] {
            let gpu_buffer = GpuBuffer::new(context, i + 1, generation);
            if tx_empty_buffers
                .try_send(Box::new(gpu_buffer) as Box<dyn Buffer + Send>)
                .is_err()
            {
                warn!(
                    "GPU worker: buffer pool is full, not adding another buffer for worker {}",
                    i + 1
                );
            }
        }
    }

    for context in contexts.into_iter() {
        let gpu_workers = workers.clone();
        let rx = rxs.remove(0);
        let tx_empty_buffers = tx_empty_buffers.clone();
        let tx_nonce_data = tx_nonce_data.clone();
        thread::spawn(create_gpu_worker_task(
            cfg.benchmark,
            rx,
            tx_empty_buffers,
            tx_nonce_data,
            context,
            gpu_workers,
        ));
    }
}

impl Miner {
    pub fn new(cfg: Cfg, executor: Handle) -> Miner {
        let (drive_id_to_plots, total_size) =
            scan_plots(&cfg.plot_dirs, cfg.hdd_use_direct_io, cfg.benchmark_cpu());

        let cpu_threads = cfg.cpu_threads.max(1);
        info!("🖥️  Using {} CPU thread(s)", cpu_threads);
        let cpu_worker_task_count = cfg.cpu_worker_task_count;

        let cpu_buffer_count = cpu_worker_task_count
            + if cpu_worker_task_count > 0 {
                cpu_threads
            } else {
                0
            };

        let reader_thread_count = if cfg.hdd_reader_thread_count == 0 {
            drive_id_to_plots.len()
        } else {
            cfg.hdd_reader_thread_count
        };

        #[cfg(feature = "opencl")]
        let gpu_worker_task_count = cfg.gpu_worker_task_count;
        #[cfg(feature = "opencl")]
        let gpu_threads = cfg.gpu_threads;
        #[cfg(feature = "opencl")]
        let gpu_buffer_count = if gpu_worker_task_count > 0 {
            if cfg.gpu_async {
                gpu_worker_task_count + 2 * gpu_threads
            } else {
                gpu_worker_task_count + gpu_threads
            }
        } else {
            0
        };
        #[cfg(feature = "opencl")]
        {
            info!(
                "reader-threads={}, CPU-threads={}, GPU-threads={}",
                reader_thread_count, cpu_threads, gpu_threads,
            );
            info!("→ Starting now");
            info!(
                "CPU-buffer={}(+{}), GPU-buffer={}(+{})",
                cpu_worker_task_count,
                if cpu_worker_task_count > 0 {
                    cpu_threads
                } else {
                    0
                },
                gpu_worker_task_count,
                if gpu_worker_task_count > 0 {
                    if cfg.gpu_async {
                        2 * gpu_threads
                    } else {
                        gpu_threads
                    }
                } else {
                    0
                }
            );

            {
                if cpu_threads * cpu_worker_task_count + gpu_threads * gpu_worker_task_count == 0 {
                    error!("CPU, GPU: no active workers. Check thread and task configuration. Shutting down...");
                    process::exit(0);
                }
            }
        }

        #[cfg(not(feature = "opencl"))]
        {
            info!(
                "reader-threads={} CPU-threads={}",
                reader_thread_count, cpu_threads
            );
            info!("CPU-buffer={}(+{})", cpu_worker_task_count, cpu_threads);
            {
                if cpu_threads * cpu_worker_task_count == 0 {
                    error!(
                    "CPU: no active workers. Check thread and task configuration. Shutting down..."
                );
                    process::exit(0);
                }
            }
        }

        #[cfg(not(feature = "opencl"))]
        let buffer_count = cpu_buffer_count;
        #[cfg(feature = "opencl")]
        let buffer_count = cpu_buffer_count + gpu_buffer_count;
        let buffer_size_cpu = cfg.cpu_nonces_per_cache * SCOOP_SIZE as usize;
        // The pool is one generation of buffers larger than the number of
        // buffers, so the buffers of a rebuilt GPU worker always fit next to the
        // ones the old worker is still holding. The surplus is retired by the
        // readers as they come across it.
        #[cfg(feature = "opencl")]
        let pool_capacity = buffer_count as usize + gpu_buffer_count;
        #[cfg(not(feature = "opencl"))]
        let pool_capacity = buffer_count as usize;
        let (tx_empty_buffers, rx_empty_buffers) = crossbeam_channel::bounded(pool_capacity);
        let (tx_read_replies_cpu, rx_read_replies_cpu) =
            crossbeam_channel::bounded(cpu_buffer_count);

        for _ in 0..cpu_buffer_count {
            let cpu_buffer = CpuBuffer::new(buffer_size_cpu);
            tx_empty_buffers
                .send(Box::new(cpu_buffer) as Box<dyn Buffer + Send>)
                .unwrap();
        }

        let (tx_nonce_data, rx_nonce_data) = mpsc::channel(buffer_count);

        thread::spawn({
            create_cpu_worker_task(
                cfg.benchmark_io(),
                new_thread_pool(cpu_threads, cfg.cpu_thread_pinning),
                rx_read_replies_cpu.clone(),
                tx_empty_buffers.clone(),
                tx_nonce_data.clone(),
            )
        });

        #[cfg(feature = "opencl")]
        if cfg.gpu_async {
            warn!(
                "gpu_async is not supported by this build (the async GPU worker was removed \
                 with the futures 0.1 -> 0.3 migration); using the synchronous GPU worker \
                 instead. Prefer gpu_mem_mapping = true."
            );
        }

        // The GPU side is built by the same function that rebuilds it later, so a
        // worker that had to be replaced is built exactly like these are.
        #[cfg(feature = "opencl")]
        let gpu_farm = if gpu_threads > 0 {
            let workers = GpuWorkers::new(Vec::new());
            let farm = GpuFarm {
                cfg: GpuCfg {
                    benchmark: cfg.benchmark_io(),
                    platform: cfg.gpu_platform,
                    device: cfg.gpu_device,
                    nonces_per_cache: cfg.gpu_nonces_per_cache,
                    mapping: if cfg.benchmark_io() {
                        false
                    } else {
                        cfg.gpu_mem_mapping
                    },
                    buffers_per_worker: (0..gpu_threads)
                        .map(|i| {
                            gpu_buffer_count / gpu_threads
                                + if i == 0 {
                                    gpu_buffer_count % gpu_threads
                                } else {
                                    0
                                }
                        })
                        .collect(),
                },
                workers: workers.clone(),
                tx_empty_buffers: tx_empty_buffers.clone(),
                tx_nonce_data: tx_nonce_data.clone(),
                rebuilding: Arc::new(AtomicBool::new(false)),
            };
            spawn_gpu_workers(
                &farm.cfg,
                &farm.workers,
                &farm.tx_empty_buffers,
                &farm.tx_nonce_data,
            );
            Some(Arc::new(farm))
        } else {
            None
        };

        #[cfg(feature = "opencl")]
        let tx_read_replies_gpu = gpu_farm.as_ref().map(|farm| farm.workers.clone());
        #[cfg(not(feature = "opencl"))]
        let tx_read_replies_gpu = None;

        // The round watchdog and the submission loop both need the nonce
        // channel's sender: the first to close a round that got stuck.
        Miner {
            reader_task_count: drive_id_to_plots.len(),
            reader: Reader::new(
                drive_id_to_plots,
                total_size,
                reader_thread_count,
                rx_empty_buffers,
                tx_empty_buffers,
                tx_read_replies_cpu,
                tx_read_replies_gpu,
                cfg.show_progress,
                cfg.show_drive_stats,
                cfg.cpu_thread_pinning,
                cfg.benchmark_cpu(),
            ),
            #[cfg(feature = "opencl")]
            gpu_farm,
            rx_nonce_data,
            tx_nonce_data: tx_nonce_data.clone(),
            target_deadline: cfg.target_deadline,
            account_id_to_target_deadline: cfg.account_id_to_target_deadline,
            request_handler: RequestHandler::new(
                cfg.url,
                cfg.account_id_to_secret_phrase,
                cfg.timeout,
                (total_size * 4 / 1024 / 1024) as usize,
                cfg.send_proxy_details,
                cfg.additional_headers,
                executor.clone(),
            ),
            state: Arc::new(Mutex::new(State::new())),
            // floor at 1s to protect servers
            get_mining_info_interval: max(1000, cfg.get_mining_info_interval),
            executor,
            wakeup_after: cfg.hdd_wakeup_after * 1000, // ms -> s
            submit_only_best : cfg.submit_only_best,
        }
    }

    pub async fn run(self) {
        use tokio::time::{sleep, Duration};
        let request_handler = self.request_handler.clone();
        let total_size = self.reader.total_size;

        // TODO: this doesn't need to be arc mutex if we manage to separate
        // reader from miner so that we can simply move it
        let reader = Arc::new(Mutex::new(self.reader));


        let state = self.state.clone();
        // there might be a way to solve this without two nested moves
        let get_mining_info_interval = self.get_mining_info_interval;
        let wakeup_after = self.wakeup_after;
        let tx_nonce_data = self.tx_nonce_data.clone();
        let reader_task_count = self.reader_task_count;
        #[cfg(feature = "opencl")]
        let gpu_farm = self.gpu_farm.clone();
        tokio::spawn(async move {
            info!("→ Interval task started");
            Interval::new_interval(Duration::from_millis(get_mining_info_interval))
                .for_each(move |_| {
                    let state = state.clone();
                    let reader = reader.clone();
                    let request_handler = request_handler.clone();
                    let tx_nonce_data = tx_nonce_data.clone();
                    #[cfg(feature = "opencl")]
                    let gpu_farm = gpu_farm.clone();
                    async move {
                        // The pool side can go wrong in more ways than a
                        // returned error: building the request parses headers,
                        // and the answer is parsed too. A panic in there must
                        // not end the only task that starts rounds.
                        let mining_info = AssertUnwindSafe(request_handler.get_mining_info())
                            .catch_unwind()
                            .await;
                        // Starting a round is this task's one job and no other
                        // task does it, so the setup runs behind `catch_panic`
                        // as well: it touches the reader tasks and the thread
                        // pool, and a panic there has to cost one round, not
                        // every round from now on.
                        catch_panic("miner: starting a round", || match mining_info {
                                Ok(Ok(mining_info)) => {
                                    let mut state = lock(&state);
                                    state.first = false;
                                    if state.outage {
                                        error!("{: <80}", "outage resolved.");
                                        state.outage = false;
                                    }
                                    if mining_info.generation_signature != state.generation_signature {
                                        // A refused signature must not start a
                                        // round: there would be nothing to scan
                                        // for.
                                        if state.update_mining_info(&mining_info) {
                                            let skipped = lock(&reader).start_reading(
                                                mining_info.height,
                                                state.block,
                                                mining_info.base_target,
                                                state.scoop,
                                                &Arc::new(state.generation_signature_bytes),
                                            );
                                            // A drive whose previous task never came
                                            // back cannot report this round, so it is
                                            // counted as reported here. Otherwise the
                                            // round waits for an answer that will never
                                            // arrive, and the miner goes quiet with it.
                                            state.processed_reader_tasks += skipped;
                                            if skipped >= reader_task_count {
                                                error!(
                                                    "{: <80}",
                                                    format!(
                                                        "no drive could start height={}: all {} \
                                                         drives are busy from earlier rounds",
                                                        mining_info.height, reader_task_count
                                                    )
                                                );
                                                let _ = tx_nonce_data.try_send(NonceData {
                                                    height: mining_info.height,
                                                    block: state.block,
                                                    base_target: mining_info.base_target,
                                                    deadline: u64::MAX,
                                                    nonce: 0,
                                                    reader_task_processed: true,
                                                    account_id: 0,
                                                });
                                            }
                                        }
                                        drop(state);
                                    } else if state.height > 0
                                        && state.processed_reader_tasks < reader_task_count
                                        && state.round_sw.elapsed_ms() > ROUND_STALL_MS
                                    {
                                        // A round that does not close: a drive task is
                                        // waiting for a buffer that never comes back, or
                                        // a hashing worker is gone. Close the round so
                                        // the miner keeps submitting what it found, and
                                        // say why - silence was the only symptom before.
                                        if !state.stalled_warned {
                                            state.stalled_warned = true;
                                            let free = lock(&reader).free_buffers();
                                            error!(
                                                "round for height={} has not closed for {}s ({} of {} \
                                                 reader tasks reported, {} buffer(s) free) -> closing it \
                                                 and carrying on",
                                                state.height,
                                                state.round_sw.elapsed_ms() / 1000,
                                                state.processed_reader_tasks,
                                                reader_task_count,
                                                free
                                            );
                                        }
                                        let (height, block, base_target) =
                                            (state.height, state.block, state.base_target);
                                        drop(state);
                                        let _ = tx_nonce_data.try_send(NonceData {
                                            height,
                                            block,
                                            base_target,
                                            deadline: u64::MAX,
                                            nonce: 0,
                                            reader_task_processed: true,
                                            account_id: 0,
                                        });
                                    } else if !state.scanning
                                        && wakeup_after != 0
                                        && state.sw.elapsed_ms() > wakeup_after
                                    {
                                        info!("HDD, wakeup!");
                                        lock(&reader).wakeup();
                                        state.sw.restart();
                                    }
                                }
                                Ok(Err(_)) => {
                                    let mut state = lock(&state);
                                    if state.first {
                                        error!(
                                            "{: <80}",
                                            "error getting mining info, please check server config"
                                        );
                                        state.first = false;
                                        state.outage = true;
                                    } else if !state.outage {
                                        error!(
                                            "{: <80}",
                                            "error getting mining info => connection outage..."
                                        );
                                        state.outage = true;
                                    }
                                }
                                Err(_) => {
                                    let mut state = lock(&state);
                                    error!(
                                        "{: <80}",
                                        "getting mining info panicked -> carrying on"
                                    );
                                    state.first = false;
                                    state.outage = true;
                                }
                        });

                        // A GPU worker that stopped answering cannot be talked
                        // to again: the driver call it sits in may never return.
                        // Build a new one instead - on its own thread, so a
                        // driver that blocks the rebuild too cannot block this
                        // task, which is the only one that starts rounds.
                        #[cfg(feature = "opencl")]
                        if let Some(gpu) = gpu_farm.as_ref() {
                            let scanning = lock(&state).scanning;
                            if scanning {
                                let free = lock(&reader).free_buffers();
                                let failures = gpu.workers.failures_in_a_row();
                                let silent = gpu.workers.silent_for_ms();
                                let reason = if failures >= GPU_FAILURES_IN_A_ROW {
                                    Some(format!("{} windows in a row failed", failures))
                                } else if silent > GPU_SILENT_MS && free == 0 {
                                    Some(format!(
                                        "no window hashed for {}s and not one buffer came back",
                                        silent / 1000
                                    ))
                                } else {
                                    None
                                };
                                if let Some(reason) = reason {
                                    gpu.rebuild(&reason);
                                }
                            }
                        }
                    }
                })
                .await; 
        });

        // only start submitting nonces after a while
        let mut best_nonce_data = NonceData {
            height: 0,
            block: 0,
            base_target: 0,
            deadline: 0,
            nonce: 0,
            reader_task_processed: false,
            account_id: 0,
        };

        let target_deadline = self.target_deadline;
        let account_id_to_target_deadline = self.account_id_to_target_deadline;
        let request_handler = self.request_handler.clone();
        let state = self.state.clone();
        let reader_task_count = self.reader_task_count;
        let inner_submit_only_best = self.submit_only_best;
        self.executor.clone().spawn(
            ReceiverStream::new(self.rx_nonce_data)
                .for_each(move |nonce_data| {
                    // Every share from here on goes through this task, so a
                    // single bad share - a header that will not parse, an
                    // answer that will not either - must not be able to end it.
                    catch_panic("miner: handling a share", || {
                        let mut state = lock(&state);

                        let deadline = nonce_data.deadline / nonce_data.base_target;
                        if state.height == nonce_data.height {
                            let best_deadline = *state
                                .account_id_to_best_deadline
                                .get(&nonce_data.account_id)
                                .unwrap_or(&u64::MAX);
                            if best_deadline > deadline
                                && deadline
                                    < min(
                                        state.server_target_deadline,
                                        *(account_id_to_target_deadline
                                            .get(&nonce_data.account_id)
                                            .unwrap_or(&target_deadline)),
                                    )
                            {
                                // The round's best so far. A pool refusing a share
                                // costs a request and nothing else, so nothing is
                                // held back on a guess about the pool's limit: the
                                // answer is the only way to know it.
                                state
                                    .account_id_to_best_deadline
                                    .insert(nonce_data.account_id, deadline);

                                if inner_submit_only_best {
                                    best_nonce_data = nonce_data.clone();
                                } else {
                                    request_handler.submit_nonce(
                                        nonce_data.account_id,
                                        nonce_data.nonce,
                                        nonce_data.height,
                                        nonce_data.block,
                                        nonce_data.deadline,
                                        deadline,
                                        state.generation_signature_bytes,
                                    );
                                }
                            }

                            if nonce_data.reader_task_processed {
                                state.processed_reader_tasks += 1;
                                if state.processed_reader_tasks >= reader_task_count {
                                    let elapsed = state.sw.elapsed_ms();
                                    // A round that closed without a single scan
                                    // has no speed worth printing; the line above
                                    // already says why it closed.
                                    let speed = if elapsed > 0 {
                                        format!(
                                            "{:.2}MiB/s",
                                            total_size as f64 * 1000.0
                                                / 1024.0
                                                / 1024.0
                                                / elapsed as f64
                                        )
                                    } else {
                                        "nothing was scanned".to_string()
                                    };
                                    info!(
                                        "{: <80}",
                                        format!(
                                            "round finished: roundtime={}ms, speed={}",
                                            elapsed, speed
                                        )
                                    );

                                    // Submit now our best one, if configured that way
                                    if best_nonce_data.height == state.height {
                                        let deadline = best_nonce_data.deadline / best_nonce_data.base_target;
                                        request_handler.submit_nonce(
                                            best_nonce_data.account_id,
                                            best_nonce_data.nonce,
                                            best_nonce_data.height,
                                            best_nonce_data.block,
                                            best_nonce_data.deadline,
                                            deadline,
                                            state.generation_signature_bytes,
                                        );
                                    }

                                    state.sw.restart();
                                    state.scanning = false;
                                }
                            }
                        }
                    });
                    futures_util::future::ready(())
                }),
        );
        loop {
        sleep(Duration::from_secs(60)).await;
        }
    }

}

#[cfg(test)]
mod tests {
    use super::AlignedBuffer;

    /// Regression guard for the heap corruption that killed the GPU worker:
    /// `CpuBuffer` used to hand page-aligned memory to a `Vec<u8>`, which frees
    /// its block with `align == 1`. Dropping such a buffer corrupted the heap
    /// (STATUS_HEAP_CORRUPTION) instead of failing loudly, so the only way to
    /// catch a regression is to allocate, use and drop real buffers.
    #[test]
    fn aligned_buffer_frees_with_its_own_layout() {
        for _ in 0..100 {
            let mut buffer = AlignedBuffer::zeroed(3 * 1024 * 1024);
            assert_eq!(buffer.len(), 3 * 1024 * 1024);
            assert_eq!(
                buffer.as_ptr() as usize % page_size::get(),
                0,
                "buffer must stay page aligned"
            );
            buffer[0] = 1;
            let last = buffer.len() - 1;
            buffer[last] = 2;
            drop(buffer);
        }
    }

    /// The `gpu_signal` dummies are zero-sized and must not allocate.
    #[test]
    fn empty_buffer_allocates_nothing() {
        let mut buffer = AlignedBuffer::zeroed(0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert!(buffer.iter().next().is_none());
        drop(buffer);
    }
}
