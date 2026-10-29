use crate::miner::{Buffer, NonceData};
use crate::ocl::GpuContext;
use crate::ocl::{gpu_hash, gpu_transfer, GpuError};
use crate::reader::{BufferInfo, GpuWorkers, ReadReply};
use crossbeam_channel::{Receiver, Sender};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::u64;
use tokio::sync::mpsc::Sender as TokioSender;

/// How many hashing failures in a row before this worker stops hashing.
const MAX_FAILURES: u32 = 5;

/// How many times one window is hashed before it is considered lost. A failed
/// driver call is reported by `ocl.rs` now, so a second try only costs the
/// window's transfer; more than that and the device itself is in trouble.
const MAX_WINDOW_ATTEMPTS: u32 = 2;

/// A worker that has given up says so once, and then this often, so a farm
/// running at reduced capacity does not look like a farm running fine.
const GIVE_UP_REPORT_EVERY: u64 = 50;

/// A drive that reached the end of its plots, so the miner can count it and
/// close the round.
fn report_drive_finished(tx_nonce_data: &TokioSender<NonceData>, info: &BufferInfo) {
    let _ = tx_nonce_data.blocking_send(NonceData {
        height: info.height,
        block: info.block,
        base_target: info.base_target,
        deadline: u64::MAX,
        nonce: 0,
        reader_task_processed: true,
        account_id: info.account_id,
    });
}

/// Hands a buffer back to the pool without blocking. See `reader::return_buffer`:
/// after a rebuild there can be more buffers than the pool was made for.
fn return_buffer(tx_empty_buffers: &Sender<Box<dyn Buffer + Send>>, buffer: Box<dyn Buffer + Send>) {
    match tx_empty_buffers.try_send(buffer) {
        Ok(()) => (),
        Err(crossbeam_channel::TrySendError::Full(_)) => (),
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => (),
    }
}

pub fn create_gpu_worker_task(
    benchmark: bool,
    rx_read_replies: Receiver<ReadReply>,
    tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
    tx_nonce_data: TokioSender<NonceData>,
    context_mu: Arc<GpuContext>,
    gpu_workers: Arc<GpuWorkers>,
) -> impl FnOnce() + Send + 'static {
    move || {
        // Failures in a row. A worker whose driver calls keep failing cannot
        // hash anything any more, but it can still hand the buffers back, and
        // that alone is what keeps the rounds closing - and `gpu_workers` is
        // what tells the miner to build a new one.
        let mut failures = 0u32;
        let mut gave_up = false;
        let mut dropped = 0u64;

        for read_reply in rx_read_replies {
            let buffer = read_reply.buffer;
            let info = read_reply.info;
            // handle empty buffers (read errors) && benchmark
            if info.len == 0 || benchmark {
                // forward 'drive finished signal'
                if info.finished {
                    report_drive_finished(&tx_nonce_data, &info);
                }
                return_buffer(&tx_empty_buffers, buffer);
                continue;
            }

            // consume and ignore all signals
            if info.len == 1 && info.gpu_signal > 0 {
                continue;
            }

            if gave_up {
                dropped += 1;
                if dropped % GIVE_UP_REPORT_EVERY == 0 {
                    warn!(
                        "GPU worker: still not hashing ({} windows dropped since it gave up; \
                         waiting for it to be rebuilt)",
                        dropped
                    );
                }
                if info.finished {
                    report_drive_finished(&tx_nonce_data, &info);
                }
                return_buffer(&tx_empty_buffers, buffer);
                continue;
            }

            // `ocl.rs` reports a failed driver call as a value now, so a failing
            // window is not a panic any more. The window still gets a second
            // try - a transient failure costs nothing that way - and only a
            // window that keeps failing is dropped, with the buffer handed back
            // either way. `catch_unwind` stays as a backstop: a panic in a
            // spawned thread is one thing, but rayon taught this fork what an
            // unexpected one can do.
            let mut hashed = None;
            let mut last_error: Option<String> = None;
            for attempt in 1..=MAX_WINDOW_ATTEMPTS {
                let tried = panic::catch_unwind(AssertUnwindSafe(|| {
                    let window = || -> Result<(u64, u64), GpuError> {
                        let gpu_buffer = match buffer.get_gpu_buffers() {
                            Some(gpu_buffer) => gpu_buffer,
                            None => {
                                return Err(GpuError {
                                    call: "buffer",
                                    detail: "not a GPU buffer".to_string(),
                                })
                            }
                        };
                        let data_gpu = match buffer.get_gpu_data() {
                            Some(data_gpu) => data_gpu,
                            None => {
                                return Err(GpuError {
                                    call: "buffer",
                                    detail: "no device memory".to_string(),
                                })
                            }
                        };
                        gpu_transfer(&context_mu, gpu_buffer, *info.gensig)?;
                        gpu_hash(&context_mu, info.len / 64, &data_gpu)
                    };
                    window()
                }));
                match tried {
                    Ok(Ok(result)) => {
                        hashed = Some(result);
                        break;
                    }
                    Ok(Err(err)) if attempt < MAX_WINDOW_ATTEMPTS => {
                        warn!(
                            "GPU worker: hashing failed at height={}, window starting at nonce={} \
                             ({}) -> trying it again (attempt {} of {})",
                            info.height,
                            info.start_nonce,
                            err,
                            attempt,
                            MAX_WINDOW_ATTEMPTS
                        );
                    }
                    Ok(Err(err)) => {
                        last_error = Some(err.to_string());
                        break;
                    }
                    Err(_) if attempt < MAX_WINDOW_ATTEMPTS => warn!(
                        "GPU worker: hashing panicked at height={}, window starting at nonce={} \
                         -> trying it again (attempt {} of {})",
                        info.height, info.start_nonce, attempt, MAX_WINDOW_ATTEMPTS
                    ),
                    Err(_) => {
                        last_error = Some("panicked".to_string());
                        break;
                    }
                }
            }

            // Every window the worker looked at is a sign of life, whether it
            // hashed or failed; only silence is not, and the miner watches that.
            gpu_workers.note_window(hashed.is_some());

            match hashed {
                Some((deadline, offset)) => {
                    failures = 0;
                    let _ = tx_nonce_data.blocking_send(NonceData {
                        height: info.height,
                        block: info.block,
                        base_target: info.base_target,
                        deadline,
                        nonce: offset + info.start_nonce,
                        reader_task_processed: info.finished,
                        account_id: info.account_id,
                    });
                }
                None => {
                    failures += 1;
                    dropped += 1;
                    error!(
                        "GPU worker: window at height={}, nonce={} failed {} times in a row ({}) \
                         -> dropping it ({} window(s) dropped so far)",
                        info.height,
                        info.start_nonce,
                        MAX_WINDOW_ATTEMPTS,
                        last_error.unwrap_or_else(|| "unknown error".to_string()),
                        dropped
                    );
                    if failures >= MAX_FAILURES {
                        gave_up = true;
                        error!(
                            "GPU worker: giving up hashing after {} failures in a row; buffers will \
                             still be returned, and the miner will rebuild this worker",
                            failures
                        );
                    } else if info.finished {
                        report_drive_finished(&tx_nonce_data, &info);
                    }
                }
            }

            return_buffer(&tx_empty_buffers, buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use self::core::Event;
    use crate::ocl::gpu_hash;
    use crate::ocl::GpuContext;
    use hex;
    use ocl_core as core;
    use std::sync::Arc;
    use std::u64;

    #[test]
    fn test_deadline_hashing() {
        let len: u64 = 16;
        let gensig =
            hex::decode("4a6f686e6e7946464d206861742064656e206772f6df74656e2050656e697321")
                .unwrap();
        let mut data: [u8; 64 * 16] = [0; 64 * 16];
        for i in 0..32 {
            data[i * 32..i * 32 + 32].clone_from_slice(&gensig);
        }

        let context = Arc::new(GpuContext::new(0, 0, 16, false));

        let buffer_gpu = unsafe {
            core::create_buffer::<_, u8>(&context.context, core::MEM_READ_ONLY, 64 * 16, None)
                .unwrap()
        };

        unsafe {
            core::enqueue_write_buffer(
                &context.queue_transfer,
                &context.gensig_gpu,
                true,
                0,
                &gensig,
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }

        unsafe {
            core::enqueue_write_buffer(
                &context.queue_transfer,
                &buffer_gpu,
                true,
                0,
                &data,
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }

        let result = gpu_hash(&context, len as usize, &buffer_gpu)
            .expect("the window hashes");
        assert_eq!(18043101931632730606u64, result.0);
    }
}
