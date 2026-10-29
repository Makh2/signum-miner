use crate::miner::{Buffer, NonceData};
use crate::ocl::GpuContext;
use crate::ocl::{gpu_hash, gpu_transfer};
use crate::reader::{BufferInfo, ReadReply};
use crossbeam_channel::{Receiver, Sender};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::u64;
use tokio::sync::mpsc::Sender as TokioSender;

/// How many hashing failures in a row before this worker stops hashing.
const MAX_FAILURES: u32 = 5;

/// How many times one window is hashed before it is considered lost. The
/// failures here are OpenCL calls that panicked, so a second try is cheap and
/// often enough; more than that and the device itself is in trouble.
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

pub fn create_gpu_worker_task(
    benchmark: bool,
    rx_read_replies: Receiver<ReadReply>,
    tx_empty_buffers: Sender<Box<dyn Buffer + Send>>,
    tx_nonce_data: TokioSender<NonceData>,
    context_mu: Arc<GpuContext>,
) -> impl FnOnce() + Send + 'static {
    move || {
        // Failures in a row. A worker whose OpenCL calls keep failing cannot
        // hash anything any more, but it can still hand the buffers back, and
        // that alone is what keeps the rounds closing.
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
                let _ = tx_empty_buffers.send(buffer);
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
                         restart the miner to bring it back)",
                        dropped
                    );
                }
                if info.finished {
                    report_drive_finished(&tx_nonce_data, &info);
                }
                let _ = tx_empty_buffers.send(buffer);
                continue;
            }

            // An OpenCL call that fails panics - ocl.rs unwraps its results - so
            // the window is hashed under catch_unwind. One failure used to cost
            // the whole window, and a panicking worker used to take the round
            // down with it. Now the window gets a second try - a transient
            // failure then costs nothing - and only a window that keeps failing
            // is dropped, with the buffer handed back either way.
            let mut hashed = None;
            for attempt in 1..=MAX_WINDOW_ATTEMPTS {
                let tried = panic::catch_unwind(AssertUnwindSafe(|| {
                    gpu_transfer(
                        &context_mu,
                        buffer.get_gpu_buffers().unwrap(),
                        *info.gensig,
                    );
                    gpu_hash(
                        &context_mu,
                        info.len / 64,
                        buffer.get_gpu_data().as_ref().unwrap(),
                    )
                }));
                match tried {
                    Ok(result) => {
                        hashed = Some(result);
                        break;
                    }
                    Err(_) if attempt < MAX_WINDOW_ATTEMPTS => warn!(
                        "GPU worker: hashing failed at height={}, window starting at nonce={} \
                         -> trying it again (attempt {} of {})",
                        info.height, info.start_nonce, attempt, MAX_WINDOW_ATTEMPTS
                    ),
                    Err(_) => break,
                }
            }

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
                        "GPU worker: window at height={}, nonce={} failed {} times in a row \
                         -> dropping it ({} window(s) dropped so far)",
                        info.height, info.start_nonce, MAX_WINDOW_ATTEMPTS, dropped
                    );
                    if failures >= MAX_FAILURES {
                        gave_up = true;
                        error!(
                            "GPU worker: giving up hashing after {} failures in a row; buffers will \
                             still be returned and rounds will still close, but no shares come \
                             from this worker any more",
                            failures
                        );
                    } else if info.finished {
                        report_drive_finished(&tx_nonce_data, &info);
                    }
                }
            }

            let _ = tx_empty_buffers.send(buffer);
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

        let result = gpu_hash(&context, len as usize, &buffer_gpu);
        assert_eq!(18043101931632730606u64, result.0);
    }
}
