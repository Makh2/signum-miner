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
                if info.finished {
                    report_drive_finished(&tx_nonce_data, &info);
                }
                let _ = tx_empty_buffers.send(buffer);
                continue;
            }

            // An OpenCL call that fails panics - ocl.rs unwraps its results -
            // and a panicking worker used to take the round down with it: the
            // buffer never came back, the drive task waiting for it never
            // reached the end of its plots, and the round could never close, so
            // the miner went silent until someone restarted it. Catch it here
            // instead: say what happened, hand the buffer back, carry on.
            let hashed = panic::catch_unwind(AssertUnwindSafe(|| {
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

            match hashed {
                Ok((deadline, offset)) => {
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
                Err(_) => {
                    failures += 1;
                    error!(
                        "GPU worker: hashing failed at height={}, window starting at nonce={} \
                         ({} in a row) -> dropping this window and returning the buffer",
                        info.height, info.start_nonce, failures
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
