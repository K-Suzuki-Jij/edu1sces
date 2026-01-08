use anyhow::{bail, Result};
use rayon::ThreadPool;

pub fn build_pool(num_threads: usize) -> Result<ThreadPool> {
    if num_threads == 0 {
        bail!("num_threads must be positive");
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build rayon thread pool: {e}"))
}

pub fn with_pool<R, F>(num_threads: usize, f: F) -> Result<R>
where
    R: Send,
    F: FnOnce() -> Result<R> + Send,
{
    let pool = build_pool(num_threads)?;
    pool.install(f)
}
