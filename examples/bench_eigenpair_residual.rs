use std::time::Instant;

use edu1sces::blas::matrix_vector_operation::eigenpair_residual_norm1;
use edu1sces::blas::CsrMatrix;
use edu1sces::utility::rayon_pool::build_pool;
use rand::Rng;

/// Generate a random sparse matrix in CSR format.
fn generate_sparse_csr(n: usize, density: f64) -> CsrMatrix {
    let mut rng = rand::rng();

    let expected_nnz = (n as f64 * n as f64 * density) as usize;

    // Sample random indices and values
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(expected_nnz);
    for _ in 0..expected_nnz {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let val = rng.random_range(-1.0..1.0);
        entries.push((i, j, val));
    }

    // Sort by (row, col) and remove duplicates
    entries.sort_by_key(|&(i, j, _)| (i, j));
    entries.dedup_by_key(|e| (e.0, e.1));

    // Build CSR
    let mut rows = vec![0usize; n + 1];
    let mut cols = Vec::with_capacity(entries.len());
    let mut vals = Vec::with_capacity(entries.len());

    for &(i, j, val) in &entries {
        rows[i + 1] += 1;
        cols.push(j);
        vals.push(val);
    }

    // Cumulative sum
    for i in 0..n {
        rows[i + 1] += rows[i];
    }

    CsrMatrix {
        row_dim: n,
        col_dim: n,
        rows,
        cols,
        vals,
    }
}

fn benchmark(
    m: &CsrMatrix,
    x: &[f64],
    lambda: f64,
    num_threads: usize,
    num_iterations: usize,
) -> f64 {
    let pool = build_pool(num_threads).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = eigenpair_residual_norm1(&pool, m, x, lambda).unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..num_iterations {
        eigenpair_residual_norm1(&pool, m, x, lambda).unwrap();
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    avg_time_ms
}

fn main() {
    let n = 1000000;
    let density = 0.0001;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    println!("=== eigenpair_residual_norm1 Benchmark ===");
    println!("n={}, density={}\n", n, density);

    println!("Generating sparse matrix...");
    let m = generate_sparse_csr(n, density);
    println!("  nnz={}", m.nnz());

    println!("Generating random eigenvector...");
    let mut rng = rand::rng();
    let x: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let lambda = 2.5;

    println!("\n--- eigenpair_residual_norm1 (||A*x - lambda*x||_1) ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&m, &x, lambda, threads, num_iterations);
        if let Some(base) = baseline_time_ms {
            let speedup = base / time;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x",
                threads, time, speedup
            );
        } else {
            println!("  threads={:2}, avg_time={:8.2}ms", threads, time);
            baseline_time_ms = Some(time);
        }
    }
}
