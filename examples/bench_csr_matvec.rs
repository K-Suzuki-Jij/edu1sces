use std::time::Instant;

use edu1sces::blas::matrix_vector_operation::csr_matvec;
use edu1sces::blas::CsrMatrix;
use edu1sces::examples_util::generate_sparse_csr;
use edu1sces::utility::rayon_pool::build_pool;

fn benchmark(m: &CsrMatrix, x: &[f64], num_threads: usize, num_iterations: usize) -> f64 {
    let n = m.row_dim;
    let pool = build_pool(num_threads).unwrap();
    let mut y = vec![0.0; n];
    let shift = 0.0;

    // Warmup
    for _ in 0..3 {
        csr_matvec(&pool, &mut y, m, x, shift).unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..num_iterations {
        csr_matvec(&pool, &mut y, m, x, shift).unwrap();
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(y);
    avg_time_ms
}

fn main() {
    // Parameters
    let n = 1000000;
    let density = 0.0001;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    println!("=== CSR Matrix-Vector Multiplication Benchmark ===");
    println!("n={}, density={}\n", n, density);

    println!("Generating sparse matrix...");
    let m = generate_sparse_csr(n, density);
    println!("  nnz={}", m.nnz());

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    println!("\n--- csr_matvec ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&m, &x, threads, num_iterations);
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
