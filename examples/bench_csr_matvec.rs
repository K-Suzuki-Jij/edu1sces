use std::time::Instant;

use edu1sces::blas::matrix_vector_operation::{csr_matvec, csr_sym_lower_matvec};
use edu1sces::blas::CsrMatrix;
use edu1sces::utility::rayon_pool::build_pool;
use rand::Rng;

/// Generate a random symmetric sparse matrix in CSR format (lower triangle only).
fn generate_symmetric_lower_csr(n: usize, density: f64) -> CsrMatrix {
    let mut rng = rand::rng();

    // Lower triangle has n*(n+1)/2 elements
    let total_elements = n * (n + 1) / 2;
    let expected_nnz = (total_elements as f64 * density) as usize;

    // Sample random indices and values
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(expected_nnz);
    for _ in 0..expected_nnz {
        // Sample a random index in lower triangle [0, total_elements)
        let idx = rng.random_range(0..total_elements);
        // Convert linear index to (i, j) where i >= j
        // idx = i*(i+1)/2 + j, solve for i: i = floor((sqrt(8*idx+1)-1)/2)
        let i = (((8 * idx + 1) as f64).sqrt() as usize - 1) / 2;
        let j = idx - i * (i + 1) / 2;
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

/// Expand a lower triangle CSR matrix to full symmetric matrix.
fn expand_lower_to_full(lower: &CsrMatrix) -> CsrMatrix {
    let n = lower.row_dim;

    // Collect all entries and add upper triangle
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(lower.nnz() * 2);
    for i in 0..n {
        for k in lower.rows[i]..lower.rows[i + 1] {
            let j = lower.cols[k];
            let val = lower.vals[k];
            entries.push((i, j, val));
            if i != j {
                entries.push((j, i, val));
            }
        }
    }

    // Sort by (row, col)
    entries.sort_by_key(|&(i, j, _)| (i, j));

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

fn benchmark_lower(m: &CsrMatrix, x: &[f64], num_threads: usize, num_iterations: usize) {
    let n = m.row_dim;
    let pool = build_pool(num_threads).unwrap();
    let mut y = vec![0.0; n];
    let mut y_locals: Vec<Vec<f64>> = (0..num_threads).map(|_| vec![0.0; n]).collect();
    let shift = 0.0;

    // Warmup
    for _ in 0..3 {
        csr_sym_lower_matvec(&pool, m, x, shift, &mut y, &mut y_locals).unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..num_iterations {
        csr_sym_lower_matvec(&pool, m, x, shift, &mut y, &mut y_locals).unwrap();
    }
    let dt = t0.elapsed();

    let nnz = m.nnz();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    let y_sum: f64 = y.iter().sum();
    println!(
        "  threads={:2}, nnz={:12}, avg_time={:8.2}ms, y_sum={:.10e}",
        num_threads, nnz, avg_time_ms, y_sum
    );

    std::hint::black_box(y);
}

fn benchmark_full(m: &CsrMatrix, x: &[f64], num_threads: usize, num_iterations: usize) {
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

    let nnz = m.nnz();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    let y_sum: f64 = y.iter().sum();
    println!(
        "  threads={:2}, nnz={:12}, avg_time={:8.2}ms, y_sum={:.10e}",
        num_threads, nnz, avg_time_ms, y_sum
    );

    std::hint::black_box(y);
}

fn main() {
    // Parameters
    let n = 1000000;
    let density = 0.00001;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    println!("=== CSR Matrix-Vector Multiplication Benchmark ===");
    println!("n={}, density={}\n", n, density);

    // Lower triangle matrix
    println!("Generating lower triangle matrix...");
    let m_lower = generate_symmetric_lower_csr(n, density);
    println!("  nnz={}", m_lower.nnz());

    // Full matrix (from lower)
    println!("Expanding to full matrix...");
    let m_full = expand_lower_to_full(&m_lower);
    println!("  nnz={}", m_full.nnz());

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    println!("\n--- Lower triangle only (csr_sym_lower_matvec) ---");
    for &threads in &thread_counts {
        benchmark_lower(&m_lower, &x, threads, num_iterations);
    }

    println!("\n--- Full matrix (csr_matvec) ---");
    for &threads in &thread_counts {
        benchmark_full(&m_full, &x, threads, num_iterations);
    }
}
