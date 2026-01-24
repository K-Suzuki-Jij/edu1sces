use std::time::Instant;

use edu1sces::basis::Basis;
use edu1sces::examples_util::build_heisenberg_chain;
use edu1sces::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use edu1sces::model::{HeisenbergModel, QuantumModel};

fn benchmark(
    basis: &Basis,
    model: &HeisenbergModel,
    num_threads: usize,
    num_iterations: usize,
) -> (f64, usize) {
    let t0 = Instant::now();
    let mut h = None;
    for _ in 0..num_iterations {
        h = Some(make_heisenberg_hamiltonian(basis, model, num_threads).unwrap());
    }
    let dt = t0.elapsed();

    let h = h.unwrap();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;
    let nnz = h.nnz();

    std::hint::black_box(h);
    (avg_time_ms, nnz)
}

fn main() {
    let n = 22;
    let two_s = 1; // S = 3/2
    let total_sz = 0.0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    let model = build_heisenberg_chain(two_s, n, 1.0, 1.0, 0.0, 0.0);

    println!("=== make_heisenberg_hamiltonian Benchmark ===");
    println!("n={}, two_s={}, total_sz={}\n", n, two_s, total_sz);

    println!("Building basis...");
    let t0 = Instant::now();
    // target_quantum_numbers = [2*Sz]
    let total_sz2 = (2.0_f64 * total_sz).round() as i32;
    let basis = model.build_basis(&[total_sz2]).unwrap();
    let dt = t0.elapsed();
    println!("  dim={}, time={:?}\n", basis.dim(), dt);

    println!("--- make_heisenberg_hamiltonian ---");
    let mut baseline_time_ms = None;
    let mut last_nnz = 0;
    for &threads in &thread_counts {
        let (time, nnz) = benchmark(&basis, &model, threads, num_iterations);
        last_nnz = nnz;
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
    println!("\n  nnz={}", last_nnz);
}
