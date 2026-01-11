use std::time::Instant;

use edu1sces::basis::{HeisenbergBasis, HilbertBasis};
use edu1sces::examples_util::build_heisenberg_chain;
use edu1sces::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use edu1sces::model::HeisenbergModel;

fn benchmark(
    basis: &HeisenbergBasis,
    model: &HeisenbergModel,
    num_threads: usize,
    num_iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..2 {
        let _ = make_heisenberg_hamiltonian(basis, model, num_threads).unwrap();
    }

    let t0 = Instant::now();
    let mut h = None;
    for _ in 0..num_iterations {
        h = Some(make_heisenberg_hamiltonian(basis, model, num_threads).unwrap());
    }
    let dt = t0.elapsed();

    let h = h.unwrap();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(h);
    avg_time_ms
}

fn main() {
    let n = 10;
    let two_s = 3; // S = 3/2
    let total_sz = 0.0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    let model = build_heisenberg_chain(two_s, n, 1.0, 1.0, 0.3, 0.2);

    println!("=== make_heisenberg_hamiltonian Benchmark ===");
    println!("n={}, two_s={}, total_sz={}\n", n, two_s, total_sz);

    println!("Building basis...");
    let t0 = Instant::now();
    let basis = HeisenbergBasis::new(model.clone(), total_sz).unwrap();
    let dt = t0.elapsed();
    println!("  dim={}, time={:?}", basis.dim(), dt);

    // Build once to get nnz
    let h = make_heisenberg_hamiltonian(&basis, &model, 1).unwrap();
    println!("  nnz={}\n", h.nnz());

    println!("--- make_heisenberg_hamiltonian ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&basis, &model, threads, num_iterations);
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
