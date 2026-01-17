use std::collections::HashMap;
use std::time::Instant;

use edu1sces::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use edu1sces::model::SU2HeisenbergModel;

fn build_su2_heisenberg_chain(two_s: i32, n: usize, j: f64) -> SU2HeisenbergModel {
    let spin_list: Vec<f64> = vec![two_s as f64 / 2.0; n];
    let mut exchange = HashMap::new();
    for i in 0..n - 1 {
        exchange.insert((i, i + 1), j);
    }
    SU2HeisenbergModel::new(spin_list, exchange).unwrap()
}

fn benchmark(
    model: &SU2HeisenbergModel,
    total_s: f64,
    num_threads: usize,
    num_iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..2 {
        let _ = make_su2_heisenberg_hamiltonian(model, total_s, num_threads).unwrap();
    }

    let t0 = Instant::now();
    let mut h = None;
    for _ in 0..num_iterations {
        h = Some(make_su2_heisenberg_hamiltonian(model, total_s, num_threads).unwrap());
    }
    let dt = t0.elapsed();

    let h = h.unwrap();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(h);
    avg_time_ms
}

fn main() {
    let n = 18;
    let two_s = 1; // S = 1/2
    let total_s = 0.0;
    let num_iterations = 5;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    let model = build_su2_heisenberg_chain(two_s, n, 1.0);

    println!("=== make_su2_heisenberg_hamiltonian Benchmark ===");
    println!("n={}, two_s={}, total_s={}\n", n, two_s, total_s);

    println!("Building basis...");
    let t0 = Instant::now();
    let (basis, _) = model.build_basis(total_s).unwrap();
    let dt = t0.elapsed();
    println!("  dim={}, time={:?}", basis.len(), dt);

    // Build once to get nnz
    let h = make_su2_heisenberg_hamiltonian(&model, total_s, 1).unwrap();
    println!("  nnz={}\n", h.nnz());

    println!("--- make_su2_heisenberg_hamiltonian ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&model, total_s, threads, num_iterations);
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
