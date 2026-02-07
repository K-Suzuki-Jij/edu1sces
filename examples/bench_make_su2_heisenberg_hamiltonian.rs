use std::collections::HashMap;
use std::time::Instant;

use edu1sces::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use edu1sces::model::SU2HeisenbergModel;

fn benchmark(
    basis: &edu1sces::basis::SU2HeisenbergBasis,
    model: &SU2HeisenbergModel,
    num_threads: usize,
    num_iterations: usize,
) -> (f64, usize) {
    let t0 = Instant::now();
    let mut h = None;
    for _ in 0..num_iterations {
        h = Some(make_su2_heisenberg_hamiltonian(basis, model, num_threads).unwrap());
    }
    let dt = t0.elapsed();

    let h = h.unwrap();
    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;
    let nnz = h.nnz();

    std::hint::black_box(h);
    (avg_time_ms, nnz)
}

/// Generate 1D nearest-neighbor bonds: 0-1-2-...(n-1)
fn generate_1d_chain_bonds(n: usize) -> HashMap<(usize, usize), f64> {
    let mut exchange = HashMap::new();
    for i in 0..n - 1 {
        exchange.insert((i, i + 1), 1.0);
    }
    exchange
}

fn main() {
    let n = 20;
    let spin = 0.5;
    let total_s = 0.0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    // 1D nearest-neighbor chain
    let exchange = generate_1d_chain_bonds(n);

    let model = SU2HeisenbergModel::new(vec![spin; n], exchange).unwrap();

    println!("=== make_su2_heisenberg_hamiltonian Benchmark ===");
    println!(
        "n={}, spin={}, total_s={}, 1D chain (nearest-neighbor)\n",
        n, spin, total_s
    );

    // Build basis (automatically optimizes coupling order)
    println!("Building basis (with auto-optimized coupling order)...");
    let t0 = Instant::now();
    let basis = model.build_basis(total_s).unwrap();
    let dt = t0.elapsed();
    println!("  dim={}, time={:?}", basis.dim(), dt);
    println!("  coupling_order: {:?}\n", basis.coupling_order);

    println!("--- make_su2_heisenberg_hamiltonian ---");
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
