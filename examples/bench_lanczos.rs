use std::collections::HashMap;
use std::time::Instant;

use edu1sces::basis::{HeisenbergBasis, HilbertBasis};
use edu1sces::blas::{lanczos, CsrMatrix, LanczosParameters};
use edu1sces::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use edu1sces::model::HeisenbergModel;

fn build_chain_model(two_s: i32, n: usize, jxy: f64, jz: f64, hz: f64, d: f64) -> HeisenbergModel {
    let mut exchange_xy = HashMap::new();
    let mut exchange_z = HashMap::new();

    for i in 0..n - 1 {
        exchange_xy.insert((i, i + 1), jxy);
        exchange_z.insert((i, i + 1), jz);
    }

    HeisenbergModel {
        num_sites: n,
        two_s_list: vec![two_s; n],
        hz_list: vec![hz; n],
        d_list: vec![d; n],
        exchange_xy,
        exchange_z,
    }
}

fn benchmark_lanczos(
    hamiltonian: &CsrMatrix,
    num_threads: usize,
    num_iterations: usize,
) -> (f64, f64) {
    let lanczos_params = LanczosParameters {
        acc: 1e-14,
        min_step: 5,
        max_step: 1000,
        calc_eigenvec: true,
    };

    // Warmup
    for _ in 0..2 {
        let mut eigenvector = Vec::new();
        let mut energy = 0.0;
        let _ = lanczos(
            hamiltonian,
            &mut eigenvector,
            &mut energy,
            &lanczos_params,
            num_threads,
        );
    }

    let t0 = Instant::now();
    let mut final_energy = 0.0;
    for _ in 0..num_iterations {
        let mut eigenvector = Vec::new();
        let mut energy = 0.0;
        lanczos(
            hamiltonian,
            &mut eigenvector,
            &mut energy,
            &lanczos_params,
            num_threads,
        )
        .unwrap();
        final_energy = energy;
        std::hint::black_box(&eigenvector);
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;
    (avg_time_ms, final_energy)
}

fn main() {
    let n = 24;
    let two_s = 1; // S = 3/2
    let total_sz = 0.0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    let model = build_chain_model(two_s, n, 1.0, 1.0, 0.3, 0.2);

    println!("=== Lanczos Benchmark ===");
    println!("n={}, two_s={}, total_sz={}\n", n, two_s, total_sz);

    println!("Building basis...");
    let t0 = Instant::now();
    let basis = HeisenbergBasis::new(model.clone(), total_sz).unwrap();
    let dt = t0.elapsed();
    println!("  dim={}, time={:?}", basis.dim(), dt);

    println!("Building Hamiltonian...");
    let t0 = Instant::now();
    let h = make_heisenberg_hamiltonian(&basis, &model, 6).unwrap();
    let dt = t0.elapsed();
    println!("  nnz={}, time={:?}\n", h.nnz(), dt);

    println!("--- Lanczos ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let (time, energy) = benchmark_lanczos(&h, threads, num_iterations);
        if let Some(base) = baseline_time_ms {
            let speedup = base / time;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}",
                threads, time, speedup, energy
            );
        } else {
            println!(
                "  threads={:2}, avg_time={:8.2}ms, energy={:.15}",
                threads, time, energy
            );
            baseline_time_ms = Some(time);
        }
    }
}
