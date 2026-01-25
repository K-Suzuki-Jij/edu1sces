use std::time::Instant;

use edu1sces::blas::{lanczos, CsrMatrix, LanczosParameters};
use edu1sces::examples_util::{build_heisenberg_chain, build_hubbard_chain};
use edu1sces::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use edu1sces::hamiltonian::hubbard_hamiltonian::make_hubbard_hamiltonian;
use edu1sces::model::QuantumModel;

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
        output_log: false,
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
            &[],
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
            &[],
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
    // ========== Configuration ==========
    let model_type = "Heisenberg"; // "Heisenberg" or "Hubbard"

    // Common parameters
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];
    let num_threads_for_build = 6;

    // Heisenberg parameters
    let heisenberg_n = 22;
    let heisenberg_two_s = 1; // S = 1/2
    let heisenberg_total_sz = 0.0;
    let heisenberg_jxy = 1.0;
    let heisenberg_jz = 1.0;
    let heisenberg_hz = 0.3;
    let heisenberg_d = 0.2;

    // Hubbard parameters
    let hubbard_n = 12;
    let hubbard_num_electrons = 10;
    let hubbard_total_sz = 0.0;
    let hubbard_t = 1.0;
    let hubbard_u = 4.0;
    let hubbard_mu = 0.0;
    let hubbard_hz = 0.0;
    let hubbard_v = 0.5; // density-density interaction
    let hubbard_jxy = 1.0; // exchange xy
    let hubbard_jz = 1.0; // exchange z
                          // ===================================

    println!("=== Lanczos Benchmark ===");
    println!("Model type: {}\n", model_type);

    let h = match model_type {
        "Heisenberg" => {
            let model = build_heisenberg_chain(
                heisenberg_two_s,
                heisenberg_n,
                heisenberg_jxy,
                heisenberg_jz,
                heisenberg_hz,
                heisenberg_d,
            );

            println!(
                "Heisenberg model: n={}, two_s={}, total_sz={}",
                heisenberg_n, heisenberg_two_s, heisenberg_total_sz
            );
            println!(
                "  jxy={}, jz={}, hz={}, d={}\n",
                heisenberg_jxy, heisenberg_jz, heisenberg_hz, heisenberg_d
            );

            println!("Building basis...");
            let t0 = Instant::now();
            // target_quantum_numbers = [2*Sz]
            let total_sz2 = (2.0_f64 * heisenberg_total_sz).round() as i32;
            let basis = model.build_basis(&[total_sz2]).unwrap();
            let dt = t0.elapsed();
            println!("  dim={}, time={:?}", basis.dim(), dt);

            println!("Building Hamiltonian...");
            let t0 = Instant::now();
            let h = make_heisenberg_hamiltonian(&basis, &model, num_threads_for_build).unwrap();
            let dt = t0.elapsed();
            println!("  nnz={}, time={:?}\n", h.nnz(), dt);

            h
        }
        "Hubbard" => {
            let model = build_hubbard_chain(
                hubbard_n,
                hubbard_t,
                hubbard_u,
                hubbard_mu,
                hubbard_hz,
                hubbard_v,
                hubbard_jxy,
                hubbard_jz,
            );

            println!(
                "Hubbard model: n={}, num_electrons={}, total_sz={}",
                hubbard_n, hubbard_num_electrons, hubbard_total_sz
            );
            println!(
                "  t={}, U={}, mu={}, hz={}",
                hubbard_t, hubbard_u, hubbard_mu, hubbard_hz
            );
            println!(
                "  V={}, Jxy={}, Jz={}\n",
                hubbard_v, hubbard_jxy, hubbard_jz
            );

            println!("Building basis...");
            let t0 = Instant::now();
            // target_quantum_numbers = [N, 2*Sz]
            let total_sz2 = (2.0_f64 * hubbard_total_sz).round() as i32;
            let basis = model
                .build_basis(&[hubbard_num_electrons as i32, total_sz2])
                .unwrap();
            let dt = t0.elapsed();
            println!("  dim={}, time={:?}", basis.dim(), dt);

            println!("Building Hamiltonian...");
            let t0 = Instant::now();
            let h = make_hubbard_hamiltonian(&basis, &model, num_threads_for_build).unwrap();
            let dt = t0.elapsed();
            println!("  nnz={}, time={:?}\n", h.nnz(), dt);

            h
        }
        _ => panic!("Invalid model type: {}", model_type),
    };

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
