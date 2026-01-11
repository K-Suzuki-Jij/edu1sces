use std::env;
use std::time::Instant;

use edu1sces::blas::{ConjugateGradientParameters, InverseIterationParameters};
use edu1sces::examples_util::{build_heisenberg_chain, build_hubbard_chain};
use edu1sces::solver::{solve_heisenberg, solve_hubbard, SolverParameters};

fn main() {
    // Read model type from command line argument
    let args: Vec<String> = env::args().collect();
    let model_type = if args.len() > 1 {
        args[1].as_str()
    } else {
        "Heisenberg" // Default
    };

    // ========== Configuration ==========
    // Common parameters
    let num_iterations = 5;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

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

    println!("=== Solver Benchmark ===");
    println!("Model type: {}\n", model_type);

    match model_type {
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

            println!("--- solve_heisenberg ---");
            let mut baseline_time_ms = None;
            for &threads in &thread_counts {
                let params = SolverParameters {
                    eigenvalue_tol: 1e-14,
                    min_step: 5,
                    max_step: 1000,
                    num_threads: threads,
                    inverse_iteration_params: InverseIterationParameters {
                        diag_add: 10.0,
                        eigenvec_tol: 1e-8,
                        max_step: 100,
                        cg_params: ConjugateGradientParameters {
                            residual_tol: 1e-12,
                            max_step: 1000,
                        },
                    },
                };

                // Warmup
                for _ in 0..1 {
                    let _ = solve_heisenberg(&model, heisenberg_total_sz, &params).unwrap();
                }

                let t0 = Instant::now();
                let mut final_energy = 0.0;
                for _ in 0..num_iterations {
                    let result = solve_heisenberg(&model, heisenberg_total_sz, &params).unwrap();
                    final_energy = result.energy;
                }
                let dt = t0.elapsed();
                let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

                if let Some(base) = baseline_time_ms {
                    let speedup = base / avg_time_ms;
                    println!(
                        "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}",
                        threads, avg_time_ms, speedup, final_energy
                    );
                } else {
                    println!(
                        "  threads={:2}, avg_time={:8.2}ms, energy={:.15}",
                        threads, avg_time_ms, final_energy
                    );
                    baseline_time_ms = Some(avg_time_ms);
                }
            }
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

            println!("--- solve_hubbard ---");
            let mut baseline_time_ms = None;
            for &threads in &thread_counts {
                let params = SolverParameters {
                    eigenvalue_tol: 1e-14,
                    min_step: 5,
                    max_step: 1000,
                    num_threads: threads,
                    inverse_iteration_params: InverseIterationParameters {
                        diag_add: 10.0,
                        eigenvec_tol: 1e-8,
                        max_step: 100,
                        cg_params: ConjugateGradientParameters {
                            residual_tol: 1e-12,
                            max_step: 1000,
                        },
                    },
                };

                // Warmup
                for _ in 0..1 {
                    let _ = solve_hubbard(&model, hubbard_num_electrons, hubbard_total_sz, &params)
                        .unwrap();
                }

                let t0 = Instant::now();
                let mut final_energy = 0.0;
                for _ in 0..num_iterations {
                    let result =
                        solve_hubbard(&model, hubbard_num_electrons, hubbard_total_sz, &params)
                            .unwrap();
                    final_energy = result.energy;
                }
                let dt = t0.elapsed();
                let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

                if let Some(base) = baseline_time_ms {
                    let speedup = base / avg_time_ms;
                    println!(
                        "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}",
                        threads, avg_time_ms, speedup, final_energy
                    );
                } else {
                    println!(
                        "  threads={:2}, avg_time={:8.2}ms, energy={:.15}",
                        threads, avg_time_ms, final_energy
                    );
                    baseline_time_ms = Some(avg_time_ms);
                }
            }
        }
        _ => {
            eprintln!("Error: Unknown model type '{}'", model_type);
            eprintln!("Usage: {} [Heisenberg|Hubbard]", args[0]);
            std::process::exit(1);
        }
    }
}
