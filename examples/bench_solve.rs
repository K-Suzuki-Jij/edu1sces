use std::env;
use std::time::Instant;

use edu1sces::blas::{ConjugateGradientParameters, InverseIterationParameters};
use edu1sces::examples_util::{build_heisenberg_chain, build_hubbard_chain};
use edu1sces::solver::{solve_heisenberg, solve_hubbard, SolverParameters, SolverResult};

fn make_solver_params(num_threads: usize) -> SolverParameters {
    SolverParameters {
        eigenvalue_tol: 1e-14,
        min_step: 5,
        max_step: 1000,
        num_threads,
        inverse_iteration_params: InverseIterationParameters {
            diag_add: 1e-07,
            eigenvec_tol: 1e-8,
            max_step: 5,
            cg_params: ConjugateGradientParameters {
                residual_tol: 1e-12,
                max_step: 1000,
                output_log: false,
            },
        },
        output_log: false,
        num_states: 1,
    }
}

fn run_benchmark<F>(thread_counts: &[usize], num_iterations: usize, mut solve: F)
where
    F: FnMut(&SolverParameters) -> SolverResult,
{
    let mut baseline_time_ms = None;
    for &threads in thread_counts {
        let params = make_solver_params(threads);

        let t0 = Instant::now();
        let mut result: Option<SolverResult> = None;
        for _ in 0..num_iterations {
            result = Some(solve(&params));
        }
        let dt = t0.elapsed();
        let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

        let result = result.unwrap();
        let final_residual = result.inverse_iteration_logs[0]
            .residual_errors
            .last()
            .copied()
            .unwrap_or(result.inverse_iteration_logs[0].initial_residual_error);

        if let Some(base) = baseline_time_ms {
            let speedup = base / avg_time_ms;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}, residual={:.2e}",
                threads, avg_time_ms, speedup, result.energies[0], final_residual
            );
        } else {
            println!(
                "  threads={:2}, avg_time={:8.2}ms, energy={:.15}, residual={:.2e}",
                threads, avg_time_ms, result.energies[0], final_residual
            );
            baseline_time_ms = Some(avg_time_ms);
        }
    }
}

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
    let num_iterations = 1;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    // Heisenberg parameters
    let heisenberg_n = 26;
    let heisenberg_two_s = 1; // S = 1/2
    let heisenberg_total_sz = 0.0;
    let heisenberg_jxy = 1.0;
    let heisenberg_jz = 1.0;
    let heisenberg_hz = 0.0;
    let heisenberg_d = 0.0;

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
            run_benchmark(&thread_counts, num_iterations, |params| {
                solve_heisenberg(&model, heisenberg_total_sz, params).unwrap()
            });
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
            run_benchmark(&thread_counts, num_iterations, |params| {
                solve_hubbard(&model, hubbard_num_electrons, hubbard_total_sz, params).unwrap()
            });
        }
        _ => {
            eprintln!("Error: Unknown model type '{}'", model_type);
            eprintln!("Usage: {} [Heisenberg|Hubbard]", args[0]);
            std::process::exit(1);
        }
    }
}
