use std::collections::HashMap;
use std::time::Instant;

use edu1sces::blas::{ConjugateGradientParameters, InverseIterationParameters};
use edu1sces::model::su2_heisenberg::SU2HeisenbergModel;
use edu1sces::solver::{solve_su2_heisenberg, SU2SolverResult, SolverParameters};

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
    }
}

fn build_su2_heisenberg_chain(num_sites: usize, two_s: i32, j: f64) -> SU2HeisenbergModel {
    let spin = (two_s as f64) / 2.0;
    let spin_list = vec![spin; num_sites];

    let mut exchange = HashMap::new();
    for i in 0..num_sites - 1 {
        exchange.insert((i, i + 1), j);
    }

    SU2HeisenbergModel::new(spin_list, exchange).unwrap()
}

fn run_benchmark<F>(thread_counts: &[usize], num_iterations: usize, mut solve: F)
where
    F: FnMut(&SolverParameters) -> SU2SolverResult,
{
    let mut baseline_time_ms = None;
    for &threads in thread_counts {
        let params = make_solver_params(threads);

        let t0 = Instant::now();
        let mut result: Option<SU2SolverResult> = None;
        for _ in 0..num_iterations {
            result = Some(solve(&params));
        }
        let dt = t0.elapsed();
        let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

        let result = result.unwrap();
        let final_residual = result
            .inverse_iteration_log
            .residual_errors
            .last()
            .copied()
            .unwrap_or(result.inverse_iteration_log.initial_residual_error);

        if let Some(base) = baseline_time_ms {
            let speedup = base / avg_time_ms;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}, residual={:.2e}",
                threads, avg_time_ms, speedup, result.energy, final_residual
            );
        } else {
            println!(
                "  threads={:2}, avg_time={:8.2}ms, energy={:.15}, residual={:.2e}",
                threads, avg_time_ms, result.energy, final_residual
            );
            baseline_time_ms = Some(avg_time_ms);
        }
    }
}

fn main() {
    // ========== Configuration ==========
    let num_iterations = 1;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    // SU(2) Heisenberg parameters
    let num_sites = 26;
    let two_s = 1; // S = 1/2
    let total_s = 0.0; // Target total spin
    let j = 1.0; // Exchange coupling
    // ===================================

    let model = build_su2_heisenberg_chain(num_sites, two_s, j);

    println!("=== SU(2) Heisenberg Solver Benchmark ===");
    println!(
        "SU(2) Heisenberg model: n={}, two_s={}, total_s={}",
        num_sites, two_s, total_s
    );
    println!("  J={}\n", j);

    println!("--- solve_su2_heisenberg ---");
    run_benchmark(&thread_counts, num_iterations, |params| {
        solve_su2_heisenberg(&model, total_s, params).unwrap()
    });
}
