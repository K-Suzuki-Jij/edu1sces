use std::collections::HashMap;
use std::time::Instant;

use edu1sces::blas::{ConjugateGradientParameters, InverseIterationParameters};
use edu1sces::model::SU2HeisenbergModel;
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
        num_states: 1,
    }
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

        if let Some(base) = baseline_time_ms {
            let speedup = base / avg_time_ms;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x, energy={:.15}",
                threads, avg_time_ms, speedup, result.energies[0]
            );
        } else {
            println!(
                "  threads={:2}, avg_time={:8.2}ms, energy={:.15}",
                threads, avg_time_ms, result.energies[0]
            );
            baseline_time_ms = Some(avg_time_ms);
        }
    }
}

fn build_su2_heisenberg_chain(n: usize, spin: f64, j: f64) -> SU2HeisenbergModel {
    let two_s_list = vec![spin; n];
    let mut exchange = HashMap::new();
    for i in 0..n - 1 {
        exchange.insert((i, i + 1), j);
    }
    SU2HeisenbergModel::new(two_s_list, exchange).unwrap()
}

fn main() {
    // ========== Configuration ==========
    let num_iterations = 1;
    let thread_counts = [1, 2, 3, 4];

    let n = 26;
    let spin = 0.5;
    let total_s = 0.0;
    let j = 1.0;
    // ===================================

    let model = build_su2_heisenberg_chain(n, spin, j);

    println!("=== SU(2) Heisenberg Solver Benchmark ===");
    println!("n={}, spin={}, total_s={}, j={}\n", n, spin, total_s, j);

    println!("--- solve_su2_heisenberg ---");
    run_benchmark(&thread_counts, num_iterations, |params| {
        solve_su2_heisenberg(&model, total_s, params).unwrap()
    });
}
