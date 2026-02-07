use std::collections::HashMap;
use std::time::Instant;

use edu1sces::blas::{ConjugateGradientParameters, InverseIterationParameters};
use edu1sces::hamiltonian::su2_heisenberg_hamiltonian::make_su2_heisenberg_hamiltonian;
use edu1sces::model::operator::SpinOperator;
use edu1sces::model::SU2HeisenbergModel;
use edu1sces::solver::{solve_su2_heisenberg, SolverParameters};

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
    let n = 14;
    let thread_counts = [1, 2, 3, 4, 5, 6];
    let spin = 0.5;
    let total_s = 0.0;
    let m_total = 0.0;
    let j = 1.0;
    // ===================================

    let model = build_su2_heisenberg_chain(n, spin, j);
    let dim = model.calc_dim_su2_sector(total_s).unwrap();

    println!("=== SU(2) Correlation Function Benchmark ===");
    println!("N={}, dim={}\n", n, dim);

    // Build basis
    let basis = model.build_basis(total_s).unwrap();

    // Measure Hamiltonian construction time
    println!("--- Hamiltonian construction ---");
    let mut baseline_ham_time = None;
    for &threads in &thread_counts {
        let t_ham = Instant::now();
        let _h = make_su2_heisenberg_hamiltonian(&basis, &model, threads).unwrap();
        let ham_time = t_ham.elapsed().as_secs_f64();

        if let Some(base) = baseline_ham_time {
            let speedup = base / ham_time;
            println!("threads={}: {:.3}s, speedup={:.2}x", threads, ham_time, speedup);
        } else {
            println!("threads={}: {:.3}s", threads, ham_time);
            baseline_ham_time = Some(ham_time);
        }
    }

    // Solve once
    println!("\n--- Solving ---");
    let params = make_solver_params(1);
    let mut result = solve_su2_heisenberg(&model, total_s, &params).unwrap();
    println!("E={:.10}\n", result.energies[0]);

    // Measure correlation function time for different site pairs
    for (site1, site2, label) in [
        (0, 1, "<Sz_0 Sz_1> (site 0: pos 0, N-1 swaps)"),
        (n-2, n-1, &format!("<Sz_{} Sz_{}> (site N-1: pos N-1, 0 swaps)", n-2, n-1)),
    ] {
        println!("--- Correlation function {} ---", label);
        let mut baseline_corr_time = None;
        for &threads in &thread_counts {
            let t_corr = Instant::now();
            let corr = result
                .correlation_function(SpinOperator::Sz, site1, SpinOperator::Sz, site2, m_total, threads, 0)
                .unwrap();
            let corr_time = t_corr.elapsed().as_secs_f64();

            if let Some(base) = baseline_corr_time {
                let speedup = base / corr_time;
                println!("threads={}: {:.3}s, speedup={:.2}x, <SzSz>={:.10}", threads, corr_time, speedup, corr);
            } else {
                println!("threads={}: {:.3}s, <SzSz>={:.10}", threads, corr_time, corr);
                baseline_corr_time = Some(corr_time);
            }
        }
        println!();
    }
}
