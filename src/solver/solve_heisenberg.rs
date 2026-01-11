use anyhow::Result;

use crate::basis::HeisenbergBasis;
use crate::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use crate::model::HeisenbergModel;
use crate::solver::solver_core::solve_with_basis_and_hamiltonian;
use crate::solver::{SolverParameters, SolverResult};

/// Solve the Heisenberg model to find the ground state energy and eigenvector.
pub fn solve_heisenberg(
    model: &HeisenbergModel,
    target_total_sz: f64,
    params: &SolverParameters,
) -> Result<SolverResult> {
    let num_threads = params.num_threads;
    solve_with_basis_and_hamiltonian(
        || HeisenbergBasis::new(model.clone(), target_total_sz),
        |basis| make_heisenberg_hamiltonian(basis, model, num_threads),
        params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::{ConjugateGradientParameters, InverseIterationParameters};
    use std::collections::HashMap;

    const TOL: f64 = 1e-8;

    fn make_solver_params() -> SolverParameters {
        SolverParameters {
            eigenvalue_tol: 1e-10,
            min_step: 5,
            max_step: 1000,
            num_threads: 1,
            inverse_iteration_params: InverseIterationParameters {
                diag_add: 10.0,
                eigenvec_tol: 1e-8,
                max_step: 100,
                cg_params: ConjugateGradientParameters {
                    residual_tol: 1e-12,
                    max_step: 1000,
                },
            },
        }
    }

    #[test]
    fn test_solve_heisenberg_2site_spin_half() {
        // 2-site S=1/2 Heisenberg chain with periodic boundary conditions
        // H = J * S1 · S2 = J * (Sz1*Sz2 + (1/2)(S1+S2- + S1-S2+))
        // In Sz=0 sector, basis: |↑↓⟩, |↓↑⟩
        // Hamiltonian matrix:
        //      |↑↓⟩   |↓↑⟩
        // |↑↓⟩  -1/4   1/2
        // |↓↑⟩   1/2  -1/4
        // Eigenvalues: -3/4 (singlet), 1/4 (triplet)
        // Ground state energy: E = -3/4 = -0.75 for J=1

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);
        exchange_z.insert((0, 1), 1.0);

        let model = HeisenbergModel::new(
            vec![0.5, 0.5],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            exchange_xy,
            exchange_z,
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_heisenberg(&model, 0.0, &params).unwrap(); // Sz=0 sector

        assert_eq!(result.dim, 2); // |↑↓⟩, |↓↑⟩
        assert!(
            (result.energy - (-0.75)).abs() < TOL,
            "Expected energy -0.75, got {}",
            result.energy
        );

        // Eigenvector should be singlet: (|↑↓⟩ - |↓↑⟩)/√2 or its negative
        // |c0|^2 = |c1|^2 = 0.5, and c0 * c1 < 0
        let c0 = result.eigenvector[0];
        let c1 = result.eigenvector[1];
        assert!(
            (c0.abs() - 1.0 / 2.0_f64.sqrt()).abs() < TOL,
            "Expected |c0| = 1/√2, got {}",
            c0.abs()
        );
        assert!(
            (c1.abs() - 1.0 / 2.0_f64.sqrt()).abs() < TOL,
            "Expected |c1| = 1/√2, got {}",
            c1.abs()
        );
        assert!(
            c0 * c1 < 0.0,
            "Expected c0*c1 < 0 for singlet, got c0={}, c1={}",
            c0,
            c1
        );
    }

    #[test]
    fn test_solve_heisenberg_2site_spin_one() {
        // 2-site S=1 Heisenberg chain with periodic boundary conditions
        // H = J * S1 · S2
        // In Sz=0 sector, basis: |+1,-1⟩, |0,0⟩, |-1,+1⟩ (dim=3)
        // Ground state energy: E = -2J = -2.0 for J=1

        let mut exchange_xy = HashMap::new();
        let mut exchange_z = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);
        exchange_z.insert((0, 1), 1.0);

        let model = HeisenbergModel::new(
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            exchange_xy,
            exchange_z,
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_heisenberg(&model, 0.0, &params).unwrap(); // Sz=0 sector

        assert_eq!(result.dim, 3); // |+1,-1⟩, |0,0⟩, |-1,+1⟩
        assert!(
            (result.energy - (-2.0)).abs() < TOL,
            "Expected energy -2.0, got {}",
            result.energy
        );
    }
}
