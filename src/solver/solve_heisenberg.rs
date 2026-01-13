use anyhow::Result;
use pyo3::prelude::*;

use crate::basis::HeisenbergBasis;
use crate::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian;
use crate::model::HeisenbergModel;
use crate::solver::solver_core::solve_with_basis_and_hamiltonian;
use crate::solver::{CachedBasis, LocalStateLabels, SolverParameters, SolverResult};

/// Sector builder for Heisenberg model.
/// Builds basis for sectors with specified 2*Sz.
pub struct HeisenbergSectorBuilder {
    model: HeisenbergModel,
}

impl HeisenbergSectorBuilder {
    pub fn new(model: HeisenbergModel) -> Self {
        Self { model }
    }
}

impl LocalStateLabels for HeisenbergSectorBuilder {
    /// Return quantum numbers [2*sz] for a local state at a given site.
    /// For spin S, local states are 0..=2S corresponding to sz = S, S-1, ..., -S
    /// local_state=0 → sz=+S, local_state=2S → sz=-S
    fn quantum_numbers(&self, site: usize, local_state: usize) -> Vec<i32> {
        let two_s = self.model.two_s_list[site];
        // sz = S - local_state, so 2*sz = 2S - 2*local_state
        let sz2 = two_s - 2 * (local_state as i32);
        vec![sz2]
    }

    /// Build basis for sector with quantum numbers [2*total_sz].
    fn build_basis(&self, target_quantum_numbers: &[i32]) -> Result<CachedBasis> {
        let total_sz = target_quantum_numbers[0] as f64 / 2.0;

        let heisenberg_basis = HeisenbergBasis::new(self.model.clone(), total_sz)?;

        Ok(CachedBasis {
            dim: heisenberg_basis.basis.len(),
            basis: heisenberg_basis.basis,
            inverse_basis: heisenberg_basis.inverse_basis,
        })
    }
}

/// Solve the Heisenberg model to find the ground state energy and eigenvector.
#[pyfunction]
pub fn solve_heisenberg(
    model: &HeisenbergModel,
    target_total_sz: f64,
    params: &SolverParameters,
) -> Result<SolverResult> {
    let num_threads = params.num_threads;
    let model_clone = model.clone();

    // Calculate 2*Sz for quantum numbers
    let total_sz2 = (2.0 * target_total_sz).round() as i32;
    let current_quantum_numbers = vec![total_sz2];

    solve_with_basis_and_hamiltonian(
        || HeisenbergBasis::new(model.clone(), target_total_sz),
        |basis| make_heisenberg_hamiltonian(basis, model, num_threads),
        move || Box::new(HeisenbergSectorBuilder::new(model_clone)),
        current_quantum_numbers,
        params,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::{csr_mul, ConjugateGradientParameters, InverseIterationParameters};
    use std::collections::HashMap;

    const TOL: f64 = 1e-8;

    fn make_solver_params() -> SolverParameters {
        SolverParameters {
            eigenvalue_tol: 1e-14,
            min_step: 5,
            max_step: 1000,
            num_threads: 1,
            inverse_iteration_params: InverseIterationParameters {
                diag_add: 1e-7,
                eigenvec_tol: 1e-8,
                max_step: 100,
                cg_params: ConjugateGradientParameters {
                    residual_tol: 1e-12,
                    max_step: 1000,
                    output_log: false,
                },
            },
            output_log: false,
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
        let mut result = solve_heisenberg(&model, 0.0, &params).unwrap(); // Sz=0 sector

        assert_eq!(result.eigenvector.len(), 2); // |↑↓⟩, |↓↑⟩
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

        // Test expectation values: <Sz_i> = 0 in singlet state
        let sz_op = model.make_local_op_sz(0).unwrap();
        for site in 0..2 {
            let sz = result.expectation_onsite(&sz_op, site, 1).unwrap();
            assert!(sz.abs() < TOL, "Expected <Sz_{}> = 0, got {}", site, sz);
        }

        // Test <Sx_i> = 0 (Sx takes states outside Sz=0 sector)
        let sx_op = model.make_local_op_sx(0).unwrap();
        for site in 0..2 {
            let sx = result.expectation_onsite(&sx_op, site, 1).unwrap();
            assert!(sx.abs() < TOL, "Expected <Sx_{}> = 0, got {}", site, sx);
        }

        // Test <Sz_i^2> = 1/4 for S=1/2
        let szsz_op = csr_mul(1.0, &sz_op, 1.0, &sz_op).unwrap();
        for site in 0..2 {
            let szsz = result.expectation_onsite(&szsz_op, site, 1).unwrap();
            assert!(
                (szsz - 0.25).abs() < TOL,
                "Expected <Sz_{}^2> = 0.25, got {}",
                site,
                szsz
            );
        }

        // Test correlation functions
        // For singlet: <Sz_0 Sz_1> = -1/4
        let sz_corr = result
            .correlation_function(&sz_op, 0, &sz_op, 1, 1)
            .unwrap();
        assert!(
            (sz_corr - (-0.25)).abs() < TOL,
            "Expected <Sz_0 Sz_1> = -0.25, got {}",
            sz_corr
        );

        // For singlet: <S+_0 S-_1> = -1/2
        let sp_op = model.make_local_op_sp(0).unwrap();
        let sm_op = model.make_local_op_sm(0).unwrap();
        let sp_sm_corr = result
            .correlation_function(&sp_op, 0, &sm_op, 1, 1)
            .unwrap();
        assert!(
            (sp_sm_corr - (-0.5)).abs() < TOL,
            "Expected <S+_0 S-_1> = -0.5, got {}",
            sp_sm_corr
        );

        // For singlet: <S-_0 S+_1> = -1/2
        let sm_sp_corr = result
            .correlation_function(&sm_op, 0, &sp_op, 1, 1)
            .unwrap();
        assert!(
            (sm_sp_corr - (-0.5)).abs() < TOL,
            "Expected <S-_0 S+_1> = -0.5, got {}",
            sm_sp_corr
        );

        // Test Sx correlation: <Sx_0 Sx_1> = (1/4)(<S+_0 S-_1> + <S-_0 S+_1> + <S+_0 S+_1> + <S-_0 S-_1>)
        // = (1/4)(-0.5 + -0.5 + 0 + 0) = -0.25
        let sx_corr = result
            .correlation_function(&sx_op, 0, &sx_op, 1, 1)
            .unwrap();
        assert!(
            (sx_corr - (-0.25)).abs() < TOL,
            "Expected <Sx_0 Sx_1> = -0.25, got {}",
            sx_corr
        );

        // For isotropic Heisenberg model: <Sz_0 Sz_1> = <Sx_0 Sx_1>
        assert!(
            (sz_corr - sx_corr).abs() < TOL,
            "Expected <Sz_0 Sz_1> = <Sx_0 Sx_1>, got {} vs {}",
            sz_corr,
            sx_corr
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
        let mut result = solve_heisenberg(&model, 0.0, &params).unwrap(); // Sz=0 sector

        assert_eq!(result.eigenvector.len(), 3); // |+1,-1⟩, |0,0⟩, |-1,+1⟩
        assert!(
            (result.energy - (-2.0)).abs() < TOL,
            "Expected energy -2.0, got {}",
            result.energy
        );

        // Test expectation values: <Sz_i> = 0 in Sz=0 sector
        let sz_op = model.make_local_op_sz(0).unwrap();
        for site in 0..2 {
            let sz = result.expectation_onsite(&sz_op, site, 1).unwrap();
            assert!(sz.abs() < TOL, "Expected <Sz_{}> = 0, got {}", site, sz);
        }

        // Test <Sx_i> = 0 (Sx takes states outside Sz=0 sector)
        let sx_op = model.make_local_op_sx(0).unwrap();
        for site in 0..2 {
            let sx = result.expectation_onsite(&sx_op, site, 1).unwrap();
            assert!(sx.abs() < TOL, "Expected <Sx_{}> = 0, got {}", site, sx);
        }

        // Test <Sz_i^2> for S=1 singlet state
        // In the singlet (S_total=0) state: |singlet⟩ = (|+1,-1⟩ - |0,0⟩ + |-1,+1⟩)/√3
        // <Sz_i^2> = (1/3)(1 + 0 + 1) = 2/3
        let szsz_op = csr_mul(1.0, &sz_op, 1.0, &sz_op).unwrap();
        for site in 0..2 {
            let szsz = result.expectation_onsite(&szsz_op, site, 1).unwrap();
            assert!(
                (szsz - 2.0 / 3.0).abs() < TOL,
                "Expected <Sz_{}^2> = 2/3, got {}",
                site,
                szsz
            );
        }
    }
}
