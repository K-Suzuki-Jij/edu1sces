use anyhow::Result;

use crate::basis::{HilbertBasis, HubbardBasis};
use crate::blas::{inverse_iteration, lanczos, LanczosParameters};
use crate::hamiltonian::hubbard_hamiltonian::make_hubbard_hamiltonian;
use crate::model::HubbardModel;
use crate::solver::{SolverParameters, SolverResult};

/// Solve the Hubbard model to find the ground state energy and eigenvector.
pub fn solve_hubbard(
    model: &HubbardModel,
    num_electrons: usize,
    target_total_sz: f64,
    params: &SolverParameters,
) -> Result<SolverResult> {
    // Build basis
    let basis_obj = HubbardBasis::new(model.clone(), num_electrons, target_total_sz)?;
    let dim = basis_obj.dim();

    // Build Hamiltonian
    let hamiltonian = make_hubbard_hamiltonian(&basis_obj, model, params.num_threads)?;

    // Prepare Lanczos parameters
    let lanczos_params = LanczosParameters {
        acc: params.eigenvalue_tol,
        min_step: params.min_step,
        max_step: params.max_step,
        calc_eigenvec: true,
    };

    // Run Lanczos
    let mut eigenvector = Vec::new();
    let mut energy = 0.0;
    lanczos(
        &hamiltonian,
        &mut eigenvector,
        &mut energy,
        &lanczos_params,
        params.num_threads,
    )?;

    // Refine eigenvector using inverse iteration
    inverse_iteration(
        &hamiltonian,
        &mut eigenvector,
        energy,
        &params.inverse_iteration_params,
        params.num_threads,
    )?;

    // Extract basis data
    let mut basis = Vec::with_capacity(dim);
    for i in 0..dim {
        basis.push(basis_obj.basis_state_at(i));
    }
    let inverse_basis = basis_obj.inverse_basis().clone();

    Ok(SolverResult {
        dim,
        energy,
        eigenvector,
        basis,
        inverse_basis,
    })
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
    fn test_solve_hubbard_2site_1electron() {
        // 2-site Hubbard model with 1 electron, t=1, U=0
        // N=1, Sz=0.5 sector: 2 states |up,vac>, |vac,up>
        // H = -t (c_0^dag c_1 + c_1^dag c_0)
        // Matrix: [[0, -t], [-t, 0]]
        // Eigenvalues: -t, +t
        // Ground state energy: E = -t = -1.0

        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let model = HubbardModel::new(
            hopping,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_hubbard(&model, 1, 0.5, &params).unwrap();

        assert_eq!(result.dim, 2);
        assert!(
            (result.energy - (-1.0)).abs() < TOL,
            "Expected energy -1.0, got {}",
            result.energy
        );

        // Eigenvector should be (|up,vac> + |vac,up>)/√2 or its negative
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
        // For bonding state, c0 and c1 should have the same sign
        assert!(
            c0 * c1 > 0.0,
            "Expected c0*c1 > 0 for bonding state, got c0={}, c1={}",
            c0,
            c1
        );
    }

    #[test]
    fn test_solve_hubbard_2site_2electrons_u0() {
        // 2-site Hubbard model with 2 electrons, t=1, U=0
        // N=2, Sz=0 sector: 4 states
        // |updown,vac>=3, |down,up>=6, |up,down>=9, |vac,updown>=12
        //
        // For U=0, the ground state has energy E = -2t (both electrons in bonding orbital)

        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let model = HubbardModel::new(
            hopping,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_hubbard(&model, 2, 0.0, &params).unwrap();

        assert_eq!(result.dim, 4);
        assert!(
            (result.energy - (-2.0)).abs() < TOL,
            "Expected energy -2.0, got {}",
            result.energy
        );
    }

    #[test]
    fn test_solve_hubbard_2site_2electrons_large_u() {
        // 2-site Hubbard model with 2 electrons, t=1, U=100 (strong coupling)
        // N=2, Sz=0 sector
        //
        // In the strong coupling limit (U >> t), the ground state approaches
        // the Heisenberg singlet with energy ≈ -4t^2/U = -0.04
        // More precisely, for finite U, E = U/2 - sqrt((U/2)^2 + 4t^2)
        // For U=100, t=1: E = 50 - sqrt(2500 + 4) ≈ 50 - 50.04 ≈ -0.04

        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);

        let model = HubbardModel::new(
            hopping,
            vec![100.0, 100.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_hubbard(&model, 2, 0.0, &params).unwrap();

        assert_eq!(result.dim, 4);

        // Expected energy: E = U/2 - sqrt((U/2)^2 + 4t^2)
        let u: f64 = 100.0;
        let t: f64 = 1.0;
        let expected_energy = u / 2.0 - ((u / 2.0).powi(2) + 4.0 * t.powi(2)).sqrt();

        assert!(
            (result.energy - expected_energy).abs() < TOL,
            "Expected energy {}, got {}",
            expected_energy,
            result.energy
        );
    }

    #[test]
    fn test_solve_hubbard_4site_half_filling() {
        // 4-site Hubbard ring at half-filling (4 electrons), t=1, U=4
        // N=4, Sz=0 sector: dim = C(4,2)*C(4,2) = 36
        //
        // This is a non-trivial case; we just check that the solver runs
        // and produces reasonable output

        let mut hopping = HashMap::new();
        hopping.insert((0, 1), 1.0);
        hopping.insert((1, 2), 1.0);
        hopping.insert((2, 3), 1.0);
        hopping.insert((3, 0), 1.0); // periodic boundary

        let model = HubbardModel::new(
            hopping,
            vec![4.0, 4.0, 4.0, 4.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap();

        let params = make_solver_params();
        let result = solve_hubbard(&model, 4, 0.0, &params).unwrap();

        assert_eq!(result.dim, 36);

        // Energy should be negative (attractive correlations)
        assert!(
            result.energy < 0.0,
            "Expected negative energy, got {}",
            result.energy
        );

        // Eigenvector should be normalized
        let norm_sq: f64 = result.eigenvector.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < TOL,
            "Expected normalized eigenvector, got norm^2 = {}",
            norm_sq
        );
    }
}
