use anyhow::Result;
use pyo3::prelude::*;

use crate::hamiltonian::kondo_lattice_hamiltonian::make_kondo_lattice_hamiltonian;
use crate::model::KondoLatticeModel;
use crate::solver::solver_core::solve_model;
use crate::solver::{SolverParameters, SolverResult};

/// Solve the Kondo lattice model to find the ground state energy and eigenvector.
#[pyfunction]
pub fn solve_kondo_lattice(
    model: &KondoLatticeModel,
    num_electrons: usize,
    target_total_sz: f64,
    params: &SolverParameters,
) -> Result<SolverResult> {
    let num_threads = params.num_threads;
    let model_clone = model.clone();

    // Calculate 2*Sz for quantum numbers
    let total_sz2 = (2.0 * target_total_sz).round() as i32;
    let current_quantum_numbers = vec![num_electrons as i32, total_sz2];

    solve_model(
        model_clone,
        |basis| make_kondo_lattice_hamiltonian(basis, model, num_threads),
        current_quantum_numbers,
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

    fn make_model(num_sites: usize, two_s: i32) -> KondoLatticeModel {
        KondoLatticeModel {
            num_sites,
            two_s_list: vec![two_s; num_sites],
            hopping: HashMap::new(),
            u_list: vec![0.0; num_sites],
            mu_list: vec![0.0; num_sites],
            hz_c_list: vec![0.0; num_sites],
            hz_f_list: vec![0.0; num_sites],
            d_list: vec![0.0; num_sites],
            density_density: HashMap::new(),
            kondo_xy_list: vec![0.0; num_sites],
            kondo_z_list: vec![0.0; num_sites],
            ff_exchange_xy: HashMap::new(),
            ff_exchange_z: HashMap::new(),
        }
    }

    #[test]
    fn test_solve_kondo_lattice_1site_kondo_singlet() {
        // 1-site Kondo model with S=1/2 localized spin and 1 electron
        // Kondo coupling J (isotropic): H = J S·s = J (Sz*sz + (1/2)(S+s- + S-s+))
        // For J > 0 (antiferromagnetic), the ground state is a singlet with energy E = -3J/4
        //
        // N=1, Sz=0 sector: 2 states |+1/2>|down>, |-1/2>|up>
        // H = J [Sz*sz + (1/2)(S+s- + S-s+)]
        //   = J [(1/2)(-1/2) + (1/2)*1] for off-diagonal
        // Matrix: [[-J/4, J/2], [J/2, -J/4]]
        // Eigenvalues: -J/4 - J/2 = -3J/4 (singlet), -J/4 + J/2 = J/4 (triplet component)

        let j = 2.0;
        let mut model = make_model(1, 1); // S=1/2
        model.kondo_xy_list = vec![j];
        model.kondo_z_list = vec![j];

        let params = make_solver_params();
        let result = solve_kondo_lattice(&model, 1, 0.0, &params).unwrap();

        assert_eq!(result.eigenvector.len(), 2);

        let expected_energy = -3.0 * j / 4.0;
        assert!(
            (result.energy - expected_energy).abs() < TOL,
            "Expected energy {}, got {}",
            expected_energy,
            result.energy
        );

        // Eigenvector should be (|+1/2>|down> - |-1/2>|up>)/√2 for singlet
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
        // For singlet state, c0 and c1 should have opposite signs
        assert!(
            c0 * c1 < 0.0,
            "Expected c0*c1 < 0 for singlet state, got c0={}, c1={}",
            c0,
            c1
        );
    }

    #[test]
    fn test_solve_kondo_lattice_2site_hopping_only() {
        // 2-site Kondo lattice with S=1/2 localized spins, 2 electrons, t=1, no Kondo coupling
        // This should behave like free electrons hopping between sites (Hubbard U=0)
        // The localized spins don't interact with the electrons
        //
        // N=2, Sz=0 sector
        // Since there's no Kondo coupling, the ground state energy should be -2t = -2
        // (both electrons in bonding orbitals)

        let mut model = make_model(2, 1); // S=1/2
        model.hopping.insert((0, 1), 1.0);

        let params = make_solver_params();
        let result = solve_kondo_lattice(&model, 2, 0.0, &params).unwrap();

        // Ground state energy should be -2t = -2.0 (both electrons in bonding state)
        assert!(
            (result.energy - (-2.0)).abs() < TOL,
            "Expected energy -2.0, got {}",
            result.energy
        );
    }

    #[test]
    fn test_solve_kondo_lattice_2site_ff_exchange() {
        // 2-site Kondo lattice with S=1/2 localized spins, 0 electrons
        // Only localized spin exchange J (Heisenberg-like)
        // This should behave exactly like a 2-site Heisenberg model
        //
        // N=0, Sz=0 sector: 2 states |+1/2,-1/2>|vac,vac>, |-1/2,+1/2>|vac,vac>
        // H = J (Sz1*Sz2 + (1/2)(S1+S2- + S1-S2+))
        // Ground state is singlet with energy E = -3J/4

        let j = 2.0;
        let mut model = make_model(2, 1); // S=1/2
        model.ff_exchange_xy.insert((0, 1), j);
        model.ff_exchange_z.insert((0, 1), j);

        let params = make_solver_params();
        let result = solve_kondo_lattice(&model, 0, 0.0, &params).unwrap();

        assert_eq!(result.eigenvector.len(), 2);

        let expected_energy = -3.0 * j / 4.0;
        assert!(
            (result.energy - expected_energy).abs() < TOL,
            "Expected energy {}, got {}",
            expected_energy,
            result.energy
        );
    }

    #[test]
    fn test_solve_kondo_lattice_2site_full() {
        // 2-site Kondo lattice with S=1/2 localized spins, 2 electrons
        // t=1, J_kondo=1 (antiferromagnetic)
        // This is a non-trivial test; we check that the solver runs
        // and produces reasonable output

        let mut model = make_model(2, 1); // S=1/2
        model.hopping.insert((0, 1), 1.0);
        model.kondo_xy_list = vec![1.0, 1.0];
        model.kondo_z_list = vec![1.0, 1.0];

        let params = make_solver_params();
        let result = solve_kondo_lattice(&model, 2, 0.0, &params).unwrap();

        // Energy should be negative
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
