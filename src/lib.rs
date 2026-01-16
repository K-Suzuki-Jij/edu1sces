pub mod basis;
pub mod blas;
pub mod examples_util;
pub mod hamiltonian;
pub mod model;
pub mod solver;
pub mod utility;

extern crate lapack_src;

use pyo3::prelude::*;

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::operator::SpinOperator>()?;
    m.add_class::<model::operator::ElectronOperator>()?;
    m.add_class::<model::heisenberg::HeisenbergModel>()?;
    m.add_class::<model::hubbard::HubbardModel>()?;
    m.add_class::<model::kondo_lattice::KondoLatticeModel>()?;
    m.add_class::<model::kondo_lattice_2ch::KondoLattice2ChModel>()?;
    m.add_class::<blas::CsrMatrix>()?;
    m.add_class::<solver::SolverResult>()?;
    m.add_class::<solver::SolverParameters>()?;
    m.add_class::<blas::InverseIterationParameters>()?;
    m.add_class::<blas::ConjugateGradientParameters>()?;
    m.add_class::<blas::LanczosLog>()?;
    m.add_class::<blas::InverseIterationLog>()?;
    m.add_class::<blas::ConjugateGradientLog>()?;
    m.add_function(wrap_pyfunction!(
        crate::solver::solve_heisenberg::solve_heisenberg,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::solver::solve_hubbard::solve_hubbard,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::solver::solve_kondo_lattice::solve_kondo_lattice,
        m
    )?)?;
    Ok(())
}
