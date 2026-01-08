pub mod basis;
pub mod blas;
pub mod hamiltonian;
pub mod model;
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
    Ok(())
}
