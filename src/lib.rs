pub mod model;

use pyo3::prelude::*;

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::operator::SpinOperator>()?;
    m.add_class::<model::operator::ElectronOperator>()?;
    Ok(())
}
