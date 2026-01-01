use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpinOperator {
    Sz,
    Sp,
    Sm,
    Sx,
    ISy,
}

impl SpinOperator {
    fn latex(&self) -> &'static str {
        match self {
            SpinOperator::Sz => r"\hat{S}_z",
            SpinOperator::Sp => r"\hat{S}^{+}",
            SpinOperator::Sm => r"\hat{S}^{-}",
            SpinOperator::Sx => r"\hat{S}_x",
            SpinOperator::ISy => r"i\hat{S}_y",
        }
    }

    fn ascii(&self) -> &'static str {
        match self {
            SpinOperator::Sz => "Sz",
            SpinOperator::Sp => "S+",
            SpinOperator::Sm => "S-",
            SpinOperator::Sx => "Sx",
            SpinOperator::ISy => "iSy",
        }
    }
}

#[pymethods]
impl SpinOperator {
    /// Jupyter: LaTeX math rendering
    fn _repr_latex_(&self) -> String {
        format!("${}$", self.latex())
    }

    /// Python REPL / debug
    fn __repr__(&self) -> String {
        self.ascii().to_string()
    }

    /// print()
    fn __str__(&self) -> String {
        self.ascii().to_string()
    }
}

/// Conduction-electron local operators (site-local)
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElectronOperator {
    /// c_{↑}
    CUp,
    /// c_{↓}
    CDown,
    /// c^†_{↑}
    CDagUp,
    /// c^†_{↓}
    CDagDown,
    /// n_{↑}
    NUp,
    /// n_{↓}
    NDown,
    /// n = n_{↑} + n_{↓}
    N,
    /// s^z = (n_{↑}-n_{↓})/2
    Sz,
    /// s^+ = c^†_{↑} c_{↓}
    Sp,
    /// s^- = c^†_{↓} c_{↑}
    Sm,
}

impl ElectronOperator {
    fn latex(self) -> &'static str {
        match self {
            ElectronOperator::CUp => r"\hat{c}_{\uparrow}",
            ElectronOperator::CDown => r"\hat{c}_{\downarrow}",
            ElectronOperator::CDagUp => r"\hat{c}^{\dagger}_{\uparrow}",
            ElectronOperator::CDagDown => r"\hat{c}^{\dagger}_{\downarrow}",
            ElectronOperator::NUp => r"\hat{n}_{\uparrow}",
            ElectronOperator::NDown => r"\hat{n}_{\downarrow}",
            ElectronOperator::N => r"\hat{n}",
            ElectronOperator::Sz => r"\hat{s}_z",
            ElectronOperator::Sp => r"\hat{s}^{+}",
            ElectronOperator::Sm => r"\hat{s}^{-}",
        }
    }

    fn ascii(self) -> &'static str {
        match self {
            ElectronOperator::CUp => "c_up",
            ElectronOperator::CDown => "c_down",
            ElectronOperator::CDagUp => "c†_up",
            ElectronOperator::CDagDown => "c†_down",
            ElectronOperator::NUp => "n_up",
            ElectronOperator::NDown => "n_down",
            ElectronOperator::N => "n",
            ElectronOperator::Sz => "Sz",
            ElectronOperator::Sp => "S+",
            ElectronOperator::Sm => "S-",
        }
    }
}

#[pymethods]
impl ElectronOperator {
    fn _repr_latex_(&self) -> String {
        format!("${}$", (*self).latex())
    }

    fn __repr__(&self) -> String {
        (*self).ascii().to_string()
    }

    fn __str__(&self) -> String {
        (*self).ascii().to_string()
    }
}
