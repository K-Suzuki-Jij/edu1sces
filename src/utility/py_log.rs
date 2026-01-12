use pyo3::prelude::*;

/// Print a line to Python's stdout and flush immediately.
/// This enables real-time output in Jupyter notebooks.
pub fn py_print_flush(line: &str) {
    let _ = Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let stdout = sys.getattr("stdout")?;
        stdout.call_method1("write", (line,))?;
        stdout.call_method0("flush")?;
        Ok(())
    });
}

/// Print a line that overwrites the current line (using \r).
/// Useful for progress indicators that update in place.
pub fn py_print_overwrite(line: &str) {
    let _ = Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let stdout = sys.getattr("stdout")?;
        // \r to return to line start, pad with spaces to clear previous content
        let formatted = format!("\r{:<50}", line);
        stdout.call_method1("write", (formatted,))?;
        stdout.call_method0("flush")?;
        Ok(())
    });
}
