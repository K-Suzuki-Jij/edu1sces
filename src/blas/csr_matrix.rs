use ahash::AHashMap;
use anyhow::{bail, Result};

pub const MATRIX_ZERO_EPS: f64 = 1e-13;

#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub row_dim: usize,
    pub col_dim: usize,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    pub fn new() -> Self {
        Self {
            row_dim: 0,
            col_dim: 0,
            rows: vec![0],
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    pub fn check(&self) -> Result<()> {
        if self.rows.len() != self.row_dim + 1 {
            bail!("rows.len() must be row_dim + 1");
        }
        if self.rows[0] != 0 {
            bail!("rows[0] must be 0");
        }
        if self.rows[self.row_dim] != self.cols.len() {
            bail!("rows[last] must equal nnz");
        }
        if self.cols.len() != self.vals.len() {
            bail!("cols.len() must equal vals.len()");
        }

        for i in 0..self.row_dim {
            if self.rows[i] > self.rows[i + 1] {
                bail!("rows must be non-decreasing");
            }

            let start = self.rows[i];
            let end = self.rows[i + 1];

            for p in start..end {
                if !self.vals[p].is_finite() {
                    bail!("vals must be finite (no NaN or inf)");
                }
                if self.cols[p] >= self.col_dim {
                    bail!("column index out of range");
                }
                if p + 1 < end && self.cols[p] >= self.cols[p + 1] {
                    bail!("column indices in each row must be strictly increasing");
                }
            }
        }
        Ok(())
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Returns whether this CSR matrix is symmetric within tolerance `tol`.
    /// Preconditions:
    /// - CSR is structurally valid.
    /// - Column indices in each row are strictly increasing (so binary_search is valid).
    pub fn is_symmetric(&self, tol: f64) -> Result<bool> {
        if self.row_dim != self.col_dim {
            bail!("matrix is not square");
        }

        self.check()?;

        let find_in_row = |r: usize, c: usize| -> Option<f64> {
            let start = self.rows[r];
            let end = self.rows[r + 1];
            match self.cols[start..end].binary_search(&c) {
                Ok(pos) => Some(self.vals[start + pos]),
                Err(_) => None,
            }
        };

        for i in 0..self.row_dim {
            let start = self.rows[i];
            let end = self.rows[i + 1];
            for p in start..end {
                let j = self.cols[p];
                let v_ij = self.vals[p];

                let Some(v_ji) = find_in_row(j, i) else {
                    return Ok(false);
                };

                if (v_ij - v_ji).abs() > tol {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

pub fn csr_add(c_m1: f64, m1: &CsrMatrix, c_m2: f64, m2: &CsrMatrix) -> Result<CsrMatrix> {
    if m1.row_dim != m2.row_dim || m1.col_dim != m2.col_dim {
        bail!("shape mismatch for add");
    }
    m1.check()?;
    m2.check()?;

    let mut out = CsrMatrix {
        row_dim: m1.row_dim,
        col_dim: m1.col_dim,
        rows: vec![0; m1.row_dim + 1],
        cols: Vec::new(),
        vals: Vec::new(),
    };

    let mut nnz = 0;
    for r in 0..m1.row_dim {
        let mut acc = AHashMap::new();

        for p in m1.rows[r]..m1.rows[r + 1] {
            *acc.entry(m1.cols[p]).or_insert(0.0) += c_m1 * m1.vals[p];
        }
        for p in m2.rows[r]..m2.rows[r + 1] {
            *acc.entry(m2.cols[p]).or_insert(0.0) += c_m2 * m2.vals[p];
        }

        let mut keys: Vec<usize> = acc.keys().copied().collect();
        keys.sort_unstable();
        for c in keys {
            let v = acc[&c];
            if v.abs() > MATRIX_ZERO_EPS {
                out.cols.push(c);
                out.vals.push(v);
                nnz += 1;
            }
        }
        out.rows[r + 1] = nnz;
    }

    Ok(out)
}

pub fn csr_mul(c_m1: f64, m1: &CsrMatrix, c_m2: f64, m2: &CsrMatrix) -> Result<CsrMatrix> {
    if m1.col_dim != m2.row_dim {
        bail!("shape mismatch for mul");
    }
    m1.check()?;
    m2.check()?;

    let scale = c_m1 * c_m2;

    let mut out = CsrMatrix {
        row_dim: m1.row_dim,
        col_dim: m2.col_dim,
        rows: vec![0; m1.row_dim + 1],
        cols: Vec::new(),
        vals: Vec::new(),
    };

    let mut nnz = 0;
    for r in 0..m1.row_dim {
        let mut acc = AHashMap::new();

        // out[r, c] += sum_k m1[r, k] * m2[k, c]
        for p in m1.rows[r]..m1.rows[r + 1] {
            let k = m1.cols[p];
            let a = m1.vals[p] * scale;

            for q in m2.rows[k]..m2.rows[k + 1] {
                let c = m2.cols[q];
                let b = m2.vals[q];
                *acc.entry(c).or_insert(0.0) += a * b;
            }
        }

        let mut keys: Vec<usize> = acc.keys().copied().collect();
        keys.sort_unstable();

        for c in keys {
            let v = acc[&c];
            if v.abs() > MATRIX_ZERO_EPS {
                out.cols.push(c);
                out.vals.push(v);
                nnz += 1;
            }
        }
        out.rows[r + 1] = nnz;
    }

    Ok(out)
}

pub fn csr_transpose(c: f64, m: &CsrMatrix) -> Result<CsrMatrix> {
    m.check()?;

    // out = c * m^T
    let mut out = CsrMatrix {
        row_dim: m.col_dim,
        col_dim: m.row_dim,
        rows: vec![0; m.col_dim + 1],
        cols: vec![0; m.nnz()],
        vals: vec![0.0; m.nnz()],
    };

    // 1) count nnz per output row (which corresponds to each input col)
    for &col in &m.cols {
        out.rows[col + 1] += 1;
    }

    // 2) prefix sum to build row pointers
    for i in 0..out.row_dim {
        out.rows[i + 1] += out.rows[i];
    }

    // 3) fill using a working cursor per row
    let mut cursor = out.rows.clone();
    for r in 0..m.row_dim {
        for p in m.rows[r]..m.rows[r + 1] {
            let c_in = m.cols[p];
            let dst = cursor[c_in];

            out.cols[dst] = r;
            out.vals[dst] = c * m.vals[p];

            cursor[c_in] += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_matrix() {
        let m = CsrMatrix::new();

        assert_eq!(m.row_dim, 0);
        assert_eq!(m.col_dim, 0);
        assert!(m.rows.len() == 1 && m.rows[0] == 0);
        assert!(m.cols.is_empty());
        assert!(m.vals.is_empty());

        assert!(m.check().is_ok());
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn check_accepts_valid_nonempty_csr() {
        // 2 x 3 行列
        // row 0: col 0, 2
        // row 1: col 1
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 2, 3],
            cols: vec![0, 2, 1],
            vals: vec![1.0, 2.0, 3.0],
        };

        assert!(m.check().is_ok());
        assert_eq!(m.nnz(), 3);
    }

    #[test]
    fn check_rejects_invalid_rows_length() {
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1], // should be len = row_dim + 1 = 3
            cols: vec![0],
            vals: vec![1.0],
        };

        assert!(m.check().is_err());
    }

    #[test]
    fn check_rejects_cols_vals_mismatch() {
        let m = CsrMatrix {
            row_dim: 1,
            col_dim: 2,
            rows: vec![0, 1],
            cols: vec![0, 1],
            vals: vec![1.0],
        };

        assert!(m.check().is_err());
    }

    #[test]
    fn check_rejects_out_of_range_column() {
        let m = CsrMatrix {
            row_dim: 1,
            col_dim: 2,
            rows: vec![0, 1],
            cols: vec![2], // col_dim = 2 -> invalid
            vals: vec![1.0],
        };

        assert!(m.check().is_err());
    }

    #[test]
    fn csr_add_basic() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 3],
            cols: vec![0, 1, 1],
            vals: vec![1.0, 2.0, 3.0],
        };

        let b = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 3],
            cols: vec![0, 0, 1],
            vals: vec![4.0, 5.0, 6.0],
        };

        let c = csr_add(2.0, &a, -0.5, &b).unwrap();

        assert_eq!(c.row_dim, 2);
        assert_eq!(c.col_dim, 2);
        assert_eq!(c.rows, vec![0, 1, 3]);
        assert_eq!(c.cols, vec![1, 0, 1]);
        assert_eq!(c.vals, vec![4.0, -2.5, 3.0]);
        assert!(c.check().is_ok());
    }

    #[test]
    fn csr_add_cancels_to_zero_by_eps() {
        let a = CsrMatrix {
            row_dim: 1,
            col_dim: 1,
            rows: vec![0, 1],
            cols: vec![0],
            vals: vec![1.0],
        };

        let b = CsrMatrix {
            row_dim: 1,
            col_dim: 1,
            rows: vec![0, 1],
            cols: vec![0],
            vals: vec![-1.0],
        };

        let c = csr_add(1.0, &a, 1.0, &b).unwrap();

        assert_eq!(c.rows, vec![0, 0]);
        assert!(c.cols.is_empty());
        assert!(c.vals.is_empty());
        assert!(c.check().is_ok());
    }

    #[test]
    fn csr_add_dimension_mismatch_is_error() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 0, 0],
            cols: vec![],
            vals: vec![],
        };

        let b = CsrMatrix {
            row_dim: 3, // mismatch
            col_dim: 2,
            rows: vec![0, 0, 0, 0],
            cols: vec![],
            vals: vec![],
        };

        assert!(csr_add(1.0, &a, 1.0, &b).is_err());
    }

    #[test]
    fn csr_mul_basic_with_coefficients() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 2, 3],
            cols: vec![0, 1, 1],
            vals: vec![1.0, 2.0, 3.0],
        };

        let b = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 3],
            cols: vec![0, 0, 1],
            vals: vec![4.0, 5.0, 6.0],
        };

        let c = csr_mul(2.0, &a, -0.5, &b).unwrap();

        assert_eq!(c.row_dim, 2);
        assert_eq!(c.col_dim, 2);
        assert_eq!(c.rows, vec![0, 2, 4]);
        assert_eq!(c.cols, vec![0, 1, 0, 1]);
        assert_eq!(c.vals, vec![-14.0, -12.0, -15.0, -18.0]);
        assert!(c.check().is_ok());
    }

    #[test]
    fn csr_mul_zero_left() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 0, 0],
            cols: vec![],
            vals: vec![],
        };

        let b = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 1, 2],
            cols: vec![0, 1],
            vals: vec![1.0, 1.0],
        };

        let c = csr_mul(1.0, &a, 1.0, &b).unwrap();

        assert_eq!(c.rows, vec![0, 0, 0]);
        assert!(c.cols.is_empty());
        assert!(c.vals.is_empty());
        assert!(c.check().is_ok());
    }

    #[test]
    fn csr_mul_dimension_mismatch_is_error() {
        let a = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 0, 0],
            cols: vec![],
            vals: vec![],
        };

        let b = CsrMatrix {
            row_dim: 2,
            col_dim: 2,
            rows: vec![0, 0, 0],
            cols: vec![],
            vals: vec![],
        };

        assert!(csr_mul(1.0, &a, 1.0, &b).is_err());
    }

    #[test]
    fn csr_transpose_basic() {
        // m =
        // [ 1 0 2 ]
        // [ 0 3 0 ]
        let m = CsrMatrix {
            row_dim: 2,
            col_dim: 3,
            rows: vec![0, 2, 3],
            cols: vec![0, 2, 1],
            vals: vec![1.0, 2.0, 3.0],
        };

        let t = csr_transpose(1.0, &m).unwrap();

        // m^T =
        // [ 1 0 ]
        // [ 0 3 ]
        // [ 2 0 ]
        assert_eq!(t.row_dim, 3);
        assert_eq!(t.col_dim, 2);
        assert_eq!(t.rows, vec![0, 1, 2, 3]);
        assert_eq!(t.cols, vec![0, 1, 0]);
        assert_eq!(t.vals, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn csr_transpose_scales() {
        let m = CsrMatrix {
            row_dim: 1,
            col_dim: 2,
            rows: vec![0, 2],
            cols: vec![0, 1],
            vals: vec![1.5, -2.0],
        };

        let t = csr_transpose(2.0, &m).unwrap();

        assert_eq!(t.row_dim, 2);
        assert_eq!(t.col_dim, 1);
        assert_eq!(t.rows, vec![0, 1, 2]);
        assert_eq!(t.cols, vec![0, 0]);
        assert_eq!(t.vals, vec![3.0, -4.0]);
    }
}
