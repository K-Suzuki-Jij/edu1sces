use crate::basis::HeisenbergBasis;
use crate::blas::{CsrMatrix, MATRIX_ZERO_EPS};
use crate::hamiltonian::{make_intersite_elements, make_onsite_elements, TransitionStateHolder};
use crate::model::HeisenbergModel;
use anyhow::{bail, Result};

#[derive(Debug)]
struct HeisenbergLocalOps {
    pub onsite_ham: Vec<CsrMatrix>, // H_i = hz_i Sz_i + d_i Sz_i^2
    pub sz: Vec<CsrMatrix>,
    pub sp: Vec<CsrMatrix>,
    pub sm: Vec<CsrMatrix>,
}

impl HeisenbergLocalOps {
    pub fn new(model: &HeisenbergModel) -> Result<Self> {
        let n = model.num_sites;

        let mut sz = Vec::with_capacity(n);
        let mut sp = Vec::with_capacity(n);
        let mut sm = Vec::with_capacity(n);
        let mut onsite_ham = Vec::with_capacity(n);

        for site in 0..n {
            let sz_i = model.make_local_op_sz(site)?;
            let sp_i = model.make_local_op_sp(site)?;
            let sm_i = model.make_local_op_sm(site)?; // transpose(sp)

            let on_i = model.make_local_onsite_hamiltonian(site)?; // hz*Sz + d*Sz^2

            sz.push(sz_i);
            sp.push(sp_i);
            sm.push(sm_i);
            onsite_ham.push(on_i);
        }

        Ok(Self {
            onsite_ham,
            sz,
            sp,
            sm,
        })
    }
}

fn make_hamiltonian_elements(
    basis_state: i128,
    basis: &HeisenbergBasis,
    model: &HeisenbergModel,
    local_ops: &HeisenbergLocalOps,
    holder: &mut TransitionStateHolder,
) {
    holder.vals.clear();

    // Local basis for the current input basis
    for site in 0..model.num_sites {
        holder.local_basis[site] = basis.find_local_basis(basis_state, site);
    }

    // Onsite terms: H_i = hz_i * Sz_i + d_i * Sz_i^2
    for site in 0..model.num_sites {
        make_onsite_elements(basis_state, site, &local_ops.onsite_ham[site], 1.0, holder);
    }

    // Intersite terms: Jz * Sz_i Sz_j
    for (&(i, j), &val) in model.exchange_z.iter() {
        make_intersite_elements(
            basis_state,
            i,
            j,
            &local_ops.sz[i],
            &local_ops.sz[j],
            val,
            1.0,
            holder,
        );
    }

    // Intersite terms: (Jxy/2) * (Sp_i Sm_j + Sm_i Sp_j)
    for (&(i, j), &val) in model.exchange_xy.iter() {
        let c = 0.5 * val;

        make_intersite_elements(
            basis_state,
            i,
            j,
            &local_ops.sp[i],
            &local_ops.sm[j],
            c,
            1.0,
            holder,
        );

        make_intersite_elements(
            basis_state,
            i,
            j,
            &local_ops.sm[i],
            &local_ops.sp[j],
            c,
            1.0,
            holder,
        );
    }
}

pub fn make_heisenberg_hamiltonian(
    basis: &HeisenbergBasis,
    model: &HeisenbergModel,
    lower_only: bool,
) -> Result<CsrMatrix> {
    let dim = basis.basis.len();
    if dim == 0 {
        bail!("Target Hilbert space has zero dimension.");
    }

    let local_ops = HeisenbergLocalOps::new(model)?;
    let zero_eps = MATRIX_ZERO_EPS;

    let mut holder = TransitionStateHolder {
        vals: ahash::AHashMap::new(),
        site_base: basis.site_base.clone(),
        local_basis: vec![0; model.num_sites],
        zero_eps,
    };

    // ---------- pass 1: count nnz ----------
    let mut row_nnz = vec![0; dim];

    for row in 0..dim {
        let basis_state = basis.basis[row];
        make_hamiltonian_elements(basis_state, basis, model, &local_ops, &mut holder);

        let mut cnt = 0;
        for (&transition_basis, &_val) in holder.vals.iter() {
            let Some(&col) = basis.inverse_basis.get(&transition_basis) else {
                bail!("transition_basis not found in inverse_basis");
            };
            if lower_only {
                if col <= row {
                    cnt += 1;
                }
            } else {
                cnt += 1;
            }
        }

        if lower_only && holder.vals.get(&basis_state).is_none() {
            // inject diagonal zero
            cnt += 1;
        }

        row_nnz[row] = cnt;
    }

    // prefix sum
    let mut out = CsrMatrix::new();
    out.row_dim = dim;
    out.col_dim = dim;

    out.rows = vec![0; dim + 1];
    for i in 0..dim {
        out.rows[i + 1] = out.rows[i] + row_nnz[i];
    }

    let nnz = out.rows[dim];
    out.cols = vec![0; nnz];
    out.vals = vec![0.0; nnz];

    // ---------- pass 2: fill ----------
    for row in 0..dim {
        let basis_state = basis.basis[row];
        make_hamiltonian_elements(basis_state, basis, model, &local_ops, &mut holder);

        let mut entries = Vec::with_capacity(row_nnz[row]);
        for (&transition_basis, &v) in holder.vals.iter() {
            let Some(&col) = basis.inverse_basis.get(&transition_basis) else {
                bail!("transition_basis not found in inverse_basis");
            };
            if lower_only && col > row {
                continue;
            }
            entries.push((col, v));
        }

        if lower_only && holder.vals.get(&basis_state).is_none() {
            entries.push((row, 0.0));
        }

        // Sort entries by column index
        entries.sort_unstable_by_key(|&(col, _v)| col);

        // Write entries to output CSR matrix
        let mut write = out.rows[row];
        for (col, v) in entries {
            out.cols[write] = col;
            out.vals[write] = v;
            write += 1;
        }

        assert_eq!(
            write,
            out.rows[row + 1],
            "internal error: write pointer mismatch"
        );
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn ham_two_spin_half_two_sites_sz0_full_matrix_with_onsite() {
        let tol = 1e-12;

        // Two sites, S=1/2 each, total Sz = 0 sector
        //
        // Basis order (sorted):
        // |↓↑>, |↑↓>
        //
        // Parameters:
        //   Jz = 1
        //   Jxy = 1
        //   hz = [h, h] with h = 2
        //   d  = [d, d] with d = 3
        //
        // Onsite contribution per site:
        //   h Sz + d Sz^2
        //
        // For |↓↑> and |↑↓>:
        //   Sz total = 0  -> hz term cancels
        //   Sz^2 total = 1/4 + 1/4 = 1/2
        //   onsite = d * 1/2 = 3/2 = 1.5
        //
        // Intersite:
        //   Sz1 Sz2 = -1/4
        //   flip-flop = 1/2
        //
        // Total matrix:
        // [ 1.5 - 0.25,  0.5 ]
        // [ 0.5,  1.5 - 0.25 ]
        // =
        // [ 1.25, 0.5 ]
        // [ 0.5,  1.25 ]

        let mut exchange_xy = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);

        let mut exchange_z = HashMap::new();
        exchange_z.insert((0, 1), 1.0);

        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![2.0, 2.0],
            d_list: vec![3.0, 3.0],
            exchange_xy,
            exchange_z,
        };

        let basis = HeisenbergBasis::new(model.clone(), 0.0).unwrap();
        assert_eq!(basis.basis.len(), 2);

        let h = make_heisenberg_hamiltonian(&basis, &model, false).unwrap();

        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert_eq!(h.rows, vec![0, 2, 4]);

        // CSR sorted by column
        assert_eq!(h.cols, vec![0, 1, 0, 1]);

        assert!((h.vals[0] - 1.25).abs() <= tol);
        assert!((h.vals[1] - 0.5).abs() <= tol);
        assert!((h.vals[2] - 0.5).abs() <= tol);
        assert!((h.vals[3] - 1.25).abs() <= tol);

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_two_spin_half_two_sites_sz0_lower_only_with_onsite() {
        let tol = 1e-12;

        // Same as full-matrix test, but store only lower triangle (col <= row).
        //
        // Parameters:
        //   Jz = 1, Jxy = 1
        //   hz = [2, 2], d = [3, 3]
        //
        // In {|↓↑>, |↑↓>}:
        // diagonal = (-1/4) + d*(Sz1^2+Sz2^2) = -0.25 + 3*(1/2) = 1.25
        // offdiag  = 1/2 (only (1,0) kept in lower triangle)
        //
        // Expected lower-triangular CSR:
        // row 0: (0,0) = 1.25
        // row 1: (1,0) = 0.5, (1,1) = 1.25

        let mut exchange_xy = HashMap::new();
        exchange_xy.insert((0, 1), 1.0);

        let mut exchange_z = HashMap::new();
        exchange_z.insert((0, 1), 1.0);

        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![2.0, 2.0],
            d_list: vec![3.0, 3.0],
            exchange_xy,
            exchange_z,
        };

        let basis = HeisenbergBasis::new(model.clone(), 0.0).unwrap();
        let h = make_heisenberg_hamiltonian(&basis, &model, true).unwrap();

        assert_eq!(h.row_dim, 2);
        assert_eq!(h.col_dim, 2);
        assert_eq!(h.rows, vec![0, 1, 3]);
        assert_eq!(h.cols, vec![0, 0, 1]);

        assert!((h.vals[0] - 1.25).abs() <= tol);
        assert!((h.vals[1] - 0.5).abs() <= tol);
        assert!((h.vals[2] - 1.25).abs() <= tol);

        assert!(h.check().is_ok());
        assert!(!h.is_symmetric(tol).unwrap());
    }

    #[test]
    fn ham_one_site_onsite_only_s1_total_sz_plus_one() {
        let tol = 1e-12;

        // One site, S=1, total Sz = +1 sector -> 1x1 matrix.
        // H = hz*Sz + d*Sz^2, with hz=2, d=3:
        // Sz = +1, Sz^2 = 1 -> H = 2*1 + 3*1 = 5
        let model = HeisenbergModel {
            num_sites: 1,
            two_s_list: vec![2],
            hz_list: vec![2.0],
            d_list: vec![3.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        let basis = HeisenbergBasis::new(model.clone(), 1.0).unwrap();
        assert_eq!(basis.basis.len(), 1);

        let h = make_heisenberg_hamiltonian(&basis, &model, false).unwrap();

        assert_eq!(h.row_dim, 1);
        assert_eq!(h.col_dim, 1);
        assert_eq!(h.rows, vec![0, 1]);
        assert_eq!(h.cols, vec![0]);
        assert_eq!(h.vals.len(), 1);
        assert!((h.vals[0] - 5.0).abs() <= tol);

        assert!(h.check().is_ok());
        assert!(h.is_symmetric(tol).unwrap());
    }
}
