use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::basis::hilbert_basis::HilbertBasis;
use crate::blas::CsrMatrix;
use crate::blas::MATRIX_ZERO_EPS;
use crate::hamiltonian::hamiltonian_element_generator::HamiltonianElementGenerator;
use crate::hamiltonian::transition_state_holder::TransitionStateHolder;

pub fn make_hamiltonian_parallel<Basis, Generator>(
    basis: &Basis,
    generator: &Generator,
    lower_only: bool,
    num_threads: usize,
) -> Result<CsrMatrix>
where
    Basis: HilbertBasis + Sync,
    Generator: HamiltonianElementGenerator<Basis>,
{
    let dim = basis.dim();
    if dim == 0 {
        bail!("Target Hilbert space has zero dimension.");
    }

    let num_sites = basis.site_base().len();
    if num_sites == 0 {
        bail!("The system size is zero.");
    }

    let make_holder = || TransitionStateHolder {
        vals: ahash::AHashMap::new(),
        site_base: basis.site_base().to_vec(),
        local_basis: vec![0; num_sites],
        zero_eps: MATRIX_ZERO_EPS,
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build rayon thread pool: {e}"))?;

    pool.install(|| -> Result<CsrMatrix> {
        // ---------- pass 1: count nnz (parallel) ----------
        let mut row_nnz = vec![0; dim];

        row_nnz
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row, slot)| -> Result<()> {
                let basis_state = basis.basis_state_at(row);
                let mut holder = make_holder();

                generator.make_elements(basis_state, basis, &mut holder)?;

                let mut cnt = 0;
                for (&transition_basis, &_v) in holder.vals.iter() {
                    let col = basis.inverse_basis_at(transition_basis)?;
                    if !lower_only || col <= row {
                        cnt += 1;
                    }
                }

                if lower_only && holder.vals.get(&basis_state).is_none() {
                    cnt += 1; // inject diagonal zero
                }

                *slot = cnt;
                Ok(())
            })?;

        // ---------- prefix sum (sequential) ----------
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

        // Prepare per-row mutable slices for parallel fill (sequential pre-step).
        //
        // out.cols / out.vals are laid out contiguously according to row_nnz.
        // Split them into per-row mutable slices in advance.
        // Each parallel closure writes only to its own row slice.
        let mut cols_rem = out.cols.as_mut_slice();
        let mut vals_rem = out.vals.as_mut_slice();

        let mut row_slices: Vec<(&mut [usize], &mut [f64])> = Vec::with_capacity(dim);
        for row in 0..dim {
            let len = row_nnz[row];
            let (c_head, c_tail) = cols_rem.split_at_mut(len);
            let (v_head, v_tail) = vals_rem.split_at_mut(len);
            row_slices.push((c_head, v_head));
            cols_rem = c_tail;
            vals_rem = v_tail;
        }

        // ---------- pass 2: fill (parallel) ----------
        row_slices.into_par_iter().enumerate().try_for_each(
            |(row, (row_cols, row_vals))| -> Result<()> {
                let basis_state = basis.basis_state_at(row);
                let mut holder = make_holder();

                generator.make_elements(basis_state, basis, &mut holder)?;

                let mut entries = Vec::with_capacity(row_cols.len());

                for (&transition_basis, &v) in holder.vals.iter() {
                    let col = basis.inverse_basis_at(transition_basis)?;
                    if !lower_only || col <= row {
                        entries.push((col, v));
                    }
                }

                if lower_only && holder.vals.get(&basis_state).is_none() {
                    entries.push((row, 0.0));
                }

                entries.sort_unstable_by_key(|&(col, _)| col);

                if entries.len() != row_cols.len() {
                    bail!("internal error: nnz mismatch at row={}", row);
                }

                for (k, (col, v)) in entries.into_iter().enumerate() {
                    row_cols[k] = col;
                    row_vals[k] = v;
                }

                Ok(())
            },
        )?;

        Ok(out)
    })
}
