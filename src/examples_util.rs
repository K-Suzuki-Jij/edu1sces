use std::collections::HashMap;

use rand::Rng;

use crate::blas::CsrMatrix;
use crate::model::{HeisenbergModel, HubbardModel};

pub fn build_heisenberg_chain(
    two_s: i32,
    n: usize,
    jxy: f64,
    jz: f64,
    hz: f64,
    d: f64,
) -> HeisenbergModel {
    let mut exchange_xy = HashMap::new();
    let mut exchange_z = HashMap::new();

    for i in 0..n - 1 {
        exchange_xy.insert((i, i + 1), jxy);
        exchange_z.insert((i, i + 1), jz);
    }

    HeisenbergModel {
        num_sites: n,
        two_s_list: vec![two_s; n],
        hz_list: vec![hz; n],
        d_list: vec![d; n],
        exchange_xy,
        exchange_z,
    }
}

pub fn build_hubbard_chain(
    n: usize,
    t: f64,
    u: f64,
    mu: f64,
    hz: f64,
    v: f64,
    jxy: f64,
    jz: f64,
) -> HubbardModel {
    let mut hopping = HashMap::new();
    let mut density_density = HashMap::new();
    let mut exchange_xy = HashMap::new();
    let mut exchange_z = HashMap::new();

    for i in 0..n - 1 {
        hopping.insert((i, i + 1), t);
        density_density.insert((i, i + 1), v);
        exchange_xy.insert((i, i + 1), jxy);
        exchange_z.insert((i, i + 1), jz);
    }

    HubbardModel {
        num_sites: n,
        hopping,
        u_list: vec![u; n],
        mu_list: vec![mu; n],
        hz_list: vec![hz; n],
        density_density,
        exchange_xy,
        exchange_z,
    }
}

/// Generate a random sparse matrix in CSR format.
pub fn generate_sparse_csr(n: usize, density: f64) -> CsrMatrix {
    let mut rng = rand::rng();

    let expected_nnz = (n as f64 * n as f64 * density) as usize;

    // Sample random indices and values
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(expected_nnz);
    for _ in 0..expected_nnz {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let val = rng.random_range(-1.0..1.0);
        entries.push((i, j, val));
    }

    // Sort by (row, col) and remove duplicates
    entries.sort_by_key(|&(i, j, _)| (i, j));
    entries.dedup_by_key(|e| (e.0, e.1));

    // Build CSR
    let mut rows = vec![0usize; n + 1];
    let mut cols = Vec::with_capacity(entries.len());
    let mut vals = Vec::with_capacity(entries.len());

    for &(i, j, val) in &entries {
        rows[i + 1] += 1;
        cols.push(j);
        vals.push(val);
    }

    // Cumulative sum
    for i in 0..n {
        rows[i + 1] += rows[i];
    }

    CsrMatrix {
        row_dim: n,
        col_dim: n,
        rows,
        cols,
        vals,
    }
}
