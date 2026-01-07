// src/bin/bench_make_hamiltonian.rs

use std::collections::HashMap;
use std::time::Instant;

use edu1sces::basis::HeisenbergBasis;
use edu1sces::hamiltonian::heisenberg_hamiltonian::make_heisenberg_hamiltonian_parallel;
use edu1sces::model::HeisenbergModel;

fn build_chain_model(tow_s: i32, n: usize, jxy: f64, jz: f64, hz: f64, d: f64) -> HeisenbergModel {
    let mut exchange_xy = HashMap::new();
    let mut exchange_z = HashMap::new();

    for i in 0..n - 1 {
        exchange_xy.insert((i, i + 1), jxy);
        exchange_z.insert((i, i + 1), jz);
    }

    HeisenbergModel {
        num_sites: n,
        two_s_list: vec![tow_s; n], // S = 1/2
        hz_list: vec![hz; n],
        d_list: vec![d; n],
        exchange_xy,
        exchange_z,
    }
}

fn main() {
    let n = 26; // 本番サイズに変更
    let total_sz = 0.0;
    let lower_only = false;

    let model = build_chain_model(1, n, 1.0, 1.0, 0.3, 0.2);

    let t0 = Instant::now();
    let basis = HeisenbergBasis::new(model.clone(), total_sz).unwrap();
    let dt = t0.elapsed();
    println!("time to build basis = {:?}", dt);

    let t0 = Instant::now();
    let h = make_heisenberg_hamiltonian_parallel(&basis, &model, lower_only, 1).unwrap();
    let dt = t0.elapsed();

    println!("lower_only = {}", lower_only);
    println!("time = {:?}", dt);
    println!("dim = {}", h.row_dim);
    println!("nnz = {}", h.nnz());

    std::hint::black_box(h);
}
