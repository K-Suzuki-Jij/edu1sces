use crate::basis::HilbertBasis;
use crate::model::HubbardModel;
use ahash::AHashMap;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct HubbardBasis {
    pub model: HubbardModel,
    pub num_electrons: usize,
    pub total_sz: f64,
    pub total_sz2: i32,
    pub basis: Vec<i128>,
    pub inverse_basis: AHashMap<i128, usize>,
    pub site_base: Vec<i128>,
}

impl HubbardBasis {
    fn to_total_sz2(num_electrons: usize, total_sz: f64) -> Result<i32> {
        let two_m_f = 2.0 * total_sz;
        let two_m = two_m_f.round() as i32;

        if (two_m_f - two_m as f64).abs() > 1e-12 {
            bail!("total_sz must be integer or half-integer (got {total_sz})");
        }

        let n = num_electrons as i32;
        // N_up = (N + 2Sz)/2, N_dn = (N - 2Sz)/2 must be non-negative integers
        if ((n + two_m) & 1) != 0 || ((n - two_m) & 1) != 0 {
            bail!(
                "parity mismatch: num_electrons = {} but 2*total_sz = {}",
                num_electrons,
                two_m
            );
        }

        Ok(two_m)
    }

    pub fn new(model: HubbardModel, num_electrons: usize, target_total_sz: f64) -> Result<Self> {
        let total_sz2 = Self::to_total_sz2(num_electrons, target_total_sz)?;
        let total_sz = total_sz2 as f64 / 2.0;

        let num_sites = model.num_sites;
        let dim = model.calc_dim_u1_sector(num_electrons, total_sz)?;

        if dim < 0 {
            bail!("internal error: negative dim");
        }

        // site_base: each site has 4 states (|vac>, |up>, |dn>, |updn>)
        let mut site_base = Vec::with_capacity(num_sites);
        let mut stride: i128 = 1;
        for _ in 0..num_sites {
            site_base.push(stride);
            stride = stride
                .checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("i128 overflow"))?;
        }

        let mut basis = Vec::with_capacity(dim as usize);

        // Local state encoding:
        // 0 -> |vac>  (n=0, sz2=0)
        // 1 -> |up>   (n=1, sz2=+1)
        // 2 -> |dn>   (n=1, sz2=-1)
        // 3 -> |updn> (n=2, sz2=0)

        // Precompute suffix bounds for pruning
        // Each site can contribute 0..2 electrons and -1..+1 to 2*Sz
        // suffix_min_n[i] = min electrons from sites [i..]  = 0
        // suffix_max_n[i] = max electrons from sites [i..]  = 2 * (num_sites - i)
        // suffix_min_sz2[i] = min 2*Sz from sites [i..] = -(num_sites - i)
        // suffix_max_sz2[i] = max 2*Sz from sites [i..] = +(num_sites - i)
        let mut suffix_min_n = vec![0i32; num_sites + 1];
        let mut suffix_max_n = vec![0i32; num_sites + 1];
        let mut suffix_min_sz2 = vec![0i32; num_sites + 1];
        let mut suffix_max_sz2 = vec![0i32; num_sites + 1];
        for i in (0..num_sites).rev() {
            suffix_min_n[i] = suffix_min_n[i + 1]; // min contribution = 0
            suffix_max_n[i] = suffix_max_n[i + 1] + 2; // max contribution = 2
            suffix_min_sz2[i] = suffix_min_sz2[i + 1] - 1; // min sz2 = -1
            suffix_max_sz2[i] = suffix_max_sz2[i + 1] + 1; // max sz2 = +1
        }

        fn dfs(
            site: usize,
            num_sites: usize,
            site_base: &[i128],
            suffix_min_n: &[i32],
            suffix_max_n: &[i32],
            suffix_min_sz2: &[i32],
            suffix_max_sz2: &[i32],
            target_n: i32,
            target_sz2: i32,
            n_sum: i32,
            sz2_sum: i32,
            basis_packed: i128,
            out: &mut Vec<i128>,
        ) {
            if site == num_sites {
                if n_sum == target_n && sz2_sum == target_sz2 {
                    out.push(basis_packed);
                }
                return;
            }

            const DN: [i32; 4] = [0, 1, 1, 2];
            const DSZ2: [i32; 4] = [0, 1, -1, 0];

            let remain_min_n = suffix_min_n[site + 1];
            let remain_max_n = suffix_max_n[site + 1];
            let remain_min_sz2 = suffix_min_sz2[site + 1];
            let remain_max_sz2 = suffix_max_sz2[site + 1];

            let base = site_base[site];

            for d in 0..4 {
                let nn = n_sum + DN[d];
                let ss = sz2_sum + DSZ2[d];

                let need_n = target_n - nn;
                let need_sz2 = target_sz2 - ss;

                // Pruning: check if target is reachable from remaining sites
                if need_n < remain_min_n || need_n > remain_max_n {
                    continue;
                }
                if need_sz2 < remain_min_sz2 || need_sz2 > remain_max_sz2 {
                    continue;
                }

                dfs(
                    site + 1,
                    num_sites,
                    site_base,
                    suffix_min_n,
                    suffix_max_n,
                    suffix_min_sz2,
                    suffix_max_sz2,
                    target_n,
                    target_sz2,
                    nn,
                    ss,
                    basis_packed + (d as i128) * base,
                    out,
                );
            }
        }

        let target_n = num_electrons as i32;
        dfs(
            0,
            num_sites,
            &site_base,
            &suffix_min_n,
            &suffix_max_n,
            &suffix_min_sz2,
            &suffix_max_sz2,
            target_n,
            total_sz2,
            0,
            0,
            0i128,
            &mut basis,
        );

        if basis.len() != dim as usize {
            bail!(
                "internal error: basis.len() = {}, expected dim = {}",
                basis.len(),
                dim
            );
        }

        basis.sort_unstable();

        let mut inverse_basis = AHashMap::with_capacity(basis.len());
        for (i, &basis_code) in basis.iter().enumerate() {
            inverse_basis.insert(basis_code, i);
        }

        Ok(Self {
            model,
            num_electrons,
            total_sz,
            total_sz2,
            basis,
            inverse_basis,
            site_base,
        })
    }

    /// Return the local basis index (digit) at `site` from packed `basis`.
    /// 0 -> |vac>, 1 -> |up>, 2 -> |dn>, 3 -> |updn>
    pub fn find_local_basis(&self, basis: i128, site: usize) -> usize {
        let base = self.site_base[site];
        ((basis / base).rem_euclid(4)) as usize
    }
}

impl HilbertBasis for HubbardBasis {
    #[inline]
    fn dim(&self) -> usize {
        self.basis.len()
    }

    #[inline]
    fn basis_state_at(&self, row: usize) -> i128 {
        self.basis[row]
    }

    #[inline]
    fn inverse_basis(&self) -> &ahash::AHashMap<i128, usize> {
        &self.inverse_basis
    }

    #[inline]
    fn site_base(&self) -> &[i128] {
        &self.site_base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_model(num_sites: usize) -> HubbardModel {
        HubbardModel::new(
            HashMap::new(),
            vec![1.0; num_sites],
            vec![0.0; num_sites],
            vec![0.0; num_sites],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn basis_two_sites_two_electrons_sz_zero() {
        // L=2, N=2, Sz=0 => n_up=1, n_dn=1 => dim = C(2,1)*C(2,1) = 4
        let model = make_model(2);
        let b = HubbardBasis::new(model, 2, 0.0).unwrap();

        assert_eq!(b.total_sz2, 0);
        assert_eq!(b.num_electrons, 2);
        assert_eq!(b.site_base, vec![1, 4]);
        assert_eq!(b.dim(), 4);

        // States with n_up=1, n_dn=1:
        // site0=|up>, site1=|dn> => 1 + 2*4 = 9
        // site0=|dn>, site1=|up> => 2 + 1*4 = 6
        // site0=|updn>, site1=|vac> => 3 + 0*4 = 3
        // site0=|vac>, site1=|updn> => 0 + 3*4 = 12
        assert_eq!(b.basis, vec![3, 6, 9, 12]);
    }

    #[test]
    fn basis_two_sites_two_electrons_sz_one() {
        // L=2, N=2, Sz=1 => n_up=2, n_dn=0 => dim = C(2,2)*C(2,0) = 1
        let model = make_model(2);
        let b = HubbardBasis::new(model, 2, 1.0).unwrap();

        assert_eq!(b.total_sz2, 2);
        assert_eq!(b.dim(), 1);

        // Both sites have |up>: 1 + 1*4 = 5
        assert_eq!(b.basis, vec![5]);
    }

    #[test]
    fn basis_two_sites_zero_electrons() {
        // L=2, N=0, Sz=0 => dim = 1
        let model = make_model(2);
        let b = HubbardBasis::new(model, 0, 0.0).unwrap();

        assert_eq!(b.dim(), 1);
        assert_eq!(b.basis, vec![0]); // both sites |vac>
    }

    #[test]
    fn basis_two_sites_four_electrons() {
        // L=2, N=4, Sz=0 => n_up=2, n_dn=2 => dim = C(2,2)*C(2,2) = 1
        let model = make_model(2);
        let b = HubbardBasis::new(model, 4, 0.0).unwrap();

        assert_eq!(b.dim(), 1);
        // Both sites have |updn>: 3 + 3*4 = 15
        assert_eq!(b.basis, vec![15]);
    }

    #[test]
    fn basis_four_sites_half_filling_sz_zero() {
        // L=4, N=4, Sz=0 => n_up=2, n_dn=2 => dim = C(4,2)*C(4,2) = 36
        let model = make_model(4);
        let b = HubbardBasis::new(model, 4, 0.0).unwrap();

        assert_eq!(b.dim(), 36);
    }

    #[test]
    fn find_local_basis_works() {
        let model = make_model(2);
        let b = HubbardBasis::new(model, 2, 0.0).unwrap();

        // basis_code = 9 means site0=|up>(1), site1=|dn>(2)
        assert_eq!(b.find_local_basis(9, 0), 1); // |up>
        assert_eq!(b.find_local_basis(9, 1), 2); // |dn>

        // basis_code = 3 means site0=|updn>(3), site1=|vac>(0)
        assert_eq!(b.find_local_basis(3, 0), 3); // |updn>
        assert_eq!(b.find_local_basis(3, 1), 0); // |vac>
    }

    #[test]
    fn parity_mismatch_error() {
        let model = make_model(2);
        // N=3, Sz=0 is impossible (parity mismatch)
        assert!(HubbardBasis::new(model, 3, 0.0).is_err());
    }
}
