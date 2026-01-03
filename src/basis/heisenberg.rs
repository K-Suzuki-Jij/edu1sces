use ahash::AHashMap;
use anyhow::{bail, Result};

use crate::model::heisenberg::HeisenbergModel;

#[derive(Debug, Clone)]
pub struct HeisenbergBasis {
    pub model: HeisenbergModel,
    pub total_sz: f64,
    pub total_sz2: i32,
    pub basis: Vec<u128>,
    pub inverse_basis: AHashMap<u128, usize>,
    pub site_base: Vec<u128>,
}

impl HeisenbergBasis {
    pub fn new(model: HeisenbergModel, total_sz: f64) -> Result<Self> {
        let total_sz2 = (2.0 * total_sz).round() as i32;
        if ((2.0 * total_sz) - total_sz2 as f64).abs() > 1e-12 {
            bail!("total_sz must be integer or half-integer");
        }

        let num_sites = model.num_sites;
        let dim = model.calc_dim_u1_sector(total_sz)?;
        let mut site_base = Vec::with_capacity(num_sites);

        let mut site_stride: u128 = 1;
        for &two_s in model.two_s_list.iter() {
            site_base.push(site_stride);
            let local_dim = (two_s as u128) + 1;
            site_stride = site_stride
                .checked_mul(local_dim)
                .ok_or_else(|| anyhow::anyhow!("u128 overflow"))?;
        }

        let mut basis = Vec::with_capacity(dim as usize);

        // Precompute remaining min/max of sum(sz2) for fast pruning
        // suffix_min_sz2[i] = sum_{k=i..} (-two_s_list[k])
        // suffix_max_sz2[i] = sum_{k=i..} ( two_s_list[k])
        let mut suffix_min_sz2 = vec![0; num_sites + 1];
        let mut suffix_max_sz2 = vec![0; num_sites + 1];
        for i in (0..num_sites).rev() {
            let two_s = model.two_s_list[i];
            suffix_min_sz2[i] = suffix_min_sz2[i + 1] - two_s;
            suffix_max_sz2[i] = suffix_max_sz2[i + 1] + two_s;
        }

        fn dfs(
            site: usize,
            two_s_list: &[i32],
            site_base: &[u128],
            suffix_min_sz2: &[i32],
            suffix_max_sz2: &[i32],
            total_sz2_target: i32,
            sz2_sum: i32,
            basis_code: u128,
            out: &mut Vec<u128>,
        ) {
            if site == two_s_list.len() {
                if sz2_sum == total_sz2_target {
                    out.push(basis_code);
                }
                return;
            }

            let two_s = two_s_list[site];
            let min_sz2 = -two_s;
            let max_sz2 = two_s;

            // Remaining range from sites [site+1 ..]
            let remain_min = suffix_min_sz2[site + 1];
            let remain_max = suffix_max_sz2[site + 1];

            // Prune by checking if target is reachable
            let need = total_sz2_target - sz2_sum;
            if need < min_sz2 + remain_min || need > max_sz2 + remain_max {
                return;
            }

            let base = site_base[site];

            let mut sz2 = min_sz2;
            while sz2 <= max_sz2 {
                let digit: u128 = ((sz2 + two_s) / 2) as u128;
                dfs(
                    site + 1,
                    two_s_list,
                    site_base,
                    suffix_min_sz2,
                    suffix_max_sz2,
                    total_sz2_target,
                    sz2_sum + sz2,
                    basis_code + digit * base,
                    out,
                );
                sz2 += 2;
            }
        }

        dfs(
            0,
            &model.two_s_list,
            &site_base,
            &suffix_min_sz2,
            &suffix_max_sz2,
            total_sz2,
            0,
            0u128,
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
            total_sz,
            basis,
            inverse_basis,
            site_base,
            total_sz2,
        })
    }

    pub fn dim(&self) -> usize {
        self.basis.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn basis_two_spin_half_two_sites_total_sz_zero_manual() {
        // two_s_list = [1, 1]
        // local_dim = [2, 2]
        // site_base = [1, 2]
        //
        // digit <-> sz2 mapping for two_s = 1
        // digit 0 -> sz2 = -1
        // digit 1 -> sz2 = +1
        //
        // target total_sz = 0.0 -> total_sz2 = 0
        //
        // valid basis elements
        // site0 digit 1 sz2 +1, site1 digit 0 sz2 -1 -> basis = 1*1 + 0*2 = 1
        // site0 digit 0 sz2 -1, site1 digit 1 sz2 +1 -> basis = 0*1 + 1*2 = 2
        //
        // sorted basis should be [1, 2]
        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![0.0, 0.0],
            d_list: vec![0.0, 0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        let b = HeisenbergBasis::new(model, 0.0).unwrap();

        assert_eq!(b.total_sz2, 0);
        assert_eq!(b.site_base, vec![1, 2]);
        assert_eq!(b.basis, vec![1, 2]);

        assert_eq!(b.inverse_basis.get(&1).copied().unwrap(), 0);
        assert_eq!(b.inverse_basis.get(&2).copied().unwrap(), 1);
        assert_eq!(b.inverse_basis.len(), 2);
    }

    #[test]
    fn basis_mixed_spins_three_sites_total_sz_zero_manual() {
        // two_s_list = [1, 2, 1]
        // local_dim = [2, 3, 2]
        // site_base = [1, 2, 6]
        //
        // digit <-> sz2 mapping
        // site0 two_s=1: digit 0 -> sz2 = -1, digit 1 -> sz2 = +1
        // site1 two_s=2: digit 0 -> sz2 = -2, digit 1 -> sz2 =  0, digit 2 -> sz2 = +2
        // site2 two_s=1: digit 0 -> sz2 = -1, digit 1 -> sz2 = +1
        //
        // target total_sz2 = 0
        //
        // enumerate valid configs by hand
        // site1 digit 2 sz2 +2 -> site0+site2 must be -2 -> digit0=0, digit2=0 -> basis = 0*1 + 2*2 + 0*6 = 4
        // site1 digit 1 sz2  0 -> site0+site2 must be  0 -> two cases
        //   digit0=1, digit2=0 -> basis = 1*1 + 1*2 + 0*6 = 3
        //   digit0=0, digit2=1 -> basis = 0*1 + 1*2 + 1*6 = 8
        // site1 digit 0 sz2 -2 -> site0+site2 must be +2 -> digit0=1, digit2=1 -> basis = 1*1 + 0*2 + 1*6 = 7
        //
        // sorted basis should be [3, 4, 7, 8]
        let model = HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 2, 1],
            hz_list: vec![0.0, 0.0, 0.0],
            d_list: vec![0.0, 0.0, 0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        let b = HeisenbergBasis::new(model, 0.0).unwrap();

        assert_eq!(b.total_sz2, 0);
        assert_eq!(b.site_base, vec![1, 2, 6]);
        assert_eq!(b.basis, vec![3, 4, 7, 8]);

        assert_eq!(b.inverse_basis.get(&3).copied().unwrap(), 0);
        assert_eq!(b.inverse_basis.get(&4).copied().unwrap(), 1);
        assert_eq!(b.inverse_basis.get(&7).copied().unwrap(), 2);
        assert_eq!(b.inverse_basis.get(&8).copied().unwrap(), 3);
        assert_eq!(b.inverse_basis.len(), 4usize);
    }

    #[test]
    fn non_half_integer_total_sz_is_error() {
        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![0.0, 0.0],
            d_list: vec![0.0, 0.0],
            exchange_xy: HashMap::new(),
            exchange_z: HashMap::new(),
        };

        assert!(HeisenbergBasis::new(model, 0.1).is_err());
    }
}
