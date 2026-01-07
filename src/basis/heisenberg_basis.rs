use crate::basis::HilbertBasis;
use crate::model::HeisenbergModel;
use ahash::AHashMap;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct HeisenbergBasis {
    pub model: HeisenbergModel,
    pub total_sz: f64,
    pub total_sz2: i32,
    pub basis: Vec<i128>,
    pub inverse_basis: AHashMap<i128, usize>,
    pub site_base: Vec<i128>,
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

        let mut site_stride: i128 = 1;
        for &two_s in model.two_s_list.iter() {
            site_base.push(site_stride);
            let local_dim = (two_s as i128) + 1;
            site_stride = site_stride
                .checked_mul(local_dim)
                .ok_or_else(|| anyhow::anyhow!("i128 overflow"))?;
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
            site_base: &[i128],
            suffix_min_sz2: &[i32],
            suffix_max_sz2: &[i32],
            total_sz2_target: i32,
            sz2_sum: i32,
            basis_code: i128,
            out: &mut Vec<i128>,
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
                let digit = ((two_s - sz2) / 2) as i128;
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
            total_sz,
            basis,
            inverse_basis,
            site_base,
            total_sz2,
        })
    }

    /// Return the local basis index (digit) at `site` from packed `basis`.
    /// Convention matches the encoder in `dfs`:
    /// digit = (two_s - sz2)/2, i.e. digit 0 corresponds to m=+S.
    pub fn find_local_basis(&self, basis: i128, site: usize) -> usize {
        let base = self.site_base[site];
        let local_dim = (self.model.two_s_list[site] as i128) + 1;
        ((basis / base).rem_euclid(local_dim)) as usize
    }
}

impl HilbertBasis for HeisenbergBasis {
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

    #[test]
    fn basis_two_spin_half_two_sites_total_sz_zero_manual() {
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
        assert_eq!(b.inverse_basis.len(), 4);
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

    #[test]
    fn basis_all_up_and_all_down_two_sites_spin_half() {
        let model = HeisenbergModel {
            num_sites: 2,
            two_s_list: vec![1, 1],
            hz_list: vec![0.0, 0.0],
            d_list: vec![0.0, 0.0],
            exchange_xy: std::collections::HashMap::new(),
            exchange_z: std::collections::HashMap::new(),
        };

        let b_up = HeisenbergBasis::new(model.clone(), 1.0).unwrap();
        assert_eq!(b_up.basis, vec![0]);
        assert_eq!(b_up.inverse_basis.get(&0).copied().unwrap(), 0);

        let b_dn = HeisenbergBasis::new(model, -1.0).unwrap();
        assert_eq!(b_dn.basis, vec![3]);
        assert_eq!(b_dn.inverse_basis.get(&3).copied().unwrap(), 0);
    }

    #[test]
    fn basis_all_up_and_all_down_three_sites_mixed_spin() {
        let model = HeisenbergModel {
            num_sites: 3,
            two_s_list: vec![1, 2, 1],
            hz_list: vec![0.0, 0.0, 0.0],
            d_list: vec![0.0, 0.0, 0.0],
            exchange_xy: std::collections::HashMap::new(),
            exchange_z: std::collections::HashMap::new(),
        };

        let b_up = HeisenbergBasis::new(model.clone(), 2.0).unwrap();
        assert_eq!(b_up.basis, vec![0]);
        assert_eq!(b_up.inverse_basis.get(&0).copied().unwrap(), 0);

        let b_dn = HeisenbergBasis::new(model, -2.0).unwrap();
        assert_eq!(b_dn.basis, vec![11]);
        assert_eq!(b_dn.inverse_basis.get(&11).copied().unwrap(), 0);
    }
}
