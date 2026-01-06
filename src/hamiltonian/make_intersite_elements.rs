use super::transition_state_holder::TransitionStateHolder;
use crate::blas::CsrMatrix;

/// Add intersite operator contributions for a pair of sites into holder.vals.
/// This mirrors C++ ED_Make_Intersite_Element.
///
/// Convention:
/// - src_local1/src_local2 are the source local indices at site1/site2 (for the given `basis`)
/// - Iterate over CSR row entries of op1/op2 at those source rows
/// - Destination local indices are the CSR column indices
/// - Destination basis is computed by shifting the digits at site1 and site2
pub fn make_intersite_elements(
    basis: i128,
    site1: usize,
    site2: usize,
    op1: &CsrMatrix,
    op2: &CsrMatrix,
    coeff: f64,
    sign: f64,
    holder: &mut TransitionStateHolder,
) {
    let src_local1 = holder.local_basis[site1];
    let src_local2 = holder.local_basis[site2];

    let base1 = holder.site_base[site1];
    let base2 = holder.site_base[site2];

    let start1 = op1.rows[src_local1];
    let end1 = op1.rows[src_local1 + 1];
    let start2 = op2.rows[src_local2];
    let end2 = op2.rows[src_local2 + 1];

    for i1 in start1..end1 {
        let v1 = op1.vals[i1];
        let delta1 = (op1.cols[i1] as i128 - src_local1 as i128) * base1;

        for i2 in start2..end2 {
            let v2 = op2.vals[i2];
            let v = sign * coeff * v1 * v2;
            if v.abs() < holder.zero_eps {
                continue;
            }

            let delta2 = (op2.cols[i2] as i128 - src_local2 as i128) * base2;
            let transition_basis = basis + delta1 + delta2;

            let entry = holder.vals.entry(transition_basis).or_insert(0.0);
            *entry += v;
        }
    }
}
