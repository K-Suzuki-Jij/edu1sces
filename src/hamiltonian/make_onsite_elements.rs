use super::transition_state_holder::TransitionStateHolder;
use crate::blas::CsrMatrix;

/// Add onsite operator contributions for a single site into holder.vals.
/// This mirrors C++ ED_Make_Onsite_Element.
///
/// Convention is the same as your current C++ ED:
/// - Use row = local_basis as the source local index
/// - Iterate over CSR row entries
/// - Destination local index is the column index stored in CSR
/// - Destination basis is computed by shifting the digit at `site`
pub fn make_onsite_elements(
    basis: i128,
    site: usize,
    op: &CsrMatrix,
    coeff: f64,
    holder: &mut TransitionStateHolder,
) {
    let src_local = holder.local_basis[site];
    let base = holder.site_base[site];

    let start = op.rows[src_local];
    let end = op.rows[src_local + 1];

    for i in start..end {
        let v = coeff * op.vals[i];

        if v.abs() < holder.zero_eps {
            continue;
        }

        let delta = (op.cols[i] as i128 - src_local as i128) * base;
        let transition_basis = basis + delta;

        let entry = holder.vals.entry(transition_basis).or_insert(0.0);
        *entry += v;
    }
}
