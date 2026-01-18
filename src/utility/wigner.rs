//! Wigner 3j, 6j symbols for angular momentum coupling.
//!
//! All functions use "two_j" convention: arguments are 2*j (integers) to avoid
//! floating-point issues with half-integers.
//!
//! Uses log-factorials for numerical stability with log-exp trick.

use ahash::AHashMap;

/// Check if three angular momenta satisfy the triangle inequality.
/// Arguments are 2*j values (integers).
/// Returns true if |j1 - j2| <= j3 <= j1 + j2 and j1 + j2 + j3 is even.
pub fn triangle_condition(two_j1: i32, two_j2: i32, two_j3: i32) -> bool {
    if two_j1 < 0 || two_j2 < 0 || two_j3 < 0 {
        return false;
    }
    let sum = two_j1 + two_j2;
    let diff = (two_j1 - two_j2).abs();
    if two_j3 < diff || two_j3 > sum {
        return false;
    }
    (sum + two_j3) % 2 == 0
}

/// Check 6j symbol selection rules (four triangle conditions).
fn sixj_selection_ok(
    two_j1: i32,
    two_j2: i32,
    two_j3: i32,
    two_j4: i32,
    two_j5: i32,
    two_j6: i32,
) -> bool {
    triangle_condition(two_j1, two_j2, two_j3)
        && triangle_condition(two_j1, two_j5, two_j6)
        && triangle_condition(two_j4, two_j2, two_j6)
        && triangle_condition(two_j4, two_j5, two_j3)
}

/// Wigner symbol evaluator using Racah formula.
///
/// - Uses doubled integers: two_j = 2*j
/// - Log-factorial table is precomputed at initialization
/// - Uses log-exp trick for numerical stability
/// - 6j symbols are cached per-instance (create one per thread for parallel use)
pub struct WignerSymbols {
    log_fact: Vec<f64>,
    cache_6j: AHashMap<(i32, i32, i32, i32, i32, i32), f64>,
}

impl WignerSymbols {
    /// Create a new evaluator supporting 2*j values up to `max_two_j`.
    ///
    /// The log-factorial table will have size `max_two_j + 2` to handle
    /// all factorial indices that can appear in the formulas.
    pub fn new(max_two_j: usize) -> Self {
        let table_size = max_two_j + 2;
        let mut log_fact = vec![0.0; table_size];
        for i in 1..table_size {
            log_fact[i] = log_fact[i - 1] + (i as f64).ln();
        }
        Self {
            log_fact,
            cache_6j: AHashMap::new(),
        }
    }

    /// Return the number of cached 6j symbols
    pub fn cache_size(&self) -> usize {
        self.cache_6j.len()
    }

    /// Access log(n!)
    fn lf(&self, n: i32) -> f64 {
        self.log_fact[n as usize]
    }

    /// Compute log(Δ(a,b,c)) where Δ is the triangle coefficient.
    /// Δ(a,b,c) = sqrt((a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!)
    fn log_delta(&self, two_a: i32, two_b: i32, two_c: i32) -> f64 {
        let t1 = (two_a + two_b - two_c) / 2;
        let t2 = (two_a - two_b + two_c) / 2;
        let t3 = (-two_a + two_b + two_c) / 2;
        let t4 = (two_a + two_b + two_c) / 2 + 1;

        0.5 * (self.lf(t1) + self.lf(t2) + self.lf(t3) - self.lf(t4))
    }

    /// Wigner 6j symbol using Racah formula (with caching).
    ///
    /// ```text
    /// { j1  j2  j3 }
    /// { j4  j5  j6 }
    /// ```
    ///
    /// Arguments are 2*j values (integers).
    /// Returns 0.0 if any triangle condition is not satisfied.
    pub fn wigner_6j(
        &mut self,
        two_j1: i32,
        two_j2: i32,
        two_j3: i32,
        two_j4: i32,
        two_j5: i32,
        two_j6: i32,
    ) -> f64 {
        let key = (two_j1, two_j2, two_j3, two_j4, two_j5, two_j6);

        // Check cache first
        if let Some(val) = self.cache_6j.get(&key) {
            return *val;
        }

        // Compute the value
        let result = self.wigner_6j_uncached(two_j1, two_j2, two_j3, two_j4, two_j5, two_j6);

        // Store in cache
        self.cache_6j.insert(key, result);

        result
    }

    /// Wigner 6j symbol computation (uncached).
    fn wigner_6j_uncached(
        &self,
        two_j1: i32,
        two_j2: i32,
        two_j3: i32,
        two_j4: i32,
        two_j5: i32,
        two_j6: i32,
    ) -> f64 {
        // Early return for negative arguments
        if two_j1 < 0 || two_j2 < 0 || two_j3 < 0 || two_j4 < 0 || two_j5 < 0 || two_j6 < 0 {
            return 0.0;
        }

        // Special cases when one argument is zero:
        // {j1, j2, j3; j4, j5, 0} requires j1=j5 and j2=j4
        // Similar for other zero positions by symmetry
        if two_j3 == 0 && (two_j1 != two_j2 || two_j4 != two_j5) {
            return 0.0;
        }
        if two_j6 == 0 && (two_j1 != two_j5 || two_j2 != two_j4) {
            return 0.0;
        }

        if !sixj_selection_ok(two_j1, two_j2, two_j3, two_j4, two_j5, two_j6) {
            return 0.0;
        }

        // Prefactor: Δ(j1,j2,j3) * Δ(j4,j5,j3) * Δ(j1,j5,j6) * Δ(j4,j2,j6)
        let log_pref = self.log_delta(two_j1, two_j2, two_j3)
            + self.log_delta(two_j4, two_j5, two_j3)
            + self.log_delta(two_j1, two_j5, two_j6)
            + self.log_delta(two_j4, two_j2, two_j6);

        // Sum bounds
        let x1 = (two_j1 + two_j2 + two_j3) / 2;
        let x2 = (two_j4 + two_j5 + two_j3) / 2;
        let x3 = (two_j1 + two_j5 + two_j6) / 2;
        let x4 = (two_j4 + two_j2 + two_j6) / 2;

        let y1 = (two_j1 + two_j2 + two_j4 + two_j5) / 2;
        let y2 = (two_j1 + two_j4 + two_j3 + two_j6) / 2;
        let y3 = (two_j2 + two_j5 + two_j3 + two_j6) / 2;

        let z_min = x1.max(x2).max(x3).max(x4);
        let z_max = y1.min(y2).min(y3);

        if z_min > z_max {
            return 0.0;
        }

        // Collect terms with log-exp trick
        let mut terms: Vec<(f64, f64)> = Vec::with_capacity((z_max - z_min + 1) as usize);
        let mut log_max = f64::NEG_INFINITY;

        for z in z_min..=z_max {
            let sign = if z % 2 == 0 { 1.0 } else { -1.0 };

            let log_term = self.lf(z + 1)
                - self.lf(z - x1)
                - self.lf(z - x2)
                - self.lf(z - x3)
                - self.lf(z - x4)
                - self.lf(y1 - z)
                - self.lf(y2 - z)
                - self.lf(y3 - z);

            log_max = log_max.max(log_term);
            terms.push((sign, log_term));
        }

        // Sum with log-exp trick
        let sum: f64 = terms
            .iter()
            .map(|(sign, log_term)| sign * (log_term - log_max).exp())
            .sum();

        log_pref.exp() * log_max.exp() * sum
    }

    /// Wigner 3j symbol.
    ///
    /// ```text
    /// ( j1  j2  j3 )
    /// ( m1  m2  m3 )
    /// ```
    ///
    /// Arguments are 2*j and 2*m values (integers).
    /// Returns 0.0 if selection rules are violated.
    pub fn wigner_3j(
        &self,
        two_j1: i32,
        two_j2: i32,
        two_j3: i32,
        two_m1: i32,
        two_m2: i32,
        two_m3: i32,
    ) -> f64 {
        // Selection rules
        if !triangle_condition(two_j1, two_j2, two_j3) {
            return 0.0;
        }
        if two_m1 + two_m2 + two_m3 != 0 {
            return 0.0;
        }
        if two_m1.abs() > two_j1 || two_m2.abs() > two_j2 || two_m3.abs() > two_j3 {
            return 0.0;
        }
        if (two_j1 + two_m1) % 2 != 0 || (two_j2 + two_m2) % 2 != 0 || (two_j3 + two_m3) % 2 != 0 {
            return 0.0;
        }

        // Prefactor
        let n1 = (two_j1 + two_j2 - two_j3) / 2;
        let n2 = (two_j1 - two_j2 + two_j3) / 2;
        let n3 = (-two_j1 + two_j2 + two_j3) / 2;
        let n4 = (two_j1 + two_j2 + two_j3) / 2 + 1;
        let n5 = (two_j1 + two_m1) / 2;
        let n6 = (two_j1 - two_m1) / 2;
        let n7 = (two_j2 + two_m2) / 2;
        let n8 = (two_j2 - two_m2) / 2;
        let n9 = (two_j3 + two_m3) / 2;
        let n10 = (two_j3 - two_m3) / 2;

        let log_prefactor_sq = self.lf(n1) + self.lf(n2) + self.lf(n3) - self.lf(n4)
            + self.lf(n5)
            + self.lf(n6)
            + self.lf(n7)
            + self.lf(n8)
            + self.lf(n9)
            + self.lf(n10);

        // Sum bounds
        let t_min = 0
            .max((two_j2 - two_j3 - two_m1) / 2)
            .max((two_j1 - two_j3 + two_m2) / 2);
        let t_max = ((two_j1 + two_j2 - two_j3) / 2)
            .min((two_j1 - two_m1) / 2)
            .min((two_j2 + two_m2) / 2);

        if t_min > t_max {
            return 0.0;
        }

        // Collect terms with log-exp trick
        let mut terms: Vec<(f64, f64)> = Vec::with_capacity((t_max - t_min + 1) as usize);
        let mut log_max = f64::NEG_INFINITY;

        for t in t_min..=t_max {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };

            let a1 = t;
            let a2 = (two_j1 + two_j2 - two_j3) / 2 - t;
            let a3 = (two_j1 - two_m1) / 2 - t;
            let a4 = (two_j2 + two_m2) / 2 - t;
            let a5 = (two_j3 - two_j2 + two_m1) / 2 + t;
            let a6 = (two_j3 - two_j1 - two_m2) / 2 + t;

            let log_term =
                -self.lf(a1) - self.lf(a2) - self.lf(a3) - self.lf(a4) - self.lf(a5) - self.lf(a6);

            log_max = log_max.max(log_term);
            terms.push((sign, log_term));
        }

        // Sum with log-exp trick
        let sum: f64 = terms
            .iter()
            .map(|(sign, log_term)| sign * (log_term - log_max).exp())
            .sum();

        // Phase: (-1)^(j1 - j2 - m3)
        let phase_exp = (two_j1 - two_j2 - two_m3) / 2;
        let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

        phase * (0.5 * log_prefactor_sq).exp() * log_max.exp() * sum
    }

    /// Clebsch-Gordan coefficient <j1 m1; j2 m2 | j3 m3>.
    ///
    /// Related to 3j symbol by:
    /// <j1 m1; j2 m2 | j3 m3> = (-1)^(j1 - j2 + m3) * sqrt(2*j3 + 1) * (j1  j2   j3 )
    ///                                                                  (m1  m2  -m3)
    ///
    /// Arguments are 2*j and 2*m values.
    pub fn clebsch_gordan(
        &self,
        two_j1: i32,
        two_m1: i32,
        two_j2: i32,
        two_m2: i32,
        two_j3: i32,
        two_m3: i32,
    ) -> f64 {
        if two_m1 + two_m2 != two_m3 {
            return 0.0;
        }

        let threej = self.wigner_3j(two_j1, two_j2, two_j3, two_m1, two_m2, -two_m3);

        if threej == 0.0 {
            return 0.0;
        }

        let phase_exp = (two_j1 - two_j2 + two_m3) / 2;
        let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };
        let dim = (two_j3 + 1) as f64;

        phase * dim.sqrt() * threej
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn make_wigner() -> WignerSymbols {
        WignerSymbols::new(255)
    }

    #[test]
    fn test_triangle_condition() {
        assert!(triangle_condition(2, 2, 2));
        assert!(triangle_condition(1, 1, 0));
        assert!(triangle_condition(1, 1, 2));
        assert!(triangle_condition(2, 4, 4));

        assert!(!triangle_condition(2, 2, 6));
        assert!(!triangle_condition(0, 0, 2));
        assert!(!triangle_condition(1, 1, 1));
        assert!(!triangle_condition(2, 2, 1));
    }

    #[test]
    fn test_6j_zeros() {
        let mut w = make_wigner();
        // Triangle condition violation
        assert_eq!(w.wigner_6j(2, 2, 6, 2, 2, 2), 0.0);

        // Negative argument
        assert_eq!(w.wigner_6j(-1, 1, 2, 1, 1, 2), 0.0);

        // j3=0 but j1≠j2
        assert_eq!(w.wigner_6j(2, 4, 0, 2, 2, 2), 0.0);

        // j6=0 but j1≠j5
        assert_eq!(w.wigner_6j(2, 2, 2, 2, 4, 0), 0.0);

        // j6=0 but j2≠j4
        assert_eq!(w.wigner_6j(2, 2, 2, 4, 2, 0), 0.0);
    }

    #[test]
    fn test_6j_known_values() {
        let mut w = make_wigner();

        // {1/2, 1/2, 0; 1/2, 1/2, 1} = 1/2
        let val = w.wigner_6j(1, 1, 0, 1, 1, 2);
        assert!((val - 0.5).abs() < TOL, "Expected 0.5, got {}", val);

        // {1/2, 1/2, 1; 1/2, 1/2, 0} = 1/2
        let val = w.wigner_6j(1, 1, 2, 1, 1, 0);
        assert!((val - 0.5).abs() < TOL, "Expected 0.5, got {}", val);

        // {1/2, 1/2, 1; 1/2, 1/2, 1} = 1/6
        let val = w.wigner_6j(1, 1, 2, 1, 1, 2);
        assert!((val - 1.0 / 6.0).abs() < TOL, "Expected 1/6, got {}", val);

        // {1, 1, 0; 1, 1, 1} = -1/3
        let val = w.wigner_6j(2, 2, 0, 2, 2, 2);
        assert!(
            (val - (-1.0 / 3.0)).abs() < TOL,
            "Expected -1/3, got {}",
            val
        );

        // {1, 1, 1; 1, 1, 1} = 1/6
        let val = w.wigner_6j(2, 2, 2, 2, 2, 2);
        assert!((val - 1.0 / 6.0).abs() < TOL, "Expected 1/6, got {}", val);

        // {1, 1, 2; 1, 1, 1} = 1/6
        let val = w.wigner_6j(2, 2, 4, 2, 2, 2);
        assert!((val - 1.0 / 6.0).abs() < TOL, "Expected 1/6, got {}", val);

        // {1, 1, 2; 1, 1, 2} = 1/30
        let val = w.wigner_6j(2, 2, 4, 2, 2, 4);
        assert!((val - 1.0 / 30.0).abs() < TOL, "Expected 1/30, got {}", val);
    }

    #[test]
    fn test_6j_larger_spins() {
        let mut w = make_wigner();

        // {3/2, 1, 1/2; 1, 3/2, 2} = -sqrt(6)/12
        let val = w.wigner_6j(3, 2, 1, 2, 3, 4);
        let expected = -(6.0_f64.sqrt()) / 12.0;
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );

        // {2, 2, 2; 2, 2, 2} = -3/70
        let val = w.wigner_6j(4, 4, 4, 4, 4, 4);
        let expected = -3.0 / 70.0;
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );

        // {2, 2, 3; 2, 2, 1} ≈ 0 (cancellation in sum)
        let val = w.wigner_6j(4, 4, 6, 4, 4, 2);
        assert!(val.abs() < TOL, "Expected ~0, got {}", val);

        // {5/2, 3/2, 2; 3/2, 5/2, 1} = -13*sqrt(14)/420
        let val = w.wigner_6j(5, 3, 4, 3, 5, 2);
        let expected = -13.0 * 14.0_f64.sqrt() / 420.0;
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_6j_symmetry() {
        let mut w = make_wigner();

        let v1 = w.wigner_6j(2, 4, 4, 2, 2, 4);
        let v2 = w.wigner_6j(4, 2, 4, 2, 2, 4);
        assert!((v1 - v2).abs() < TOL);

        let v3 = w.wigner_6j(2, 2, 4, 2, 4, 4);
        assert!((v1 - v3).abs() < TOL);
    }

    #[test]
    fn test_clebsch_gordan_known_values() {
        let w = make_wigner();

        // <1/2, 1/2; 1/2, -1/2 | 1, 0> = 1/sqrt(2)
        let val = w.clebsch_gordan(1, 1, 1, -1, 2, 0);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );

        // <1/2, 1/2; 1/2, -1/2 | 0, 0> = 1/sqrt(2)
        let val = w.clebsch_gordan(1, 1, 1, -1, 0, 0);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );

        // <1/2, 1/2; 1/2, 1/2 | 1, 1> = 1
        let val = w.clebsch_gordan(1, 1, 1, 1, 2, 2);
        assert!((val - 1.0).abs() < TOL, "Expected 1, got {}", val);

        // <1/2, -1/2; 1/2, 1/2 | 1, 0> = 1/sqrt(2)
        let val = w.clebsch_gordan(1, -1, 1, 1, 2, 0);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );

        // <1/2, -1/2; 1/2, 1/2 | 0, 0> = -1/sqrt(2)
        let val = w.clebsch_gordan(1, -1, 1, 1, 0, 0);
        let expected = -1.0 / 2.0_f64.sqrt();
        assert!(
            (val - expected).abs() < TOL,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_3j_orthogonality() {
        let w = make_wigner();

        // For j1=j2=1/2, j3=0, m3=0
        let mut sum = 0.0;
        for two_m1 in [-1, 1] {
            let two_m2 = -two_m1;
            let val = w.wigner_3j(1, 1, 0, two_m1, two_m2, 0);
            sum += val * val;
        }
        assert!((sum - 1.0).abs() < TOL, "Expected 1, got {}", sum);

        // For j3=1, m3=0
        sum = 0.0;
        for two_m1 in [-1, 1] {
            let two_m2 = -two_m1;
            let val = w.wigner_3j(1, 1, 2, two_m1, two_m2, 0);
            sum += val * val;
        }
        assert!((sum - 1.0 / 3.0).abs() < TOL, "Expected 1/3, got {}", sum);
    }
}
