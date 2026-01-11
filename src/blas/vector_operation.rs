use anyhow::{bail, Result};
use rayon::prelude::*;
use rayon::ThreadPool;

pub fn zero(pool: &ThreadPool, x: &mut [f64]) -> Result<()> {
    pool.install(|| {
        x.par_iter_mut().for_each(|v| {
            *v = 0.0;
        });
    });
    Ok(())
}

pub fn copy(pool: &ThreadPool, src: &[f64], dst: &mut [f64]) -> Result<()> {
    if src.len() != dst.len() {
        bail!("dimension mismatch: src.len() != dst.len()");
    }

    pool.install(|| {
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| {
            *d = *s;
        });
    });
    Ok(())
}

/// y += a * x
pub fn axpy_inplace(pool: &ThreadPool, y: &mut [f64], a: f64, x: &[f64]) -> Result<()> {
    if y.len() != x.len() {
        bail!("dimension mismatch: y.len() != x.len()");
    }

    pool.install(|| {
        y.par_iter_mut().zip(x.par_iter()).for_each(|(yy, xx)| {
            *yy += a * (*xx);
        });
    });
    Ok(())
}

/// y += a*x + b*z (in-place update of y with two scaled vectors)
pub fn axpbz_inplace(
    pool: &ThreadPool,
    y: &mut [f64],
    a: f64,
    x: &[f64],
    b: f64,
    z: &[f64],
) -> Result<()> {
    if y.len() != x.len() || y.len() != z.len() {
        bail!("dimension mismatch: y/x/z must have the same length");
    }

    pool.install(|| {
        y.par_iter_mut()
            .zip(x.par_iter())
            .zip(z.par_iter())
            .for_each(|((yy, xx), zz)| {
                *yy += a * (*xx) + b * (*zz);
            });
    });
    Ok(())
}

/// out = a*x + b*y
pub fn axpby(
    pool: &ThreadPool,
    out: &mut [f64],
    a: f64,
    x: &[f64],
    b: f64,
    y: &[f64],
) -> Result<()> {
    if out.len() != x.len() || out.len() != y.len() {
        bail!("dimension mismatch: out/x/y must have the same length");
    }

    pool.install(|| {
        out.par_iter_mut()
            .zip(x.par_iter())
            .zip(y.par_iter())
            .for_each(|((oo, xx), yy)| {
                *oo = a * (*xx) + b * (*yy);
            });
    });
    Ok(())
}

/// x = y + a*x (in-place update of x)
pub fn xpay(pool: &ThreadPool, x: &mut [f64], a: f64, y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        bail!("dimension mismatch: x.len() != y.len()");
    }

    pool.install(|| {
        x.par_iter_mut().zip(y.par_iter()).for_each(|(xx, yy)| {
            *xx = *yy + a * (*xx);
        });
    });
    Ok(())
}

pub fn dot(pool: &ThreadPool, x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        bail!("dimension mismatch: x.len() != y.len()");
    }

    let n = x.len();
    let x_ptr = x.as_ptr() as usize;
    let y_ptr = y.as_ptr() as usize;

    let s = pool.install(|| {
        use rayon::prelude::*;

        // Determine chunk size for parallel processing
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        (0..num_threads)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                if start >= n {
                    return 0.0;
                }
                let end = ((tid + 1) * chunk_size).min(n);

                unsafe {
                    let x = x_ptr as *const f64;
                    let y = y_ptr as *const f64;

                    // 4-way unrolling with independent accumulators
                    let mut sum0 = 0.0;
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;

                    let len = end - start;
                    let unroll_end = start + (len / 4) * 4;

                    let mut i = start;
                    while i < unroll_end {
                        sum0 += *x.add(i) * *y.add(i);
                        sum1 += *x.add(i + 1) * *y.add(i + 1);
                        sum2 += *x.add(i + 2) * *y.add(i + 2);
                        sum3 += *x.add(i + 3) * *y.add(i + 3);
                        i += 4;
                    }

                    let mut sum = sum0 + sum1 + sum2 + sum3;
                    while i < end {
                        sum += *x.add(i) * *y.add(i);
                        i += 1;
                    }

                    sum
                }
            })
            .sum()
    });

    Ok(s)
}

pub fn norm2(pool: &ThreadPool, x: &[f64]) -> Result<f64> {
    let s = dot(pool, x, x)?;
    Ok(s.sqrt())
}

pub fn normalize(pool: &ThreadPool, x: &mut [f64], norm: f64) -> Result<()> {
    if norm == 0.0 {
        bail!("cannot normalize zero vector");
    }

    let inv = 1.0 / norm;
    pool.install(|| {
        x.par_iter_mut().for_each(|v| {
            *v *= inv;
        });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::rayon_pool::build_pool;

    const EPSILON: f64 = 1e-12;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < EPSILON,
            "assertion failed: |{} - {}| = {} >= {}",
            a,
            b,
            (a - b).abs(),
            EPSILON
        );
    }

    fn assert_vec_approx_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "vector lengths differ");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < EPSILON,
                "assertion failed at index {}: |{} - {}| = {} >= {}",
                i,
                x,
                y,
                (x - y).abs(),
                EPSILON
            );
        }
    }

    #[test]
    fn test_zero() {
        let pool = build_pool(2).unwrap();
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        zero(&pool, &mut x).unwrap();
        assert_vec_approx_eq(&x, &vec![0.0; 5]);
    }

    #[test]
    fn test_zero_empty() {
        let pool = build_pool(2).unwrap();
        let mut x: Vec<f64> = vec![];
        zero(&pool, &mut x).unwrap();
        assert!(x.is_empty());
    }

    #[test]
    fn test_copy() {
        let pool = build_pool(2).unwrap();
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        copy(&pool, &src, &mut dst).unwrap();
        assert_vec_approx_eq(&dst, &src);
    }

    #[test]
    fn test_copy_dimension_mismatch() {
        let pool = build_pool(2).unwrap();
        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 4];
        let result = copy(&pool, &src, &mut dst);
        assert!(result.is_err());
    }

    #[test]
    fn test_axpy() {
        let pool = build_pool(2).unwrap();
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![2.0, 3.0, 4.0, 5.0];
        let a = 2.0;
        axpy_inplace(&pool, &mut y, a, &x).unwrap();
        // y = [1, 2, 3, 4] + 2 * [2, 3, 4, 5] = [5, 8, 11, 14]
        assert_vec_approx_eq(&y, &vec![5.0, 8.0, 11.0, 14.0]);
    }

    #[test]
    fn test_axpy_zero_alpha() {
        let pool = build_pool(2).unwrap();
        let mut y = vec![1.0, 2.0, 3.0];
        let x = vec![10.0, 20.0, 30.0];
        axpy_inplace(&pool, &mut y, 0.0, &x).unwrap();
        assert_vec_approx_eq(&y, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_axpy_dimension_mismatch() {
        let pool = build_pool(2).unwrap();
        let mut y = vec![1.0, 2.0];
        let x = vec![1.0, 2.0, 3.0];
        let result = axpy_inplace(&pool, &mut y, 1.0, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_axpby() {
        let pool = build_pool(2).unwrap();
        let mut out = vec![0.0; 4];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        let a = 2.0;
        let b = 3.0;
        axpby(&pool, &mut out, a, &x, b, &y).unwrap();
        // out = 2 * [1, 2, 3, 4] + 3 * [5, 6, 7, 8] = [2, 4, 6, 8] + [15, 18, 21, 24] = [17, 22, 27, 32]
        assert_vec_approx_eq(&out, &vec![17.0, 22.0, 27.0, 32.0]);
    }

    #[test]
    fn test_axpby_zero_coefficients() {
        let pool = build_pool(2).unwrap();
        let mut out = vec![999.0; 3];
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        axpby(&pool, &mut out, 0.0, &x, 0.0, &y).unwrap();
        assert_vec_approx_eq(&out, &vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_axpby_dimension_mismatch() {
        let pool = build_pool(2).unwrap();
        let mut out = vec![0.0; 3];
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = axpby(&pool, &mut out, 1.0, &x, 1.0, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dot() {
        let pool = build_pool(2).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        let result = dot(&pool, &x, &y).unwrap();
        // dot = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_approx_eq(result, 40.0);
    }

    #[test]
    fn test_dot_orthogonal() {
        let pool = build_pool(2).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        let result = dot(&pool, &x, &y).unwrap();
        assert_approx_eq(result, 0.0);
    }

    #[test]
    fn test_dot_dimension_mismatch() {
        let pool = build_pool(2).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = dot(&pool, &x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_norm2() {
        let pool = build_pool(2).unwrap();
        let x = vec![3.0, 4.0];
        let result = norm2(&pool, &x).unwrap();
        // norm = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        assert_approx_eq(result, 5.0);
    }

    #[test]
    fn test_norm2_unit_vector() {
        let pool = build_pool(2).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let result = norm2(&pool, &x).unwrap();
        assert_approx_eq(result, 1.0);
    }

    #[test]
    fn test_norm2_zero_vector() {
        let pool = build_pool(2).unwrap();
        let x = vec![0.0, 0.0, 0.0];
        let result = norm2(&pool, &x).unwrap();
        assert_approx_eq(result, 0.0);
    }

    #[test]
    fn test_normalize() {
        let pool = build_pool(2).unwrap();
        let mut x = vec![3.0, 4.0];
        let n = norm2(&pool, &x).unwrap();
        normalize(&pool, &mut x, n).unwrap();
        // normalized = [3/5, 4/5] = [0.6, 0.8]
        assert_vec_approx_eq(&x, &vec![0.6, 0.8]);
        // verify norm is 1
        let n = norm2(&pool, &x).unwrap();
        assert_approx_eq(n, 1.0);
    }

    #[test]
    fn test_normalize_already_unit() {
        let pool = build_pool(2).unwrap();
        let mut x = vec![1.0, 0.0, 0.0];
        let n = norm2(&pool, &x).unwrap();
        normalize(&pool, &mut x, n).unwrap();
        assert_vec_approx_eq(&x, &vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let pool = build_pool(2).unwrap();
        let mut x = vec![0.0, 0.0, 0.0];
        let result = normalize(&pool, &mut x, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_vector() {
        let pool = build_pool(2).unwrap();
        let n = 10000;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();

        let result = dot(&pool, &x, &y).unwrap();
        // sum of i * (n - i) for i = 0 to n-1
        let expected: f64 = (0..n).map(|i| (i * (n - i)) as f64).sum();
        assert_approx_eq(result, expected);
    }
}
