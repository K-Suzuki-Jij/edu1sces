//! Utilities for angular momentum calculations with SU(2) symmetry.
//!
//! This module provides precomputed lookup tables for Wigner 6j symbols.

use ahash::AHashMap;
use std::cell::RefCell;
use std::sync::Arc;
use wigner_symbols::Wigner6j;

thread_local! {
    static SIXJ_TABLE_CACHE: RefCell<AHashMap<(i32, i32, i32), Arc<Sixj6Table>>> =
        RefCell::new(AHashMap::new());
}

/// Get a cached 6j symbol table for given (a, b, max_two_j) parameters.
///
/// The table is created on first access and cached in thread-local storage
/// for subsequent calls with the same parameters.
pub fn get_cached_sixj_table(a: i32, b: i32, max_two_j: i32) -> Arc<Sixj6Table> {
    SIXJ_TABLE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = (a, b, max_two_j);
        if let Some(table) = cache.get(&key) {
            return Arc::clone(table);
        }
        let table = Arc::new(Sixj6Table::new(a, b, max_two_j));
        cache.insert(key, Arc::clone(&table));
        table
    })
}

/// Dense lookup table for 6j symbols with fixed a, b values.
///
/// Stores `{x, a, j1; b, j2, t}` indexed as `[x][j1][j2][t/2]`.
///
/// All angular momentum values are in units of 2*j (i.e., integers representing
/// twice the angular momentum quantum number).
pub struct Sixj6Table {
    max_j: usize,
    max_t_idx: usize, // max_t / 2
    data: Vec<f64>,   // Flattened 4D array
}

impl Sixj6Table {
    /// Create a new 6j symbol lookup table for given (a, b) and maximum j value.
    ///
    /// # Arguments
    /// * `a` - First fixed angular momentum (2*j format)
    /// * `b` - Second fixed angular momentum (2*j format)
    /// * `max_j` - Maximum j value to precompute (2*j format)
    pub fn new(a: i32, b: i32, max_j: i32) -> Self {
        let max_j = max_j as usize;
        let max_t = (a + b) as usize;
        let max_t_idx = max_t / 2;
        let size = (max_j + 1) * (max_j + 1) * (max_j + 1) * (max_t_idx + 1);
        let mut data = vec![0.0; size];

        for x in 0..=max_j {
            for j1 in 0..=max_j {
                for j2 in 0..=max_j {
                    for t_idx in 0..=max_t_idx {
                        let t = (t_idx * 2) as i32;
                        let idx = Self::index_impl(x, j1, j2, t_idx, max_j, max_t_idx);
                        data[idx] = f64::from(
                            Wigner6j {
                                tj1: x as i32,
                                tj2: a,
                                tj3: j1 as i32,
                                tj4: b,
                                tj5: j2 as i32,
                                tj6: t,
                            }
                            .value(),
                        );
                    }
                }
            }
        }

        Self {
            max_j,
            max_t_idx,
            data,
        }
    }

    #[inline]
    fn index_impl(
        x: usize,
        j1: usize,
        j2: usize,
        t_idx: usize,
        max_j: usize,
        max_t_idx: usize,
    ) -> usize {
        ((x * (max_j + 1) + j1) * (max_j + 1) + j2) * (max_t_idx + 1) + t_idx
    }

    /// Get the 6j symbol `{x, a, j1; b, j2, t}`.
    ///
    /// # Arguments
    /// All values are in 2*j format (integers representing twice the angular momentum).
    #[inline]
    pub fn get(&self, x: i32, j1: i32, j2: i32, t: i32) -> f64 {
        let t_idx = (t / 2) as usize;
        let idx = Self::index_impl(
            x as usize,
            j1 as usize,
            j2 as usize,
            t_idx,
            self.max_j,
            self.max_t_idx,
        );
        self.data[idx]
    }
}
