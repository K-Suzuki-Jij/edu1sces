use ahash::AHashMap;
use rand::Rng;
use std::time::Instant;

/// Benchmark AHashMap access speed for Vec<u8> keys vs i128 keys
/// This is to compare performance for SU(2) basis (Vec<u8>) vs U(1) basis (i128)

fn benchmark_vec_u8(
    map: &AHashMap<Vec<u8>, usize>,
    keys: &[Vec<u8>],
    num_iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..3 {
        for key in keys.iter() {
            std::hint::black_box(map.get(key));
        }
    }

    let t0 = Instant::now();
    let mut sum: usize = 0;
    for _ in 0..num_iterations {
        for key in keys.iter() {
            if let Some(&idx) = map.get(key) {
                sum = sum.wrapping_add(idx);
            }
        }
    }
    let dt = t0.elapsed();
    std::hint::black_box(sum);

    dt.as_secs_f64() * 1000.0 / num_iterations as f64
}

fn benchmark_i128(map: &AHashMap<i128, usize>, keys: &[i128], num_iterations: usize) -> f64 {
    // Warmup
    for _ in 0..3 {
        for key in keys.iter() {
            std::hint::black_box(map.get(key));
        }
    }

    let t0 = Instant::now();
    let mut sum: usize = 0;
    for _ in 0..num_iterations {
        for key in keys.iter() {
            if let Some(&idx) = map.get(key) {
                sum = sum.wrapping_add(idx);
            }
        }
    }
    let dt = t0.elapsed();
    std::hint::black_box(sum);

    dt.as_secs_f64() * 1000.0 / num_iterations as f64
}

fn main() {
    let mut rng = rand::rng();

    println!("=== AHashMap Key Access Benchmark ===");
    println!("Comparing Vec<u8> keys (SU(2) basis) vs i128 keys (U(1) basis)");
    println!("Measuring time for 10,000 lookups per iteration\n");

    // Test different key sizes (corresponding to different numbers of sites)
    let key_sizes = [4, 8, 12, 16, 20, 24, 28, 32];
    let map_sizes = [1_000, 10_000, 100_000, 1_000_000];
    let num_iterations = 100;
    let lookup_count = 10_000;

    for &key_size in &key_sizes {
        println!(
            "--- Key size: {} bytes (~ {} sites) ---",
            key_size, key_size
        );

        for &map_size in &map_sizes {
            // Generate random Vec<u8> keys
            let vec_keys: Vec<Vec<u8>> = (0..map_size)
                .map(|_| (0..key_size).map(|_| rng.random_range(0u8..10u8)).collect())
                .collect();

            // Build Vec<u8> -> usize map
            let mut vec_map: AHashMap<Vec<u8>, usize> = AHashMap::with_capacity(map_size);
            for (idx, key) in vec_keys.iter().enumerate() {
                vec_map.insert(key.clone(), idx);
            }

            // Generate corresponding i128 keys (encode Vec<u8> as i128)
            let i128_keys: Vec<i128> = vec_keys
                .iter()
                .map(|v| {
                    let mut val: i128 = 0;
                    for (i, &b) in v.iter().enumerate() {
                        val |= (b as i128) << (i * 8);
                    }
                    val
                })
                .collect();

            // Build i128 -> usize map
            let mut i128_map: AHashMap<i128, usize> = AHashMap::with_capacity(map_size);
            for (idx, &key) in i128_keys.iter().enumerate() {
                i128_map.insert(key, idx);
            }

            // Select random keys to lookup (with replacement)
            let lookup_indices: Vec<usize> = (0..lookup_count)
                .map(|_| rng.random_range(0..map_size))
                .collect();

            let vec_lookup_keys: Vec<Vec<u8>> = lookup_indices
                .iter()
                .map(|&i| vec_keys[i].clone())
                .collect();

            let i128_lookup_keys: Vec<i128> =
                lookup_indices.iter().map(|&i| i128_keys[i]).collect();

            // Benchmark
            let vec_time = benchmark_vec_u8(&vec_map, &vec_lookup_keys, num_iterations);
            let i128_time = benchmark_i128(&i128_map, &i128_lookup_keys, num_iterations);

            let speedup = vec_time / i128_time;

            println!(
                "  map_size={:>7}: Vec<u8>={:6.3}ms, i128={:6.3}ms, i128 is {:.1}x faster",
                map_size, vec_time, i128_time, speedup
            );
        }
        println!();
    }
}
