use std::time::Instant;

use edu1sces::blas::vector_operation::{norm2, normalize};
use edu1sces::utility::rayon_pool::build_pool;
use rand::Rng;

fn benchmark(x: &mut [f64], num_threads: usize, num_iterations: usize) -> f64 {
    let pool = build_pool(num_threads).unwrap();

    // Warmup
    for _ in 0..3 {
        let n = norm2(&pool, x).unwrap();
        let _ = normalize(&pool, x, n).unwrap();
        // Reset x to avoid underflow
        for v in x.iter_mut() {
            *v = 1.0;
        }
    }

    let t0 = Instant::now();
    for _ in 0..num_iterations {
        let n = norm2(&pool, x).unwrap();
        normalize(&pool, x, n).unwrap();
        // Reset x to avoid underflow
        for v in x.iter_mut() {
            *v = 1.0;
        }
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(x);
    avg_time_ms
}

fn main() {
    let n = 10_000_000_0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    println!("=== Normalize Benchmark ===");
    println!("n={}\n", n);

    println!("Generating random vector...");
    let mut rng = rand::rng();
    let mut x: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

    println!("\n--- normalize ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&mut x, threads, num_iterations);
        if let Some(base) = baseline_time_ms {
            let speedup = base / time;
            println!(
                "  threads={:2}, avg_time={:8.2}ms, speedup={:.2}x",
                threads, time, speedup
            );
        } else {
            println!("  threads={:2}, avg_time={:8.2}ms", threads, time);
            baseline_time_ms = Some(time);
        }
    }
}
