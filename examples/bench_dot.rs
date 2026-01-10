use std::time::Instant;

use edu1sces::blas::vector_operation::dot;
use edu1sces::utility::rayon_pool::build_pool;
use rand::Rng;

fn benchmark(x: &[f64], y: &[f64], num_threads: usize, num_iterations: usize) -> f64 {
    let pool = build_pool(num_threads).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = dot(&pool, x, y).unwrap();
    }

    let t0 = Instant::now();
    let mut result = 0.0;
    for _ in 0..num_iterations {
        result = dot(&pool, x, y).unwrap();
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(result);
    avg_time_ms
}

fn main() {
    let n = 10_000_000_0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6];

    println!("=== Dot Product Benchmark ===");
    println!("n={}\n", n);

    println!("Generating random vectors...");
    let mut rng = rand::rng();
    let x: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

    println!("\n--- dot ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&x, &y, threads, num_iterations);
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
