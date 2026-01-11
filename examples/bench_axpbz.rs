use std::time::Instant;

use edu1sces::blas::vector_operation::axpbz_inplace;
use edu1sces::utility::rayon_pool::build_pool;
use rand::Rng;

fn benchmark(
    y: &mut [f64],
    a: f64,
    x: &[f64],
    b: f64,
    z: &[f64],
    num_threads: usize,
    num_iterations: usize,
) -> f64 {
    let pool = build_pool(num_threads).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = axpbz_inplace(&pool, y, a, x, b, z).unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..num_iterations {
        axpbz_inplace(&pool, y, a, x, b, z).unwrap();
    }
    let dt = t0.elapsed();

    let avg_time_ms = dt.as_millis() as f64 / num_iterations as f64;

    std::hint::black_box(y);
    avg_time_ms
}

fn main() {
    let n = 10_000_000_0;
    let num_iterations = 10;
    let thread_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    println!("=== axpbz_inplace Benchmark ===");
    println!("n={}\n", n);

    println!("Generating random vectors...");
    let mut rng = rand::rng();
    let mut y: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let x: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let z: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let a = 2.5;
    let b = 3.7;

    println!("\n--- axpbz_inplace (y += a*x + b*z) ---");
    let mut baseline_time_ms = None;
    for &threads in &thread_counts {
        let time = benchmark(&mut y, a, &x, b, &z, threads, num_iterations);
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
