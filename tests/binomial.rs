use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};

const N_SAMPLES: u64 = 1_000_000;

fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    binom(n - 1, k - 1) + binom(n - 1, k)
}

#[test]
fn binom_test() {
    assert_eq!(
        (0..6).map(|k| binom(5, k)).collect::<Vec<_>>(),
        [1, 5, 10, 10, 5, 1]
    );
}

#[test]
fn binomial() {
    const N: u64 = 20; // 21 bins for 0..=20
    const PROB: f64 = 0.5;
    const CHI2_20_005: f64 = 31.41043; // df = 21-1 = 20, 0.05 significance level

    let mut hist = [0; 1 + N as usize];
    let dist = Binomial::new(N, PROB).unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    for _ in 0..N_SAMPLES {
        let sample = dist.sample(&mut rng);
        hist[sample as usize] += 1;
    }

    fn pdf(k: u64) -> f64 {
        binom(N, k) as f64 * PROB.powi(k as i32) * (1.0 - PROB).powi((N - k) as i32)
    }

    let mut chi2 = 0.0;
    for k in 0..=N {
        let expected = pdf(k) * N_SAMPLES as f64;
        let observed = hist[k as usize] as f64;
        chi2 += (observed - expected).powi(2) / expected;
    }

    // Reject H0 if chi2 > 31.41043 (20 degrees of freedom, 0.05 significance level)
    // where H0: the sample is drawn from a binomial distribution with N=20, p=0.3
    assert!(chi2 < CHI2_20_005);
}
