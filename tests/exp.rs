use rand::prelude::*;
use rand_distr::{Distribution, Exp1};

const N_SAMPLES: u64 = 1_000_000;
const ENDPOINTS: [f64; 19] = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25,
    4.5, 4.75,
]; // [) intervals, [0.0, 0.25), [0.25, 0.5), ..., [4.75, inf)

// cdf for exp(1) distribution
fn cdf(x: f64) -> f64 {
    1.0 - (-x).exp()
}

fn var_to_partition(var: f64) -> usize {
    if var >= 4.75 {
        return 19;
    }
    //      0.01 for right edge of partition
    ((var + 0.01 - 0.25) / 0.25).ceil() as usize
}

#[test]
fn exp_test() {
    const CHI2_19_005: f64 = 30.14353; // df = 20-1 = 19, 0.05 significance level

    let mut expect_prob = [0.0; 20];
    expect_prob[0] = cdf(ENDPOINTS[0]);
    for i in 1..19 {
        expect_prob[i] = cdf(ENDPOINTS[i]) - cdf(ENDPOINTS[i - 1]);
    }
    expect_prob[19] = 1.0 - cdf(ENDPOINTS[18]);

    let dist = Exp1;
    let mut hist = [0_u64; 20];
    let mut rng = rand_pcg::Pcg32::seed_from_u64(42);

    for _ in 0..N_SAMPLES {
        let sample = dist.sample(&mut rng);
        let partition = var_to_partition(sample);
        hist[partition] += 1;
    }

    let mut chi2 = 0.0;
    for (partition, &observed) in hist.iter().enumerate() {
        let expected = expect_prob[partition] * (N_SAMPLES as f64);
        let e = (observed as f64 - expected).powi(2) / expected;
        chi2 += e;
        println!("Expected: {}, Observed: {} => {}", expected, observed, e);
    }

    // Reject H0 if chi2 > 30.14353 (19 degrees of freedom, 0.05 significance level)
    // where H0: the sample is drawn from a standard normal distribution
    assert!(chi2 < CHI2_19_005);
}
