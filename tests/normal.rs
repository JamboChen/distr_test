use rand_distr::{Distribution, Normal};

// [) intervals, (-inf, -2.25), [-2.25, -2.0), ..., [2.0, 2.25), [2.25, inf)
const ENDPOINTS: [f64; 19] = [
    -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
    1.5, 1.75, 2.0, 2.25,
];

// expected probabilities for each partition
// calculate by R language
const EXPECT_PROB: [f64; 20] = [
    0.01222447, 0.01052566, 0.01730902, 0.02674804, 0.03884257, 0.05300548, 0.06797210, 0.08191019,
    0.09275614, 0.09870633, 0.09870633, 0.09275614, 0.08191019, 0.06797210, 0.05300548, 0.03884257,
    0.02674804, 0.01730902, 0.01052566, 0.01222447,
];

const N_SAMPLES: u64 = 1_000_000;

fn var_to_partition(var: f64) -> usize {
    if var < -2.25 {
        return 0;
    }
    if var >= 2.25 {
        return 19;
    }
    //      0.01 for right edge of partition
    ((var + 0.01 + 2.25) / 0.25).ceil() as usize
}

#[test]
fn normal() {
    const CHI2_19_005: f64 = 30.14353; // df = 20-1 = 19, 0.05 significance level

    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut hist = [0_u64; 20];
    let mut rng = rand::thread_rng();

    for _ in 0..N_SAMPLES {
        let sample = dist.sample(&mut rng);
        let partition = var_to_partition(sample);
        hist[partition] += 1;
    }

    let mut chi2 = 0.0;
    for (partition, &observed) in hist.iter().enumerate() {
        let expected = EXPECT_PROB[partition] * (N_SAMPLES as f64);
        let e = (observed as f64 - expected).powi(2) / expected;
        chi2 += e;
        println!("Expected: {}, Observed: {} => {}", expected, observed, e);
    }

    // Reject H0 if chi2 > 30.14353 (19 degrees of freedom, 0.05 significance level)
    // where H0: the sample is drawn from a standard normal distribution
    assert!(chi2 < CHI2_19_005);
}
