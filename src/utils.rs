use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};

pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn std_dev(values: &[f64], ddof: usize) -> f64 {
    if values.len() <= ddof {
        return 0.0;
    }
    let mu = mean(values);
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() - ddof) as f64;
    var.sqrt()
}

pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if v.len() == 1 {
        return v[0];
    }

    let rank = (p / 100.0) * (v.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;

    if low == high {
        v[low]
    } else {
        let w = rank - low as f64;
        v[low] * (1.0 - w) + v[high] * w
    }
}

pub fn max_drawdown(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut running_max = values[0];
    let mut min_drawdown = 0.0;

    for value in values {
        if *value > running_max {
            running_max = *value;
        }
        if running_max > 0.0 {
            let drawdown = (*value - running_max) / running_max;
            if drawdown < min_drawdown {
                min_drawdown = drawdown;
            }
        }
    }

    min_drawdown
}

pub fn sample_normal(mu: f64, sigma: f64, n: usize, rng: &mut StdRng) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![mu; n];
    }

    (0..n)
        .map(|_| {
            let z: f64 = StandardNormal.sample(rng);
            mu + sigma * z
        })
        .collect()
}
