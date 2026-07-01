use std::collections::HashMap;

use dotenvy::dotenv;

#[derive(Debug, Clone)]
pub struct CityConfig {
    pub lat: f64,
    pub lon: f64,
    pub noaa_station: Option<&'static str>,
    pub nws_station: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct BacktestParams {
    pub lookback_months: i64,
    pub cities: HashMap<&'static str, CityConfig>,
    pub temperature_thresholds_f: Vec<i32>,
    pub edge_threshold: f64,
    pub kelly_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct BayesianModelParams {
    pub temperature_prior_mu: f64,
    pub temperature_prior_sigma: f64,
    pub precipitation_prior_alpha: f64,
    pub precipitation_prior_beta: f64,
    pub mcmc_draws: usize,
    pub tune: usize,
}

#[derive(Debug, Clone)]
pub struct MonteCarloParams {
    pub n_simulations: usize,
    pub random_seed: u64,
    pub volatility_window: usize,
    pub correlation_lookback: usize,
}

#[derive(Debug, Clone)]
pub struct DataPipelineParams {
    pub batch_size: usize,
    pub validation_threshold: f64,
    pub missing_data_threshold: f64,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn database_url() -> String {
    dotenv().ok();
    env_or(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/polymarket_weather",
    )
}

pub fn noaa_api_key() -> String {
    dotenv().ok();
    env_or("NOAA_API_KEY", "")
}

pub fn noaa_base_url() -> String {
    dotenv().ok();
    env_or("NOAA_BASE_URL", "https://www.ncei.noaa.gov/api/access/")
}

pub fn openweathermap_api_key() -> String {
    dotenv().ok();
    env_or("OPENWEATHERMAP_API_KEY", "")
}

pub fn accuweather_api_key() -> String {
    dotenv().ok();
    env_or("ACCUWEATHER_API_KEY", "")
}

pub fn awc_base_url() -> String {
    dotenv().ok();
    env_or("AWC_BASE_URL", "https://aviationweather.gov/api/data")
}

pub fn visual_crossing_api_key() -> String {
    dotenv().ok();
    env_or("VISUAL_CROSSING_API_KEY", "")
}

pub fn weatherapi_key() -> String {
    dotenv().ok();
    env_or("WEATHERAPI_KEY", "")
}

pub fn tomorrow_io_api_key() -> String {
    dotenv().ok();
    env_or("TOMORROW_IO_API_KEY", "")
}

/// Kalshi API host (no path). Defaults to the DEMO/paper environment so nothing touches real money
/// until you point it at prod (`https://api.elections.kalshi.com`).
pub fn kalshi_base_url() -> String {
    dotenv().ok();
    env_or("KALSHI_BASE_URL", "https://demo-api.kalshi.co")
        .trim_end_matches('/')
        .to_string()
}

pub fn kalshi_api_key_id() -> String {
    dotenv().ok();
    env_or("KALSHI_API_KEY_ID", "")
}

/// The RSA private key PEM used to sign Kalshi requests: inline `KALSHI_PRIVATE_KEY_PEM` wins, else
/// read the file at `KALSHI_PRIVATE_KEY_PATH`. Empty when neither is set (Kalshi then silently skipped).
pub fn kalshi_private_key_pem() -> String {
    dotenv().ok();
    let inline = env_or("KALSHI_PRIVATE_KEY_PEM", "");
    if !inline.is_empty() {
        return inline;
    }
    let path = env_or("KALSHI_PRIVATE_KEY_PATH", "");
    if path.is_empty() {
        return String::new();
    }
    std::fs::read_to_string(path).unwrap_or_default()
}

pub fn initial_capital() -> f64 {
    dotenv().ok();
    env_or("INITIAL_CAPITAL", "100000")
        .parse::<f64>()
        .unwrap_or(100000.0)
}

pub fn min_bid_ask_spread() -> f64 {
    dotenv().ok();
    env_or("MIN_BID_ASK_SPREAD", "0.02")
        .parse::<f64>()
        .unwrap_or(0.02)
}

pub fn bayesian_model_params() -> BayesianModelParams {
    BayesianModelParams {
        // Daily-high climatology prior, in degC. 20 degC (~68 degF) is a sane weak prior;
        // sigma=15 keeps it weak so data dominates. (Was 60.0 degC = 140 degF — nonsensical;
        // harmless under long lookbacks but biases mu upward on short windows.)
        temperature_prior_mu: 20.0,
        temperature_prior_sigma: 15.0,
        precipitation_prior_alpha: 1.0,
        precipitation_prior_beta: 1.0,
        mcmc_draws: 2000,
        tune: 1000,
    }
}

pub fn monte_carlo_params() -> MonteCarloParams {
    MonteCarloParams {
        n_simulations: 10_000,
        random_seed: 42,
        volatility_window: 30,
        correlation_lookback: 90,
    }
}

pub fn data_pipeline_params() -> DataPipelineParams {
    DataPipelineParams {
        batch_size: 1000,
        validation_threshold: 0.95,
        missing_data_threshold: 0.05,
    }
}

pub fn backtest_params() -> BacktestParams {
    let mut cities = HashMap::new();

    cities.insert(
        "NYC",
        CityConfig {
            lat: 40.71,
            lon: -74.01,
            noaa_station: Some("GHCND:USW00023023"),
            nws_station: Some("KNYC"),
        },
    );

    cities.insert(
        "LA",
        CityConfig {
            lat: 34.05,
            lon: -118.24,
            noaa_station: Some("GHCND:USW00012918"),
            nws_station: Some("KLAX"),
        },
    );

    cities.insert(
        "London",
        CityConfig {
            lat: 51.51,
            lon: -0.13,
            noaa_station: None,
            nws_station: None,
        },
    );

    BacktestParams {
        lookback_months: 6,
        cities,
        temperature_thresholds_f: vec![32, 50, 60, 70, 80, 90],
        edge_threshold: 0.05,
        kelly_fraction: 0.25,
    }
}
