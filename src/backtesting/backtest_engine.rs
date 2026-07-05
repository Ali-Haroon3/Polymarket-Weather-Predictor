use std::collections::HashMap;

use chrono::{Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::backtesting::market_simulator::{fahrenheit_to_celsius, MarketSimulator};
use crate::backtesting::performance_metrics::{PerformanceAnalyzer, PerformanceMetrics};
use crate::config::backtest_params;
use crate::data_pipeline::multi_source_aggregator::MultiSourceAggregator;
use crate::models::BayesianWeatherModel;
use crate::types::{SimulatedMarket, TradeRecord, WeatherRecord};
use crate::utils::clip;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub config: HashMap<String, String>,
    pub data_sources: HashMap<String, Vec<String>>,
    pub metrics: PerformanceMetrics,
    pub portfolio_values: Vec<f64>,
    pub total_markets_generated: usize,
    pub trades: Vec<TradeRecord>,
}

#[derive(Clone)]
pub struct BacktestConfig {
    pub cities: Option<Vec<String>>,
    pub initial_capital: f64,
    pub max_position_pct: f64,
    pub edge_threshold: f64,
    pub kelly_fraction: f64,
    pub model_lookback_days: i64,
    pub market_noise_std: f64,
    pub seed: u64,
}

#[derive(Clone)]
pub struct BacktestEngine {
    pub cities: Vec<String>,
    pub initial_capital: f64,
    pub max_position_pct: f64,
    pub edge_threshold: f64,
    pub kelly_fraction: f64,
    pub model_lookback_days: i64,
    pub seed: u64,
    pub aggregator: MultiSourceAggregator,
    pub market_sim: MarketSimulator,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let params = backtest_params();
        let city_list = config
            .cities
            .unwrap_or_else(|| params.cities.keys().map(|k| (*k).to_string()).collect());

        Self {
            cities: city_list,
            initial_capital: config.initial_capital,
            max_position_pct: config.max_position_pct,
            edge_threshold: config.edge_threshold,
            kelly_fraction: config.kelly_fraction,
            model_lookback_days: config.model_lookback_days,
            seed: config.seed,
            aggregator: MultiSourceAggregator::new(),
            market_sim: MarketSimulator::new(None, config.market_noise_std, config.seed),
        }
    }

    pub fn run(
        &mut self,
        start_date: Option<NaiveDate>,
        end_date: Option<NaiveDate>,
        weather_data_override: Option<HashMap<String, Vec<WeatherRecord>>>,
    ) -> BacktestResults {
        let params = backtest_params();

        let end = end_date.unwrap_or_else(|| Utc::now().date_naive());
        let start = start_date.unwrap_or_else(|| end - Duration::days(params.lookback_months * 30));

        let weather_data = weather_data_override.unwrap_or_else(|| {
            let fetch_start = start - Duration::days(self.model_lookback_days);
            self.aggregator
                .aggregate_multiple(&self.cities, fetch_start, end)
        });

        let markets = self.generate_markets(&weather_data);
        let (trades, portfolio_values) = self.simulate_trading(&weather_data, &markets);
        self.build_results(
            "simulated",
            start,
            end,
            &weather_data,
            markets.len(),
            trades,
            portfolio_values,
        )
    }

    pub fn run_with_real_markets(
        &mut self,
        markets: Vec<SimulatedMarket>,
        start_date: Option<NaiveDate>,
        end_date: Option<NaiveDate>,
        weather_data_override: Option<HashMap<String, Vec<WeatherRecord>>>,
    ) -> BacktestResults {
        let (min_date, max_date) = infer_market_date_bounds(&markets).unwrap_or_else(|| {
            let today = Utc::now().date_naive();
            (today, today)
        });

        let start = start_date.unwrap_or(min_date);
        let end = end_date.unwrap_or(max_date);

        let filtered_markets: Vec<SimulatedMarket> = markets
            .into_iter()
            .filter(|m| m.date >= start && m.date <= end)
            .collect();

        let market_cities: Vec<String> = {
            let mut cities = filtered_markets
                .iter()
                .map(|m| m.city.clone())
                .collect::<Vec<_>>();
            cities.sort();
            cities.dedup();
            cities
        };

        let weather_data = weather_data_override.unwrap_or_else(|| {
            let fetch_start = start - Duration::days(self.model_lookback_days);
            self.aggregator
                .aggregate_multiple(&market_cities, fetch_start, end)
        });

        let (trades, portfolio_values) = self.simulate_trading(&weather_data, &filtered_markets);

        self.build_results(
            "real_markets",
            start,
            end,
            &weather_data,
            filtered_markets.len(),
            trades,
            portfolio_values,
        )
    }

    fn generate_markets(
        &mut self,
        weather_data: &HashMap<String, Vec<WeatherRecord>>,
    ) -> Vec<SimulatedMarket> {
        let mut all = Vec::new();
        for (city, rows) in weather_data {
            let mut city_markets = self.market_sim.generate_markets(rows, city);
            all.append(&mut city_markets);
        }
        all
    }

    fn simulate_trading(
        &self,
        weather_data: &HashMap<String, Vec<WeatherRecord>>,
        markets: &[SimulatedMarket],
    ) -> (Vec<TradeRecord>, Vec<f64>) {
        if markets.is_empty() {
            return (Vec::new(), vec![self.initial_capital]);
        }

        let mut capital = self.initial_capital;
        let mut portfolio_values = vec![capital];
        let mut all_trades = Vec::new();

        let mut sorted = markets.to_vec();
        sorted.sort_by_key(|m| m.date);

        let mut dates = sorted.iter().map(|m| m.date).collect::<Vec<_>>();
        dates.sort();
        dates.dedup();

        for date in dates {
            let day_markets: Vec<&SimulatedMarket> =
                sorted.iter().filter(|m| m.date == date).collect();

            for market in day_markets {
                let Some(city_data) = weather_data.get(&market.city) else {
                    continue;
                };

                let trailing: Vec<WeatherRecord> = city_data
                    .iter()
                    .filter(|r| {
                        r.date < date && r.date >= date - Duration::days(self.model_lookback_days)
                    })
                    .cloned()
                    .collect();

                if trailing.len() < 14 {
                    continue;
                }

                let mut model = BayesianWeatherModel::default();
                if model.train_for_target(&trailing, date).is_err() {
                    continue;
                }

                let Some(our_estimate) = market_estimate(&model, market) else {
                    continue;
                };

                let edge = our_estimate - market.market_price;
                if edge.abs() < self.edge_threshold {
                    continue;
                }

                let position_size =
                    self.kelly_position_size(our_estimate, market.market_price, capital);
                if position_size < 1.0 {
                    continue;
                }

                let side = if edge > 0.0 { "BUY" } else { "SELL" };
                let pnl = if side == "BUY" {
                    position_size * (market.actual_outcome - market.market_price)
                } else {
                    position_size * (market.market_price - market.actual_outcome)
                };

                capital += pnl;

                all_trades.push(TradeRecord {
                    date,
                    city: market.city.clone(),
                    market_id: market.market_id.clone(),
                    market_title: market.market_title.clone(),
                    market_type: market.market_type.clone(),
                    threshold: market.threshold,
                    market_price: market.market_price,
                    our_estimate,
                    edge,
                    side: side.to_string(),
                    position_size,
                    actual_outcome: market.actual_outcome,
                    pnl,
                    capital_after: capital,
                });
            }

            portfolio_values.push(capital);
        }

        (all_trades, portfolio_values)
    }


    pub fn kelly_position_size(
        &self,
        probability: f64,
        market_price: f64,
        available_capital: f64,
    ) -> f64 {
        let frac = kelly_fraction_of_capital(probability, market_price, self.kelly_fraction);
        let max_pos = available_capital * self.max_position_pct;
        (frac * available_capital).min(max_pos)
    }

    fn build_results(
        &self,
        market_mode: &str,
        start: NaiveDate,
        end: NaiveDate,
        weather_data: &HashMap<String, Vec<WeatherRecord>>,
        total_markets_generated: usize,
        trades: Vec<TradeRecord>,
        portfolio_values: Vec<f64>,
    ) -> BacktestResults {
        let metrics =
            PerformanceAnalyzer::compute_metrics(&trades, &portfolio_values, self.initial_capital);

        let mut config = HashMap::new();
        config.insert("market_mode".to_string(), market_mode.to_string());
        config.insert("start_date".to_string(), start.to_string());
        config.insert("end_date".to_string(), end.to_string());
        config.insert("cities".to_string(), self.cities.join(","));
        config.insert(
            "initial_capital".to_string(),
            self.initial_capital.to_string(),
        );
        config.insert(
            "edge_threshold".to_string(),
            self.edge_threshold.to_string(),
        );
        config.insert(
            "kelly_fraction".to_string(),
            self.kelly_fraction.to_string(),
        );
        config.insert(
            "model_lookback_days".to_string(),
            self.model_lookback_days.to_string(),
        );

        let data_sources = weather_data
            .iter()
            .map(|(city, rows)| {
                let mut sources = rows
                    .iter()
                    .filter_map(|r| r.sources.clone())
                    .flat_map(|s| {
                        s.split(',')
                            .map(|x| x.trim().to_string())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                sources.sort();
                sources.dedup();
                if sources.is_empty() {
                    sources.push("unknown".to_string());
                }
                (city.clone(), sources)
            })
            .collect();

        BacktestResults {
            config,
            data_sources,
            metrics,
            portfolio_values,
            total_markets_generated,
            trades,
        }
    }
}

fn infer_market_date_bounds(markets: &[SimulatedMarket]) -> Option<(NaiveDate, NaiveDate)> {
    let min = markets.iter().map(|m| m.date).min()?;
    let max = markets.iter().map(|m| m.date).max()?;
    Some((min, max))
}

/// Bucket edge half-width applied IN THE TITLE'S UNIT before °F->°C conversion. Polymarket buckets
/// are stated at integer resolution ("be 18°C" = the 18 bin), so the priced interval is round-half-up
/// [N-0.5, N+0.5). This 0.5 is a calibration knob — set to 0.0 for a venue that floors instead.
const BUCKET_HALF_WIDTH: f64 = 0.5;
const PRICE_FLOOR: f64 = 0.01;
const PRICE_CEIL: f64 = 0.99;

/// Convert a threshold expressed in `unit` to °C. Any unit starting with 'c' (C, c, Celsius, °C) is
/// Celsius; None/unknown/Fahrenheit => °F. Normalizing covers hand-authored CSV/JSON rows.
fn to_celsius(value: f64, unit: &Option<String>) -> f64 {
    match unit.as_deref().map(|u| u.trim().to_ascii_lowercase()) {
        Some(u) if u.starts_with('c') => value,
        _ => fahrenheit_to_celsius(value),
    }
}

/// Fractional-Kelly bet size as a fraction of available capital for a binary market, given the
/// model's `probability`, the traded `market_price`, and the `kelly_fraction` multiplier. Handles
/// both sides (BUY YES when richer, SELL / BUY NO when cheaper) and never returns negative. The
/// caller multiplies by capital and applies any max-position cap. Single source of truth for the
/// Kelly math shared by the backtest sizer and the dashboard's forward-PnL readout.
pub fn kelly_fraction_of_capital(probability: f64, market_price: f64, kelly_fraction: f64) -> f64 {
    let kelly = if probability > market_price {
        let odds = 1.0 / market_price - 1.0;
        if odds <= 0.0 {
            0.0
        } else {
            (odds * probability - (1.0 - probability)) / odds
        }
    } else {
        let inv_prob = 1.0 - probability;
        let inv_price = 1.0 - market_price;
        let odds = 1.0 / inv_price - 1.0;
        if odds <= 0.0 {
            0.0
        } else {
            (odds * inv_prob - probability) / odds
        }
    };
    (kelly * kelly_fraction).max(0.0)
}

/// Price one market against a trained model: P(event) clipped to [0.01, 0.99], or None for shapes
/// the model can't price. The bucket ±0.5 widening is applied in the title's unit, then converted to
/// °C (the model's unit). The model forecasts the daily HIGH; callers train it before pricing.
pub fn market_estimate(model: &BayesianWeatherModel, market: &SimulatedMarket) -> Option<f64> {
    // The bucket methods are infallible f64 with no is_trained guard; refuse to price off the
    // untrained default (mu=0, sigma=1) so a caller can't get a confident-but-meaningless number.
    if !model.is_trained {
        return None;
    }
    let clip01 = |p: f64| clip(p, PRICE_FLOOR, PRICE_CEIL);
    match market.market_type.as_str() {
        // Legacy exceed market: P(high >= threshold), threshold in °F, no bucket widening.
        "temperature" => Some(clip01(
            model.prob_at_least(fahrenheit_to_celsius(market.threshold)),
        )),
        "temp_at_least" => {
            let lo = to_celsius(market.threshold - BUCKET_HALF_WIDTH, &market.unit);
            Some(clip01(model.prob_at_least(lo)))
        }
        "temp_at_most" => {
            let hi = to_celsius(market.threshold + BUCKET_HALF_WIDTH, &market.unit);
            Some(clip01(model.prob_at_most(hi)))
        }
        "temp_bucket" => {
            let upper = market.threshold_upper.unwrap_or(market.threshold);
            let lo = to_celsius(market.threshold - BUCKET_HALF_WIDTH, &market.unit);
            let hi = to_celsius(upper + BUCKET_HALF_WIDTH, &market.unit);
            Some(clip01(model.prob_between(lo, hi)))
        }
        "precipitation" => model
            .predict_precipitation_probability(5000)
            .ok()
            .map(|(p, _)| clip01(p)),
        _ => None,
    }
}

/// A market with the model's probability and the realized outcome side by side. Unlike a trade
/// record this covers EVERY market (not just the ones an edge triggered), which is what calibration
/// analysis needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvaluation {
    pub date: NaiveDate,
    pub city: String,
    pub market_id: String,
    pub market_title: String,
    pub market_type: String,
    pub threshold: f64,
    pub threshold_upper: Option<f64>,
    pub unit: Option<String>,
    pub market_price: f64,
    pub actual_outcome: f64,
    /// Model P(event); None when there isn't enough trailing weather to train (or city uncovered).
    pub model_estimate: Option<f64>,
}

/// Score each market: train the model on the city's trailing `lookback_days` of weather (as the
/// backtest does) and record its probability alongside the realized outcome.
pub fn evaluate_markets(
    markets: &[SimulatedMarket],
    weather_data: &HashMap<String, Vec<WeatherRecord>>,
    lookback_days: i64,
) -> Vec<MarketEvaluation> {
    evaluate_markets_inner(markets, weather_data, lookback_days, None, 0.0)
}

/// Like `evaluate_markets`, but when a daily-high FORECAST exists for a market's (city, date) it
/// prices from that point forecast — N(high, forecast_sigma) — instead of climatology, measuring
/// real forecast skill. Falls back to the climatology model wherever no forecast is available.
pub fn evaluate_markets_with_forecast(
    markets: &[SimulatedMarket],
    weather_data: &HashMap<String, Vec<WeatherRecord>>,
    lookback_days: i64,
    forecasts: &HashMap<String, HashMap<NaiveDate, f64>>,
    forecast_sigma: f64,
) -> Vec<MarketEvaluation> {
    evaluate_markets_inner(
        markets,
        weather_data,
        lookback_days,
        Some(forecasts),
        forecast_sigma,
    )
}

fn evaluate_markets_inner(
    markets: &[SimulatedMarket],
    weather_data: &HashMap<String, Vec<WeatherRecord>>,
    lookback_days: i64,
    forecasts: Option<&HashMap<String, HashMap<NaiveDate, f64>>>,
    forecast_sigma: f64,
) -> Vec<MarketEvaluation> {
    markets
        .iter()
        .map(|market| {
            let forecast_high = forecasts
                .and_then(|f| f.get(&market.city))
                .and_then(|m| m.get(&market.date))
                .copied();

            let model_estimate = if let Some(high_c) = forecast_high {
                let mut model = BayesianWeatherModel::default();
                model.set_point_forecast(high_c, forecast_sigma);
                market_estimate(&model, market)
            } else {
                weather_data.get(&market.city).and_then(|city_data| {
                    let trailing: Vec<WeatherRecord> = city_data
                        .iter()
                        .filter(|r| {
                            r.date < market.date
                                && r.date >= market.date - Duration::days(lookback_days)
                        })
                        .cloned()
                        .collect();
                    if trailing.len() < 14 {
                        return None;
                    }
                    let mut model = BayesianWeatherModel::default();
                    model.train_for_target(&trailing, market.date).ok()?;
                    market_estimate(&model, market)
                })
            };

            MarketEvaluation {
                date: market.date,
                city: market.city.clone(),
                market_id: market.market_id.clone(),
                market_title: market.market_title.clone(),
                market_type: market.market_type.clone(),
                threshold: market.threshold,
                threshold_upper: market.threshold_upper,
                unit: market.unit.clone(),
                market_price: market.market_price,
                actual_outcome: market.actual_outcome,
                model_estimate,
            }
        })
        .collect()
}
