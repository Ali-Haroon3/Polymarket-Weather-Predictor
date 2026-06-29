use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherRecord {
    pub date: NaiveDate,
    pub station_id: String,
    pub temperature_max: Option<f64>,
    pub temperature_min: Option<f64>,
    pub temperature_mean: Option<f64>,
    pub precipitation_total: Option<f64>,
    pub wind_speed_mean: Option<f64>,
    pub source_count: Option<u32>,
    pub sources: Option<String>,
    pub data_quality_score: Option<f64>,
    pub is_validated: bool,
    pub location: Option<String>,
}

impl WeatherRecord {
    pub fn new(date: NaiveDate, station_id: impl Into<String>) -> Self {
        Self {
            date,
            station_id: station_id.into(),
            temperature_max: None,
            temperature_min: None,
            temperature_mean: None,
            precipitation_total: None,
            wind_speed_mean: None,
            source_count: None,
            sources: None,
            data_quality_score: Some(1.0),
            is_validated: true,
            location: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawObservation {
    pub date: NaiveDate,
    pub station_id: String,
    pub datatype: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedMarket {
    pub date: NaiveDate,
    pub market_id: String,
    pub market_title: String,
    /// Routing discriminator for pricing. Legacy values: "temperature" (P(high >= threshold),
    /// threshold in °F) and "precipitation". Bucket shapes over the daily HIGH, bounds in `unit`:
    ///   "temp_at_least" -> P(high >= threshold)
    ///   "temp_at_most"  -> P(high <= threshold)
    ///   "temp_bucket"   -> P(threshold <= high <= threshold_upper)  ("be N" => upper == threshold)
    pub market_type: String,
    /// Lower / primary bound, in `unit`.
    pub threshold: f64,
    /// Upper bound for "temp_bucket"; None for open-ended / legacy shapes.
    #[serde(default)]
    pub threshold_upper: Option<f64>,
    /// Unit of threshold/threshold_upper: "F" or "C". None => legacy °F (back-compat for old data).
    #[serde(default)]
    pub unit: Option<String>,
    pub market_price: f64,
    pub actual_outcome: f64,
    pub city: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub date: NaiveDate,
    pub city: String,
    pub market_id: String,
    pub market_title: String,
    pub market_type: String,
    pub threshold: f64,
    pub market_price: f64,
    pub our_estimate: f64,
    pub edge: f64,
    pub side: String,
    pub position_size: f64,
    pub actual_outcome: f64,
    pub pnl: f64,
    pub capital_after: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketQuote {
    pub bid: f64,
    pub ask: f64,
    pub mid: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadQuote {
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
    pub mid: f64,
    pub market_id: String,
    pub current_inventory: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPnLPoint {
    pub timestamp: DateTime<Utc>,
    pub portfolio_value: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub date: NaiveDate,
    pub close: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityPoint {
    pub date: NaiveDate,
    pub probability: f64,
}
