use chrono::{DateTime, NaiveDate, Utc};

#[derive(Debug, Clone)]
pub struct WeatherObservation {
    pub id: i64,
    pub station_id: String,
    pub station_name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub observation_date: NaiveDate,
    pub temperature_max: Option<f64>,
    pub temperature_min: Option<f64>,
    pub temperature_mean: Option<f64>,
    pub precipitation_total: Option<f64>,
    pub wind_speed_mean: Option<f64>,
    pub data_quality_score: f64,
    pub is_validated: bool,
}

#[derive(Debug, Clone)]
pub struct ProbabilityForecast {
    pub id: i64,
    pub observation_id: Option<i64>,
    pub station_id: String,
    pub forecast_date: NaiveDate,
    pub horizon_days: i32,
    pub prob_temp_above_90f: Option<f64>,
    pub prob_temp_above_95f: Option<f64>,
    pub prob_temp_below_32f: Option<f64>,
    pub expected_temperature: Option<f64>,
    pub temperature_std: Option<f64>,
    pub prob_precipitation: Option<f64>,
    pub expected_precipitation: Option<f64>,
    pub precipitation_std: Option<f64>,
    pub model_version: String,
    pub calibration_score: Option<f64>,
    pub credible_interval_lower: Option<f64>,
    pub credible_interval_upper: Option<f64>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TradeExecution {
    pub id: i64,
    pub forecast_id: Option<i64>,
    pub market_id: String,
    pub market_type: String,
    pub order_date: DateTime<Utc>,
    pub order_type: String,
    pub position_size: f64,
    pub entry_price: f64,
    pub bid_price: Option<f64>,
    pub ask_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub position_status: String,
    pub current_price: Option<f64>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub sharpe_ratio: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
