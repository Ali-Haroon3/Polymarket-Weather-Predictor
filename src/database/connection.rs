use crate::config::database_url;
use crate::database::models::{ProbabilityForecast, TradeExecution, WeatherObservation};

#[derive(Debug, Clone)]
pub struct DatabaseConnection {
    pub url: String,
}

#[derive(Debug, Clone, Default)]
pub struct DatabaseSession {
    pub weather_observations: Vec<WeatherObservation>,
    pub probability_forecasts: Vec<ProbabilityForecast>,
    pub trade_executions: Vec<TradeExecution>,
}

pub fn get_engine() -> DatabaseConnection {
    DatabaseConnection {
        url: database_url(),
    }
}

pub fn get_session() -> DatabaseSession {
    DatabaseSession::default()
}

pub fn init_db() -> Result<(), String> {
    Ok(())
}

pub fn drop_db() -> Result<(), String> {
    Ok(())
}
