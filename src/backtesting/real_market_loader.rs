use std::path::Path;

use serde::Deserialize;

use crate::types::SimulatedMarket;
use crate::utils::parse_date;

#[derive(Debug, thiserror::Error)]
pub enum RealMarketLoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported input format for path: {0}")]
    UnsupportedFormat(String),
    #[error("invalid date '{value}'")]
    InvalidDate { value: String },
}

#[derive(Debug, Deserialize)]
struct RealMarketInputRow {
    date: String,
    market_id: String,
    #[serde(default)]
    market_title: String,
    market_type: String,
    #[serde(default)]
    threshold: Option<f64>,
    market_price: f64,
    actual_outcome: f64,
    city: String,
}

pub struct RealMarketLoader;

impl RealMarketLoader {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Vec<SimulatedMarket>, RealMarketLoadError> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "csv" => Self::load_csv(path),
            "json" => Self::load_json(path),
            _ => Err(RealMarketLoadError::UnsupportedFormat(path.display().to_string())),
        }
    }

    pub fn load_csv(path: impl AsRef<Path>) -> Result<Vec<SimulatedMarket>, RealMarketLoadError> {
        let mut reader = csv::ReaderBuilder::new().trim(csv::Trim::All).from_path(path)?;

        let mut out = Vec::new();
        for row in reader.deserialize::<RealMarketInputRow>() {
            let row = row?;
            out.push(Self::to_market(row)?);
        }

        Ok(out)
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Vec<SimulatedMarket>, RealMarketLoadError> {
        let bytes = std::fs::read(path)?;
        let rows: Vec<RealMarketInputRow> = serde_json::from_slice(&bytes)?;

        rows.into_iter().map(Self::to_market).collect()
    }

    fn to_market(row: RealMarketInputRow) -> Result<SimulatedMarket, RealMarketLoadError> {
        let date = parse_date(&row.date).ok_or_else(|| RealMarketLoadError::InvalidDate {
            value: row.date.clone(),
        })?;

        Ok(SimulatedMarket {
            date,
            market_id: row.market_id,
            market_title: if row.market_title.trim().is_empty() {
                "Polymarket market".to_string()
            } else {
                row.market_title
            },
            market_type: row.market_type.to_ascii_lowercase(),
            threshold: row.threshold.unwrap_or(0.0),
            market_price: row.market_price,
            actual_outcome: row.actual_outcome,
            city: row.city,
        })
    }
}

