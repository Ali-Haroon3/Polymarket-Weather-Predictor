use std::collections::HashMap;
use std::path::PathBuf;

use chrono::NaiveDate;

use polymarket_weather_predictor::api::PolymarketHistoryDownloader;

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1).collect())?;

    let downloader = PolymarketHistoryDownloader::default();
    let rows = downloader
        .download_weather_history(args.start_date, args.end_date, args.limit, args.delay_ms)
        .await
        .map_err(|e| format!("download failed: {e}"))?;

    if rows.is_empty() {
        return Err("no historical rows downloaded (check date range / market availability)".to_string());
    }

    let parent = args.output.parent().map(|p| p.to_path_buf());
    if let Some(dir) = parent {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("failed creating {}: {e}", dir.display()))?;
        }
    }

    let mut writer = csv::Writer::from_path(&args.output)
        .map_err(|e| format!("failed opening {}: {e}", args.output.display()))?;

    for row in &rows {
        writer
            .serialize(row)
            .map_err(|e| format!("failed writing csv row: {e}"))?;
    }
    writer
        .flush()
        .map_err(|e| format!("failed flushing {}: {e}", args.output.display()))?;

    println!("Downloaded {} rows", rows.len());
    println!("Saved: {}", args.output.display());

    Ok(())
}

#[derive(Debug)]
struct CliArgs {
    output: PathBuf,
    start_date: Option<NaiveDate>,
    end_date: Option<NaiveDate>,
    limit: usize,
    delay_ms: u64,
}

impl CliArgs {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut map: HashMap<String, String> = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = args[i].clone();
            if !key.starts_with("--") {
                return Err(format!("unexpected argument '{key}'\n{}", usage()));
            }
            if i + 1 >= args.len() {
                return Err(format!("missing value for '{key}'\n{}", usage()));
            }
            map.insert(key, args[i + 1].clone());
            i += 2;
        }

        let output = map
            .get("--output")
            .map(PathBuf::from)
            .ok_or_else(|| format!("--output is required\n{}", usage()))?;

        let start_date = map
            .get("--start")
            .map(|d| parse_date(d).map_err(|e| format!("--start {e}")))
            .transpose()?;

        let end_date = map
            .get("--end")
            .map(|d| parse_date(d).map_err(|e| format!("--end {e}")))
            .transpose()?;

        let limit = map
            .get("--limit")
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|_| format!("invalid --limit '{}'", v))
            })
            .transpose()?
            .unwrap_or(500);

        let delay_ms = map
            .get("--delay-ms")
            .map(|v| {
                v.parse::<u64>()
                    .map_err(|_| format!("invalid --delay-ms '{}'", v))
            })
            .transpose()?
            .unwrap_or(0);

        Ok(Self {
            output,
            start_date,
            end_date,
            limit,
            delay_ms,
        })
    }
}

fn parse_date(value: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map_err(|_| format!("invalid date '{value}', expected YYYY-MM-DD"))
}

fn usage() -> String {
    "usage: cargo run --bin download_polymarket_history -- \\
  --output data/polymarket_history.csv \\
  [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit 500] [--delay-ms 100]"
        .to_string()
}
