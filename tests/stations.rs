//! Station table sanity + the pure nowcast math (WU rounding, local-day phases, obs/forecast fusion).

use chrono::{NaiveDate, NaiveDateTime};

use polymarket_weather_predictor::cities;
use polymarket_weather_predictor::data_pipeline::station_obs::{
    blend_forecast_day_max_c, forecast_day_max_c, nowcast_mu_sigma, per_model_day_maxes_c,
    phase_for, wu_running_max_c, Phase,
};
use polymarket_weather_predictor::stations::{
    station_for, Station, KALSHI_STATIONS, POST_SIGMA, STATIONS,
};

fn dt(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap()
}

fn d(s: &str) -> NaiveDate {
    s.parse().unwrap()
}

fn station(city: &str) -> &'static Station {
    station_for(city, "polymarket").unwrap()
}

#[test]
fn table_is_sane_and_cities_are_canonical() {
    assert!(!STATIONS.is_empty());
    assert!(!KALSHI_STATIONS.is_empty());
    for s in STATIONS.iter().chain(KALSHI_STATIONS) {
        assert!(
            cities::CITIES.iter().any(|c| c.key == s.city),
            "{} not a canonical city key",
            s.city
        );
        for k in 0..3 {
            assert!(s.sigma[k] > 0.0, "{} sigma[{k}] must be positive", s.city);
            assert!(s.bias[k].abs() < 3.0, "{} bias[{k}] implausible", s.city);
        }
        assert!((-12..=14).contains(&s.utc_offset_hours));
        assert!(s.post_sigma > 0.0);
    }
    assert!(station_for("Denver", "polymarket").unwrap().iem_id == "BKF"); // Buckley SFB, not KDEN
    assert!(station_for("London", "polymarket").unwrap().iem_id == "EGLC"); // City Airport, not Heathrow
    assert!(station_for("Phoenix", "polymarket").is_none()); // unmapped for this venue -> legacy path

    // The venues settle the SAME city on different stations — routing by source is load-bearing.
    assert_eq!(station_for("Denver", "kalshi").unwrap().iem_id, "DEN");
    assert_eq!(station_for("Dallas", "kalshi").unwrap().iem_id, "DFW"); // Polymarket: DAL
    assert_eq!(station_for("NYC", "kalshi").unwrap().iem_id, "NYC"); // Central Park, not LGA
    assert_eq!(station_for("Houston", "kalshi").unwrap().iem_id, "HOU"); // Hobby, not IAH
    assert!(station_for("Phoenix", "kalshi").is_some());
    assert!(station_for("Tokyo", "kalshi").is_none());
    assert!(station_for("Denver", "unknown-venue").is_none());

    // Kalshi post-phase is the fitted CLI-vs-ob-max gap, not deterministic.
    let kden = station_for("Denver", "kalshi").unwrap();
    assert!(kden.post_bias > 0.0 && kden.post_sigma > POST_SIGMA);
}

#[test]
fn phase_follows_the_local_day() {
    let now = dt("2026-06-30 15:00"); // the daemon's ~capture instant (UTC)
    let denver = station("Denver"); // UTC-7: 08:00 local, inside the day
    assert_eq!(phase_for(now, d("2026-06-30"), denver), Phase::DayOf);
    assert_eq!(phase_for(now, d("2026-06-29"), denver), Phase::Post);
    assert_eq!(phase_for(now, d("2026-07-01"), denver), Phase::Lead(1));
    assert_eq!(phase_for(now, d("2026-07-03"), denver), Phase::Lead(3));

    // Tokyo (UTC+9): 15:00 UTC is already local midnight of July 1 — the target day is over, and a
    // "tomorrow" market is really the day now starting.
    let tokyo = station("Tokyo");
    assert_eq!(phase_for(now, d("2026-06-30"), tokyo), Phase::Post);
    assert_eq!(phase_for(now, d("2026-07-01"), tokyo), Phase::DayOf);
}

#[test]
fn wu_rounding_matches_the_resolution_source() {
    // Wunderground rounds each METAR ob to the display unit, then takes the max. 66.9F displays as
    // 67F; a later 66.4F ob (66F) must not raise it.
    let sea = station("Seattle"); // F station, UTC-8
    let obs = vec![
        (dt("2026-06-28 18:53"), 66.9), // 10:53 local
        (dt("2026-06-28 22:53"), 66.4),
    ];
    let max_c = wu_running_max_c(&obs, d("2026-06-28"), sea, None).unwrap();
    assert!((max_c - (67.0 - 32.0) / 1.8).abs() < 1e-9);

    // Celsius station: 80.06F = 26.7C displays as 27C.
    let tokyo = station("Tokyo");
    let obs = vec![(dt("2026-06-30 03:00"), 80.06)]; // 12:00 JST
    let max_c = wu_running_max_c(&obs, d("2026-06-30"), tokyo, None).unwrap();
    assert!((max_c - 27.0).abs() < 1e-9);

    // Local-day windowing: an ob from the previous UTC date can belong to the target's local day
    // (and vice versa). 07:00 UTC Jun 29 = 23:00 local Jun 28 for Seattle.
    let obs = vec![(dt("2026-06-29 07:00"), 70.0)];
    assert!(wu_running_max_c(&obs, d("2026-06-28"), sea, None).is_some());
    assert!(wu_running_max_c(&obs, d("2026-06-29"), sea, None).is_none());

    // Cutoff excludes obs at/after the capture instant.
    let obs = vec![
        (dt("2026-06-28 16:00"), 60.0),
        (dt("2026-06-28 23:00"), 75.0),
    ];
    let run = wu_running_max_c(&obs, d("2026-06-28"), sea, Some(dt("2026-06-28 17:00"))).unwrap();
    assert!((run - (60.0 - 32.0) / 1.8).abs() < 1e-9);
}

#[test]
fn forecast_day_max_respects_window_and_from() {
    let denver = station("Denver"); // local day = [07:00, 07:00+24h) UTC
    let hourly = vec![
        (dt("2026-06-30 06:00"), 30.0), // previous local day — excluded
        (dt("2026-06-30 18:00"), 25.0),
        (dt("2026-06-30 21:00"), 28.0),
        (dt("2026-07-01 06:00"), 20.0), // 23:00 local, still June 30
    ];
    let whole = forecast_day_max_c(&hourly, d("2026-06-30"), denver, None).unwrap();
    assert!((whole - 28.0).abs() < 1e-9);
    let after = forecast_day_max_c(
        &hourly,
        d("2026-06-30"),
        denver,
        Some(dt("2026-06-30 22:00")),
    )
    .unwrap();
    assert!((after - 20.0).abs() < 1e-9);
}

#[test]
fn nowcast_fuses_obs_and_forecast_per_phase() {
    let denver = station("Denver");
    // Post: obs max is the answer, at POST_SIGMA.
    assert_eq!(
        nowcast_mu_sigma(denver, Phase::Post, Some(31.0), None),
        Some((31.0 + denver.post_bias, denver.post_sigma))
    );
    assert!(
        (denver.post_sigma - POST_SIGMA).abs() < 1e-9,
        "polymarket post is the WU ob-max"
    );
    assert_eq!(nowcast_mu_sigma(denver, Phase::Post, None, None), None);

    // DayOf: max(running obs, remaining forecast) + fitted k0 bias/sigma; either side may be missing.
    let (mu, sig) = nowcast_mu_sigma(denver, Phase::DayOf, Some(20.0), Some(28.0)).unwrap();
    assert!((mu - (28.0 + denver.bias[0])).abs() < 1e-9);
    assert!((sig - denver.sigma[0]).abs() < 1e-9);
    let (mu, _) = nowcast_mu_sigma(denver, Phase::DayOf, Some(29.0), Some(28.0)).unwrap();
    assert!(
        (mu - (29.0 + denver.bias[0])).abs() < 1e-9,
        "hot morning floors the estimate"
    );
    assert!(nowcast_mu_sigma(denver, Phase::DayOf, None, Some(28.0)).is_some());
    assert_eq!(nowcast_mu_sigma(denver, Phase::DayOf, None, None), None);

    // Lead k uses the fitted slot; far leads use the pooled far sigma.
    let (mu, sig) = nowcast_mu_sigma(denver, Phase::Lead(1), None, Some(28.0)).unwrap();
    assert!((mu - (28.0 + denver.bias[1])).abs() < 1e-9);
    assert!((sig - denver.sigma[1]).abs() < 1e-9);
    let (_, sig) = nowcast_mu_sigma(denver, Phase::Lead(5), None, Some(28.0)).unwrap();
    assert!((sig - polymarket_weather_predictor::stations::FAR_SIGMA).abs() < 1e-9);
    assert_eq!(nowcast_mu_sigma(denver, Phase::Lead(1), None, None), None);
}

#[test]
fn blend_averages_per_model_maxes_and_needs_two_models() {
    let denver = station("Denver");
    let m1 = vec![
        (dt("2026-06-30 18:00"), 30.0),
        (dt("2026-06-30 21:00"), 32.0),
    ];
    let m2 = vec![(dt("2026-06-30 19:00"), 28.0)];
    // mean of per-model maxes (32, 28) — NOT the max of the pooled hours (32).
    let b = blend_forecast_day_max_c(&[m1.clone(), m2], d("2026-06-30"), denver, None).unwrap();
    assert!((b - 30.0).abs() < 1e-9);
    // a single surviving model is not the fitted estimator — degrade instead of misprice
    assert!(blend_forecast_day_max_c(&[m1], d("2026-06-30"), denver, None).is_none());
    assert!(blend_forecast_day_max_c(&[], d("2026-06-30"), denver, None).is_none());
}

#[test]
fn per_model_maxes_stay_aligned_with_fetch_order_for_logging() {
    // The capture daemon zips these slots with BLEND_MODELS names, so a model with no hours must
    // hold its slot as None rather than collapsing the vector.
    let denver = station("Denver");
    let m1 = vec![
        (dt("2026-06-30 18:00"), 30.0),
        (dt("2026-06-30 21:00"), 32.0),
    ];
    let empty: Vec<(chrono::NaiveDateTime, f64)> = Vec::new();
    let m3 = vec![(dt("2026-06-30 19:00"), 28.0)];
    let per = per_model_day_maxes_c(&[m1, empty, m3], d("2026-06-30"), denver, None);
    assert_eq!(per.len(), 3);
    assert!((per[0].unwrap() - 32.0).abs() < 1e-9);
    assert_eq!(per[1], None);
    assert!((per[2].unwrap() - 28.0).abs() < 1e-9);
}
