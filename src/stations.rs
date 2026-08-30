//! Polymarket resolution stations + fitted forecast-error model.
//!
//! Polymarket temperature markets resolve on the Weather Underground daily-history page for a
//! specific airport station: the day's high is the MAX OF THE WHOLE-DEGREE METAR OBSERVATIONS
//! shown there (each ob rounded to the display unit), NOT the NWS CLI climate high (continuous
//! sensor, often 1°F higher) and NOT a gridded reanalysis value. Stations were recovered from the
//! market descriptions and every mapping below reproduced 43/43 settled capture outcomes exactly.
//! Two traps this table encodes: Denver resolves at Buckley SFB (KBKF, not KDEN) and London at
//! City Airport (EGLC, not Heathrow); `cities.rs` downtown coords are the wrong microclimate for
//! several cities (LAX vs downtown LA), so forecasts must be taken at the station coords here.
//!
//! KALSHI temperature markets settle differently: on the NWS Climatological Report (CLI) daily
//! max — the CONTINUOUS sensor high, typically 0–2 °F above the METAR ob-max — at its own station
//! set (Kalshi Dallas = DFW while Polymarket Dallas = Love Field; Kalshi NYC = Central Park, not
//! LaGuardia; Kalshi Houston = Hobby, not Intercontinental). Every `KALSHI_STATIONS` row re-scored
//! 402/402 settled 2026 markets (6,030 total) exactly against IEM's CLI archive. Because CLI reads
//! the continuous sensor, Kalshi's post phase is NOT deterministic — `post_bias`/`post_sigma`
//! carry the fitted CLI-vs-ob-max gap (Polymarket rows: 0 bias, 0.10 σ — WU IS the ob-max).
//!
//! NOT MAPPABLE with this machinery (legacy path on purpose): Hong Kong (resolves on the Hong
//! Kong Observatory climate extract, one-decimal °C, not airport METARs), Moscow, Istanbul and
//! Tel Aviv (non-Wunderground national sources) — their Polymarket markets keep the legacy
//! climatology pricing until someone builds those truth feeds.
//!
//! Forecast estimators use the EQUAL-WEIGHT 4-MODEL BLEND (ecmwf_ifs025, gfs_seamless,
//! ukmo_seamless, icon_seamless): mean of the per-model daily maxes. Measured 10–40% lower error
//! than any single model (and than an ECMWF-heavy weighting) on the May–Jun holdout; the fits
//! below are AGAINST that blend, so the daemon must price with the same blend (≥2 models).
//!
//! `sigma`/`bias` are the error model of `truth − estimate` (°C) fitted on Jan–Apr 2026
//! (n≈118/city/lead) against the daemon's exact capture geometry (15:00 UTC snapshot), indexed by
//! capture offset k = target_date − captured_date in days:
//!   k=0  day-of: max(running ob-max before capture, forecast max of remaining hours)
//!   k=1  next-day forecast (previous-day model run)
//!   k=2  two-day forecast
//! Validated May–Jun 2026 holdout + out-of-sample on the settled forward captures. Refit these
//! from accrued captures (`forecast_high` is stored per snapshot) as seasons drift.
//!
//! SUMMER REFIT 2026-07-20: the k=1,2 slots of both tables were re-blended against July forward
//! captures (realized highs recovered from winning bucket markets), which showed a systematic
//! summer over-forecast the winter fit couldn't see — venue-pooled drift −0.50 °C (Polymarket) /
//! −0.98 °C (Kalshi, a full 2 °F bucket), σ scale ×1.32 (PM) / ×0.9 (Kalshi), plus per-city
//! empirical-Bayes deltas shrunk toward the venue mean (largest: Kalshi LA −2.0 °C — July marine
//! layer; Qingdao −1.1 °C — sea breeze). Walk-forward validated: +5% bucket Brier on 414
//! out-of-sample rows. k=0 and post slots keep the Jan–Apr fit (obs-anchored, no drift signal).
//! `scripts/refit_station_tables.py` reproduces the fit for the next seasonal refit.
//!
//! SUMMER REFIT 2026-08-05 (k=1,2 again): the 07-20 refit fixed the mean but its σ rescale
//! OVER-widened. On post-refit rows, z = (realized − forecast)/σ had sd 0.78 with 82% of days
//! inside ±1σ against the Normal's 68% — a too-wide σ flattens the posterior peak, under-prices
//! the buckets ADJACENT to the forecast, and manufactures claimed edge exactly where realized
//! capture is negative. Meanwhile the mean had drifted back the OTHER way as the record-hot July
//! ran ahead of constants fitted through 07-19 (realized − forecast by target week: −0.86
//! pre-refit → +0.19 → +0.19 → +0.45). This refit therefore ADDS bias (venue mean +0.32 °C
//! Polymarket / +0.04 Kalshi, per-city EB-shrunk; largest Kalshi LA −0.59, Polymarket Shenzhen
//! +0.84) and SHARPENS σ ×0.8 on both venues. Walk-forward validated: +5.5% bucket Brier on 668
//! out-of-sample rows.
//!
//! ×0.8 is a deliberate HALF-STEP, not the fitted optimum: unconstrained the fit wanted ×0.61
//! (PM) / ×0.68 (Kalshi) for +7.3%, but σ sharpening is the direction that produces overconfident
//! pricing (cf. the `temp_predictive_sigma` bug), and 167 residual observations over two weeks is
//! a thin basis for halving spread in one move. The script's `--sigma-floor` (default 0.8) caps
//! the per-refit step so each one is re-validated on a fresh sample; lower it once the sample
//! supports the rest.

/// σ (°C) for a Polymarket market priced after its local day fully elapsed: the WU max is then
/// known from the same METAR feed the market resolves on (43/43 exact), so this only guards feed
/// hiccups. (Kalshi rows fit their own `post_sigma` — the CLI continuous-max gap.)
pub const POST_SIGMA: f64 = 0.10;
/// Pooled σ (°C) for k≥3 (lead-3 fit: 2.21); per-lead fits are only materially different below that.
pub const FAR_SIGMA: f64 = 2.2;

pub struct Station {
    /// Canonical city key from `cities.rs`.
    pub city: &'static str,
    /// IEM ASOS identifiers for the METAR feed (US ids drop the K prefix).
    pub iem_id: &'static str,
    pub iem_network: &'static str,
    /// Station coordinates — forecasts MUST be fetched here, not at the city-center coords.
    pub lat: f64,
    pub lon: f64,
    /// Resolution/display unit is °C (true) or °F (false); WU rounds each ob to this unit.
    pub celsius: bool,
    /// Standard-time UTC offset, hours. Used only for local-day phase math; DST shifts the capture
    /// hour by 1h but never flips day-of/post/future for any station here (capture ≈ 15:00 UTC).
    pub utc_offset_hours: i32,
    /// IANA timezone (IEM obs requests are made in UTC; kept for reference/debugging).
    pub tz: &'static str,
    /// Fitted σ (°C) for capture offset k = 0, 1, 2.
    pub sigma: [f64; 3],
    /// Fitted mean bias truth − estimate (°C) for k = 0, 1, 2, ADDED to the forecast. Mostly the
    /// grid-vs-station offset plus hourly sampling missing the continuous peak; Tokyo's ≈ +1.2 °C
    /// is a full °C bucket, so this is load-bearing for °C cities.
    pub bias: [f64; 3],
    /// Post-phase (target day fully elapsed) error of `truth − ob_max` (°C). Zero-ish for
    /// Polymarket (WU resolves on the ob-max itself); the fitted CLI continuous-max gap for Kalshi.
    pub post_bias: f64,
    pub post_sigma: f64,
}

/// Tokyo has no k=0 fit: a 15:00 UTC capture is already local midnight, so day-of never occurs
/// (phase is post); the k=1 values stand in for the unreachable slot.
pub const STATIONS: &[Station] = &[
    Station {
        city: "Seattle",
        iem_id: "SEA",
        iem_network: "WA_ASOS",
        lat: 47.449,
        lon: -122.309,
        celsius: false,
        utc_offset_hours: -8,
        tz: "America/Los_Angeles",
        sigma: [0.936, 1.022, 1.018],
        bias: [0.704, 0.162, -0.05],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "SF",
        iem_id: "SFO",
        iem_network: "CA_ASOS",
        lat: 37.62,
        lon: -122.375,
        celsius: false,
        utc_offset_hours: -8,
        tz: "America/Los_Angeles",
        sigma: [1.247, 1.33, 1.394],
        bias: [0.452, -0.362, -0.827],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "LA",
        iem_id: "LAX",
        iem_network: "CA_ASOS",
        lat: 33.938,
        lon: -118.389,
        celsius: false,
        utc_offset_hours: -8,
        tz: "America/Los_Angeles",
        sigma: [1.408, 1.493, 1.792],
        bias: [-0.17, -2.018, -1.919],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Denver",
        iem_id: "BKF",
        iem_network: "CO_ASOS",
        lat: 39.717,
        lon: -104.75,
        celsius: false,
        utc_offset_hours: -7,
        tz: "America/Denver",
        sigma: [1.647, 1.833, 1.862],
        bias: [0.464, 0.696, 0.991],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Miami",
        iem_id: "MIA",
        iem_network: "FL_ASOS",
        lat: 25.788,
        lon: -80.317,
        celsius: false,
        utc_offset_hours: -5,
        tz: "America/New_York",
        sigma: [0.744, 0.886, 1.021],
        bias: [0.708, 0.205, 0.074],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Dallas",
        iem_id: "DAL",
        iem_network: "TX_ASOS",
        lat: 32.847,
        lon: -96.852,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.341, 1.497, 1.791],
        bias: [0.61, -0.466, -0.665],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Atlanta",
        iem_id: "ATL",
        iem_network: "GA_ASOS",
        lat: 33.63,
        lon: -84.442,
        celsius: false,
        utc_offset_hours: -5,
        tz: "America/New_York",
        sigma: [1.473, 1.666, 1.982],
        bias: [1.309, 0.637, 0.611],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "NYC",
        iem_id: "LGA",
        iem_network: "NY_ASOS",
        lat: 40.777,
        lon: -73.873,
        celsius: false,
        utc_offset_hours: -5,
        tz: "America/New_York",
        sigma: [1.45, 1.821, 2.129],
        bias: [0.628, -0.267, -0.005],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Houston",
        iem_id: "HOU",
        iem_network: "TX_ASOS",
        lat: 29.646,
        lon: -95.279,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.097, 1.192, 1.381],
        bias: [0.412, -0.256, -0.415],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Chicago",
        iem_id: "ORD",
        iem_network: "IL_ASOS",
        lat: 41.96,
        lon: -87.932,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.183, 1.453, 2.046],
        bias: [0.875, -0.228, -0.261],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Austin",
        iem_id: "AUS",
        iem_network: "TX_ASOS",
        lat: 30.183,
        lon: -97.68,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.239, 1.607, 1.884],
        bias: [0.839, 0.334, 0.009],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "London",
        iem_id: "EGLC",
        iem_network: "GB__ASOS",
        lat: 51.505,
        lon: 0.055,
        celsius: true,
        utc_offset_hours: 0,
        tz: "Europe/London",
        sigma: [0.802, 1.141, 1.314],
        bias: [-0.144, -0.121, 0.06],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Tokyo",
        iem_id: "RJTT",
        iem_network: "JP__ASOS",
        lat: 35.553,
        lon: 139.78,
        celsius: true,
        utc_offset_hours: 9,
        tz: "Asia/Tokyo",
        sigma: [1.08, 1.143, 1.28],
        bias: [0.355, 0.509, 0.569],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    // International WU/METAR cities (all °C display), fitted the same way; truth re-verified 5/5
    // against settled captures. Seoul's sigma is honestly WIDE (Incheon is a reclaimed island; the
    // forecast grid cell is mostly sea). k0 for UTC+8/+9 cities is a nearly/fully elapsed local day
    // at capture (hence tiny/filled-in sigma); +9 rows reuse k1 for the unreachable k0 slot.
    Station {
        city: "Toronto",
        iem_id: "CYYZ",
        iem_network: "CA_ON_ASOS",
        lat: 43.677,
        lon: -79.63,
        celsius: true,
        utc_offset_hours: -5,
        tz: "America/Toronto",
        sigma: [1.363, 1.638, 1.801],
        bias: [0.714, 0.105, 0.125],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Sao Paulo",
        iem_id: "SBGR",
        iem_network: "BR__ASOS",
        lat: -23.435,
        lon: -46.473,
        celsius: true,
        utc_offset_hours: -3,
        tz: "America/Sao_Paulo",
        sigma: [1.203, 1.349, 1.502],
        bias: [0.62, 0.491, 0.517],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Shanghai",
        iem_id: "ZSPD",
        iem_network: "CN__ASOS",
        lat: 31.143,
        lon: 121.805,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Shanghai",
        sigma: [0.3, 1.278, 1.5],
        bias: [-0.004, 1.355, 1.382],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Guangzhou",
        iem_id: "ZGGG",
        iem_network: "CN__ASOS",
        lat: 23.392,
        lon: 113.299,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Shanghai",
        sigma: [0.3, 1.962, 2.185],
        bias: [-0.023, 1.483, 1.31],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Madrid",
        iem_id: "LEMD",
        iem_network: "ES__ASOS",
        lat: 40.472,
        lon: -3.561,
        celsius: true,
        utc_offset_hours: 1,
        tz: "Europe/Madrid",
        sigma: [0.803, 1.282, 1.247],
        bias: [0.018, 0.48, 0.614],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Mexico City",
        iem_id: "MMMX",
        iem_network: "MX__ASOS",
        lat: 19.436,
        lon: -99.072,
        celsius: true,
        utc_offset_hours: -6,
        tz: "America/Mexico_City",
        sigma: [0.806, 0.839, 1.008],
        bias: [0.24, -0.353, -0.359],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Panama City",
        iem_id: "MPMG",
        iem_network: "PA__ASOS",
        lat: 8.973,
        lon: -79.556,
        celsius: true,
        utc_offset_hours: -5,
        tz: "America/Panama",
        sigma: [1.116, 1.174, 1.182],
        bias: [0.545, 0.815, 0.892],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Seoul",
        iem_id: "RKSI",
        iem_network: "KR__ASOS",
        lat: 37.469,
        lon: 126.451,
        celsius: true,
        utc_offset_hours: 9,
        tz: "Asia/Seoul",
        sigma: [2.192, 2.321, 2.361],
        bias: [2.02, 2.025, 2.141],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Buenos Aires",
        iem_id: "SAEZ",
        iem_network: "AR__ASOS",
        lat: -34.822,
        lon: -58.536,
        celsius: true,
        utc_offset_hours: -3,
        tz: "America/Argentina/Buenos_Aires",
        sigma: [1.462, 1.578, 2.018],
        bias: [0.643, 0.068, -0.024],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Qingdao",
        iem_id: "ZSQD",
        iem_network: "CN__ASOS",
        lat: 36.362,
        lon: 120.088,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Shanghai",
        sigma: [0.3, 1.226, 1.283],
        bias: [0.0, -0.63, -0.545],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Singapore",
        iem_id: "WSSS",
        iem_network: "SG__ASOS",
        lat: 1.359,
        lon: 103.989,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Singapore",
        sigma: [0.3, 1.002, 1.143],
        bias: [0.0, 0.735, 0.791],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Shenzhen",
        iem_id: "ZGSZ",
        iem_network: "CN__ASOS",
        lat: 22.639,
        lon: 113.811,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Shanghai",
        sigma: [0.3, 1.01, 1.229],
        bias: [-0.007, 0.008, -0.01],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Beijing",
        iem_id: "ZBAA",
        iem_network: "CN__ASOS",
        lat: 40.08,
        lon: 116.585,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Shanghai",
        sigma: [0.3, 1.607, 1.662],
        bias: [0.0, 0.199, 0.193],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Manila",
        iem_id: "RPLL",
        iem_network: "PH__ASOS",
        lat: 14.509,
        lon: 121.02,
        celsius: true,
        utc_offset_hours: 8,
        tz: "Asia/Manila",
        sigma: [0.3, 0.927, 0.94],
        bias: [0.0, 0.84, 0.859],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
    Station {
        city: "Paris",
        iem_id: "LFPB",
        iem_network: "FR__ASOS",
        lat: 48.969,
        lon: 2.441,
        celsius: true,
        utc_offset_hours: 1,
        tz: "Europe/Paris",
        sigma: [0.521, 0.902, 1.062],
        bias: [-0.027, -0.106, 0.33],
        post_bias: 0.0,
        post_sigma: 0.1,
    },
];

/// Kalshi settlement stations (NWS CLI daily max, all °F). Fitted like `STATIONS` but with CLI
/// truth; every row re-scored 402/402 settled 2026 markets against IEM's CLI archive.
#[rustfmt::skip]
pub const KALSHI_STATIONS: &[Station] = &[
    Station { city: "NYC", iem_id: "NYC", iem_network: "NY_ASOS", lat: 40.783, lon: -73.967, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.479, 1.208, 1.365], bias: [1.217, 0.321, 0.496], post_bias: 0.292, post_sigma: 0.434 },
    Station { city: "Chicago", iem_id: "MDW", iem_network: "IL_ASOS", lat: 41.786, lon: -87.752, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.205, 1.073, 1.414], bias: [1.041, 0.175, 0.019], post_bias: 0.264, post_sigma: 0.354 },
    Station { city: "Austin", iem_id: "AUS", iem_network: "TX_ASOS", lat: 30.183, lon: -97.68, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.261, 1.119, 1.292], bias: [1.254, 0.798, 0.472], post_bias: 0.414, post_sigma: 0.386 },
    Station { city: "Denver", iem_id: "DEN", iem_network: "CO_ASOS", lat: 39.847, lon: -104.656, celsius: false, utc_offset_hours: -7, tz: "America/Denver", sigma: [1.783, 1.407, 1.406], bias: [0.376, 0.568, 0.839], post_bias: 0.353, post_sigma: 0.817 },
    Station { city: "LA", iem_id: "LAX", iem_network: "CA_ASOS", lat: 33.938, lon: -118.389, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [1.335, 0.963, 1.188], bias: [0.292, -1.972, -1.873], post_bias: 0.461, post_sigma: 0.444 },
    Station { city: "Miami", iem_id: "MIA", iem_network: "FL_ASOS", lat: 25.788, lon: -80.317, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [0.794, 0.636, 0.736], bias: [1.147, 1.093, 0.968], post_bias: 0.442, post_sigma: 0.403 },
    Station { city: "Philadelphia", iem_id: "PHL", iem_network: "PA_ASOS", lat: 39.868, lon: -75.231, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.523, 1.214, 1.461], bias: [1.515, 1.163, 1.277], post_bias: 0.297, post_sigma: 0.446 },
    Station { city: "Dallas", iem_id: "DFW", iem_network: "TX_ASOS", lat: 32.898, lon: -97.019, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.375, 1.036, 1.257], bias: [1.154, 0.324, 0.035], post_bias: 0.344, post_sigma: 0.307 },
    Station { city: "Seattle", iem_id: "SEA", iem_network: "WA_ASOS", lat: 47.449, lon: -122.309, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [0.984, 0.726, 0.702], bias: [1.02, 0.61, 0.397], post_bias: 0.315, post_sigma: 0.337 },
    Station { city: "Atlanta", iem_id: "ATL", iem_network: "GA_ASOS", lat: 33.63, lon: -84.442, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.486, 1.094, 1.338], bias: [1.705, 1.177, 1.151], post_bias: 0.395, post_sigma: 0.412 },
    Station { city: "Boston", iem_id: "BOS", iem_network: "MA_ASOS", lat: 42.361, lon: -71.01, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.557, 1.219, 1.594], bias: [0.777, 0.632, 0.771], post_bias: 0.299, post_sigma: 0.346 },
    Station { city: "Phoenix", iem_id: "PHX", iem_network: "AZ_ASOS", lat: 33.428, lon: -112.004, celsius: false, utc_offset_hours: -7, tz: "America/Phoenix", sigma: [0.871, 0.668, 0.666], bias: [0.965, 0.52, 0.921], post_bias: 0.598, post_sigma: 0.418 },
    Station { city: "Vegas", iem_id: "LAS", iem_network: "NV_ASOS", lat: 36.072, lon: -115.163, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [0.795, 0.586, 0.67], bias: [0.397, 0.394, 0.862], post_bias: 0.287, post_sigma: 0.306 },
    Station { city: "Washington", iem_id: "DCA", iem_network: "VA_ASOS", lat: 38.848, lon: -77.034, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.653, 1.297, 1.536], bias: [0.535, 0.467, 0.562], post_bias: 0.348, post_sigma: 0.413 },
    Station { city: "Houston", iem_id: "HOU", iem_network: "TX_ASOS", lat: 29.646, lon: -95.279, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.153, 0.85, 0.966], bias: [0.842, -0.156, -0.325], post_bias: 0.418, post_sigma: 0.431 },
];

/// Resolution station for a (canonical city key, venue) pair, if that venue's mapping is verified.
/// The tables differ on purpose: the venues settle on different stations AND different truth
/// variables (WU ob-max vs NWS CLI continuous max) for the same city name.
pub fn station_for(city: &str, source: &str) -> Option<&'static Station> {
    let table = match source {
        "polymarket" => STATIONS,
        "kalshi" => KALSHI_STATIONS,
        _ => return None,
    };
    table.iter().find(|s| s.city == city)
}
