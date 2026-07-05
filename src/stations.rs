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
//! `sigma`/`bias` are the error model of `truth − estimate` (°C) fitted on Jan–Apr 2026
//! (n≈118/city/lead) against the daemon's exact capture geometry (15:00 UTC snapshot), indexed by
//! capture offset k = target_date − captured_date in days:
//!   k=0  day-of: max(running ob-max before capture, forecast max of remaining hours)
//!   k=1  next-day forecast (previous-day model run)
//!   k=2  two-day forecast
//! Validated May–Jun 2026 holdout + out-of-sample on the settled forward captures. Refit these
//! from accrued captures (`forecast_high` is stored per snapshot) as seasons drift.

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
        sigma: [1.508, 1.527, 1.351],
        bias: [0.785, 0.847, 0.639],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "SF",
        iem_id: "SFO",
        iem_network: "CA_ASOS",
        lat: 37.620,
        lon: -122.375,
        celsius: false,
        utc_offset_hours: -8,
        tz: "America/Los_Angeles",
        sigma: [1.642, 1.656, 1.830],
        bias: [0.431, 0.462, -1.535],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [1.735, 1.740, 2.006],
        bias: [0.279, 0.290, 0.261],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "Denver",
        iem_id: "BKF",
        iem_network: "CO_ASOS",
        lat: 39.717,
        lon: -104.750,
        celsius: false,
        utc_offset_hours: -7,
        tz: "America/Denver",
        sigma: [2.009, 2.056, 2.307],
        bias: [-0.112, -0.096, 0.881],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [1.027, 1.220, 1.248],
        bias: [0.516, 0.651, -0.036],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [1.577, 1.606, 2.075],
        bias: [0.770, 0.869, 0.310],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "Atlanta",
        iem_id: "ATL",
        iem_network: "GA_ASOS",
        lat: 33.630,
        lon: -84.442,
        celsius: false,
        utc_offset_hours: -5,
        tz: "America/New_York",
        sigma: [1.935, 2.045, 1.924],
        bias: [1.106, 1.194, 0.716],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [1.770, 2.064, 2.265],
        bias: [0.270, 0.449, 0.897],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [1.545, 1.606, 1.484],
        bias: [0.264, 0.308, -0.329],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "Chicago",
        iem_id: "ORD",
        iem_network: "IL_ASOS",
        lat: 41.960,
        lon: -87.932,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.546, 1.744, 2.249],
        bias: [0.426, 0.568, 0.498],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "Austin",
        iem_id: "AUS",
        iem_network: "TX_ASOS",
        lat: 30.183,
        lon: -97.680,
        celsius: false,
        utc_offset_hours: -6,
        tz: "America/Chicago",
        sigma: [1.623, 1.901, 1.872],
        bias: [1.199, 1.135, 0.100],
        post_bias: 0.0,
        post_sigma: 0.10,
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
        sigma: [0.840, 1.180, 1.344],
        bias: [-0.245, 0.187, 0.868],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
    Station {
        city: "Tokyo",
        iem_id: "RJTT",
        iem_network: "JP__ASOS",
        lat: 35.553,
        lon: 139.780,
        celsius: true,
        utc_offset_hours: 9,
        tz: "Asia/Tokyo",
        sigma: [1.132, 1.132, 1.132],
        bias: [1.158, 1.158, 1.158],
        post_bias: 0.0,
        post_sigma: 0.10,
    },
];

/// Kalshi settlement stations (NWS CLI daily max, all °F). Fitted like `STATIONS` but with CLI
/// truth; every row re-scored 402/402 settled 2026 markets against IEM's CLI archive.
#[rustfmt::skip]
pub const KALSHI_STATIONS: &[Station] = &[
    Station { city: "NYC", iem_id: "NYC", iem_network: "NY_ASOS", lat: 40.783, lon: -73.967, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.856, 2.034, 2.143], bias: [0.962, 1.211, 1.404], post_bias: 0.292, post_sigma: 0.434 },
    Station { city: "Chicago", iem_id: "MDW", iem_network: "IL_ASOS", lat: 41.786, lon: -87.752, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.543, 1.832, 2.304], bias: [0.864, 1.030, 0.431], post_bias: 0.264, post_sigma: 0.354 },
    Station { city: "Austin", iem_id: "AUS", iem_network: "TX_ASOS", lat: 30.183, lon: -97.680, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.692, 1.975, 1.885], bias: [1.613, 1.549, 0.514], post_bias: 0.414, post_sigma: 0.386 },
    Station { city: "Denver", iem_id: "DEN", iem_network: "CO_ASOS", lat: 39.847, lon: -104.656, celsius: false, utc_offset_hours: -7, tz: "America/Denver", sigma: [2.044, 2.116, 2.547], bias: [-0.108, -0.089, 0.778], post_bias: 0.353, post_sigma: 0.817 },
    Station { city: "LA", iem_id: "LAX", iem_network: "CA_ASOS", lat: 33.938, lon: -118.389, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [1.769, 1.771, 2.019], bias: [0.741, 0.752, 0.722], post_bias: 0.461, post_sigma: 0.444 },
    Station { city: "Miami", iem_id: "MIA", iem_network: "FL_ASOS", lat: 25.788, lon: -80.317, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.049, 1.255, 1.264], bias: [0.953, 1.088, 0.400], post_bias: 0.442, post_sigma: 0.403 },
    Station { city: "Philadelphia", iem_id: "PHL", iem_network: "PA_ASOS", lat: 39.868, lon: -75.231, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.777, 2.023, 2.208], bias: [1.164, 1.413, 1.626], post_bias: 0.297, post_sigma: 0.446 },
    Station { city: "Dallas", iem_id: "DFW", iem_network: "TX_ASOS", lat: 32.898, lon: -97.019, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.630, 1.613, 2.103], bias: [1.185, 1.268, 0.196], post_bias: 0.344, post_sigma: 0.307 },
    Station { city: "Seattle", iem_id: "SEA", iem_network: "WA_ASOS", lat: 47.449, lon: -122.309, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [1.541, 1.555, 1.338], bias: [1.100, 1.162, 0.954], post_bias: 0.315, post_sigma: 0.337 },
    Station { city: "Atlanta", iem_id: "ATL", iem_network: "GA_ASOS", lat: 33.630, lon: -84.442, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.914, 1.968, 1.909], bias: [1.502, 1.589, 1.111], post_bias: 0.395, post_sigma: 0.412 },
    Station { city: "Boston", iem_id: "BOS", iem_network: "MA_ASOS", lat: 42.361, lon: -71.010, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.497, 1.627, 2.272], bias: [0.909, 1.041, 1.390], post_bias: 0.299, post_sigma: 0.346 },
    Station { city: "Phoenix", iem_id: "PHX", iem_network: "AZ_ASOS", lat: 33.428, lon: -112.004, celsius: false, utc_offset_hours: -7, tz: "America/Phoenix", sigma: [1.219, 1.235, 1.141], bias: [0.268, 0.275, 2.019], post_bias: 0.598, post_sigma: 0.418 },
    Station { city: "Vegas", iem_id: "LAS", iem_network: "NV_ASOS", lat: 36.072, lon: -115.163, celsius: false, utc_offset_hours: -8, tz: "America/Los_Angeles", sigma: [1.427, 1.466, 1.257], bias: [-0.660, -0.633, 1.408], post_bias: 0.287, post_sigma: 0.306 },
    Station { city: "Washington", iem_id: "DCA", iem_network: "VA_ASOS", lat: 38.848, lon: -77.034, celsius: false, utc_offset_hours: -5, tz: "America/New_York", sigma: [1.873, 2.036, 2.539], bias: [0.024, 0.095, 0.535], post_bias: 0.348, post_sigma: 0.413 },
    Station { city: "Houston", iem_id: "HOU", iem_network: "TX_ASOS", lat: 29.646, lon: -95.279, celsius: false, utc_offset_hours: -6, tz: "America/Chicago", sigma: [1.587, 1.614, 1.476], bias: [0.688, 0.733, 0.085], post_bias: 0.418, post_sigma: 0.431 },
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
