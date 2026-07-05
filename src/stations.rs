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
//! `sigma`/`bias` are the error model of `truth − estimate` (°C) fitted on Jan–Apr 2026
//! (n≈118/city/lead) against the daemon's exact capture geometry (15:00 UTC snapshot), indexed by
//! capture offset k = target_date − captured_date in days:
//!   k=0  day-of: max(running WU ob-max before capture, forecast max of remaining hours)
//!   k=1  next-day forecast (previous-day model run)
//!   k=2  two-day forecast
//! Validated May–Jun 2026 holdout + out-of-sample on the settled forward captures. Refit these
//! from accrued captures (`forecast_high` is stored per snapshot) as seasons drift.

/// σ (°C) for a market priced after its local day fully elapsed: the WU max is then known from the
/// same METAR feed the market resolves on (43/43 exact), so this only guards feed hiccups.
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
    },
];

/// Resolution station for a canonical city key, if the city's Polymarket mapping is verified.
pub fn station_for(city: &str) -> Option<&'static Station> {
    STATIONS.iter().find(|s| s.city == city)
}
