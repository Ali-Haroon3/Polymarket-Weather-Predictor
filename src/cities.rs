//! Single source of truth for the cities the system knows about: the title patterns used to
//! recognize a city in a market question, and its coordinates for fetching weather. Keeping these
//! together prevents the drift where a title parses to a city that has no weather source.

use crate::utils::contains_word;

pub struct City {
    /// Canonical key used throughout the pipeline (market.city, weather location key).
    pub key: &'static str,
    /// Lowercase substrings that identify this city in a market title.
    pub patterns: &'static [&'static str],
    pub lat: f64,
    pub lon: f64,
}

/// The canonical city registry. Polymarket runs daily temperature markets across these.
pub const CITIES: &[City] = &[
    // US
    City { key: "NYC", patterns: &["new york", "nyc"], lat: 40.71, lon: -74.01 },
    City { key: "LA", patterns: &["los angeles", "l.a."], lat: 34.05, lon: -118.24 },
    City { key: "Chicago", patterns: &["chicago"], lat: 41.88, lon: -87.63 },
    City { key: "Dallas", patterns: &["dallas"], lat: 32.78, lon: -96.80 },
    City { key: "Denver", patterns: &["denver"], lat: 39.74, lon: -104.99 },
    City { key: "Miami", patterns: &["miami"], lat: 25.76, lon: -80.19 },
    City { key: "Boston", patterns: &["boston"], lat: 42.36, lon: -71.06 },
    City { key: "Seattle", patterns: &["seattle"], lat: 47.61, lon: -122.33 },
    City { key: "Atlanta", patterns: &["atlanta"], lat: 33.75, lon: -84.39 },
    City { key: "Houston", patterns: &["houston"], lat: 29.76, lon: -95.37 },
    City { key: "Phoenix", patterns: &["phoenix"], lat: 33.45, lon: -112.07 },
    City { key: "Portland", patterns: &["portland"], lat: 45.52, lon: -122.68 },
    City { key: "SF", patterns: &["san francisco"], lat: 37.77, lon: -122.42 },
    City { key: "Austin", patterns: &["austin"], lat: 30.27, lon: -97.74 },
    City { key: "Washington", patterns: &["washington dc", "washington, d.c."], lat: 38.91, lon: -77.04 },
    City { key: "Philadelphia", patterns: &["philadelphia"], lat: 39.95, lon: -75.17 },
    City { key: "Vegas", patterns: &["las vegas"], lat: 36.17, lon: -115.14 },
    // Americas (non-US)
    City { key: "Toronto", patterns: &["toronto"], lat: 43.65, lon: -79.38 },
    City { key: "Mexico City", patterns: &["mexico city"], lat: 19.43, lon: -99.13 },
    City { key: "Panama City", patterns: &["panama city"], lat: 8.98, lon: -79.52 },
    City { key: "Sao Paulo", patterns: &["sao paulo", "são paulo"], lat: -23.55, lon: -46.63 },
    City { key: "Buenos Aires", patterns: &["buenos aires"], lat: -34.60, lon: -58.38 },
    // Europe / Middle East / Africa
    City { key: "London", patterns: &["london"], lat: 51.51, lon: -0.13 },
    City { key: "Paris", patterns: &["paris"], lat: 48.85, lon: 2.35 },
    City { key: "Berlin", patterns: &["berlin"], lat: 52.52, lon: 13.40 },
    City { key: "Madrid", patterns: &["madrid"], lat: 40.42, lon: -3.70 },
    City { key: "Rome", patterns: &["rome"], lat: 41.90, lon: 12.50 },
    City { key: "Istanbul", patterns: &["istanbul"], lat: 41.01, lon: 28.98 },
    City { key: "Moscow", patterns: &["moscow"], lat: 55.76, lon: 37.62 },
    City { key: "Tel Aviv", patterns: &["tel aviv"], lat: 32.08, lon: 34.78 },
    City { key: "Dubai", patterns: &["dubai"], lat: 25.20, lon: 55.27 },
    // Asia / Pacific
    City { key: "Tokyo", patterns: &["tokyo"], lat: 35.68, lon: 139.69 },
    City { key: "Singapore", patterns: &["singapore"], lat: 1.35, lon: 103.82 },
    City { key: "Manila", patterns: &["manila"], lat: 14.60, lon: 120.98 },
    City { key: "Beijing", patterns: &["beijing"], lat: 39.90, lon: 116.41 },
    City { key: "Shanghai", patterns: &["shanghai"], lat: 31.23, lon: 121.47 },
    City { key: "Shenzhen", patterns: &["shenzhen"], lat: 22.54, lon: 114.06 },
    City { key: "Guangzhou", patterns: &["guangzhou"], lat: 23.13, lon: 113.26 },
    City { key: "Qingdao", patterns: &["qingdao"], lat: 36.07, lon: 120.38 },
    City { key: "Hong Kong", patterns: &["hong kong"], lat: 22.32, lon: 114.17 },
    City { key: "Mumbai", patterns: &["mumbai"], lat: 19.08, lon: 72.88 },
    City { key: "Delhi", patterns: &["new delhi", "delhi"], lat: 28.61, lon: 77.21 },
    City { key: "Seoul", patterns: &["seoul"], lat: 37.57, lon: 126.98 },
    City { key: "Sydney", patterns: &["sydney"], lat: -33.87, lon: 151.21 },
];

/// Identify a city from a (any-case) market title. `la` is matched only as a standalone word, after
/// every other pattern, so "Dallas"/"Atlanta"/"Manila" can't be misread as Los Angeles.
pub fn infer_city(title: &str) -> Option<&'static str> {
    let t = title.to_ascii_lowercase();
    for c in CITIES {
        if c.patterns.iter().any(|p| t.contains(p)) {
            return Some(c.key);
        }
    }
    if contains_word(&t, "la") {
        return Some("LA");
    }
    None
}

/// Coordinates for a canonical city key.
pub fn coords(key: &str) -> Option<(f64, f64)> {
    CITIES.iter().find(|c| c.key == key).map(|c| (c.lat, c.lon))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_city_basic_and_collisions() {
        assert_eq!(infer_city("New York rain market"), Some("NYC"));
        assert_eq!(infer_city("London temperature"), Some("London"));
        assert_eq!(infer_city("Will it rain in LA?"), Some("LA"));
        assert_eq!(infer_city("LA temperature above 90F"), Some("LA"));
        // "la" inside another city must not win
        assert_eq!(infer_city("Dallas temperature above 90F"), Some("Dallas"));
        assert_eq!(infer_city("Atlanta rain forecast"), Some("Atlanta"));
        assert_eq!(infer_city("highest temperature in Manila be 36C"), Some("Manila"));
        // newly covered international cities
        assert_eq!(infer_city("Highest temperature in Mexico City on April 17?"), Some("Mexico City"));
        assert_eq!(infer_city("Tokyo typhoon season"), Some("Tokyo"));
        assert_eq!(infer_city("Seattle rain this week"), Some("Seattle"));
        assert_eq!(infer_city("Unknown Town weather"), None);
    }

    #[test]
    fn test_every_city_has_coords() {
        for c in CITIES {
            assert!(coords(c.key).is_some(), "{} missing coords", c.key);
            assert!((-90.0..=90.0).contains(&c.lat) && (-180.0..=180.0).contains(&c.lon));
        }
    }
}
