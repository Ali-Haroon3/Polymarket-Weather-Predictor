pub mod accuweather_fetcher;
pub mod awc_fetcher;
pub mod multi_source_aggregator;
pub mod noaa_fetcher;
pub mod nws_fetcher;
pub mod open_meteo_fetcher;
pub mod openweathermap_fetcher;
pub mod station_obs;
pub mod station_pricer;
pub mod tomorrow_io_fetcher;
pub mod visual_crossing_fetcher;
pub mod weatherapi_fetcher;

pub use accuweather_fetcher::AccuWeatherFetcher;
pub use awc_fetcher::AWCFetcher;
pub use multi_source_aggregator::MultiSourceAggregator;
pub use noaa_fetcher::NOAAFetcher;
pub use nws_fetcher::NWSFetcher;
pub use open_meteo_fetcher::OpenMeteoFetcher;
pub use openweathermap_fetcher::OpenWeatherMapFetcher;
pub use station_obs::IemObsFetcher;
pub use station_pricer::StationPricer;
pub use tomorrow_io_fetcher::TomorrowIOFetcher;
pub use visual_crossing_fetcher::VisualCrossingFetcher;
pub use weatherapi_fetcher::WeatherAPIFetcher;

/// A blocking HTTP client with the given timeout — shared by every fetcher's `new()`.
pub(crate) fn build_client(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}
