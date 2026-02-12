use polymarket_weather_predictor::database::init_db;

fn main() {
    if let Err(err) = init_db() {
        eprintln!("failed to initialize db: {err}");
        std::process::exit(1);
    }
    println!("database initialized successfully");
}
