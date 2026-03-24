# KED_NN_pipeline

## Description

This repository provides a complete workflow for processing weather data from MeteoSwiss open data platform. It downloads raw meteorological data, cleans and preprocesses it, interpolates missing values using Kriging with External Drift (KED), and trains neural network models to produce precipitation forecasts.

The pipeline includes:
- Data downloading from MeteoSwiss APIs
- Data cleaning and transformation
- Spatial interpolation using geostatistics
- Neural network training for classification and regression
- Multi-step recursive forecasting

## Features

- **Data Acquisition**: Automated download of historical weather data from MeteoSwiss
- **Data Cleaning**: Standardization of variable names, wind vector transformation, time filtering
- **Interpolation**: Kriging with External Drift for spatial interpolation of missing data
- **Machine Learning**: LSTM and LSTM-KAN hybrid models for precipitation classification and intensity prediction
- **Forecasting**: Recursive multi-step precipitation forecasts
- **CLI Interface**: Command-line tools for each pipeline step

## Installation

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Install Dependencies

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/KED_NN_pipeline.git
cd KED_NN_pipeline
pip install -e .
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate ked_nn
```

### Required Packages

Key dependencies include:
- pandas, numpy
- tensorflow, keras-efficient-kan
- gstools, pykrige (for interpolation)
- scikit-learn
- tqdm, requests

## Usage

The pipeline is controlled via a command-line interface. Each step can be run independently.

### 1. Download Data

Download raw weather data for specified stations:

```bash
python -m weather_forecast.cli download --stations data/valais_stations.csv --output-dir data/raw
```

### 2. Clean Data

Process and clean the downloaded CSV files:

```bash
python -m weather_forecast.cli clean --raw-dir data/raw --output-dir data/processed
```

### 3. Interpolate Missing Values

Perform spatial interpolation using Kriging:

```bash
python -m weather_forecast.cli interpolate --input-dir data/processed --output-dir data/interpolated --variogram fitted
```

### 4. Merge Variables

Combine interpolated variables into a single Parquet file:

```bash
python -m weather_forecast.cli merge --input-dir data/interpolated --output data/clean/valais_clean.parquet
```

### 5. Train Classifier

Train precipitation classification models:

```bash
python -m weather_forecast.cli train-classifier --weather-parquet data/clean/valais_clean.parquet --stations data/valais_stations.csv --model-dir data/models/classifier
```

### 6. Train Regressor

Train precipitation intensity regression models:

```bash
python -m weather_forecast.cli train-regressor --weather-parquet data/clean/valais_clean.parquet --stations data/valais_stations.csv --model-dir data/models/regressor
```

### 7. Generate Forecasts

Produce multi-step precipitation forecasts:

```bash
python -m weather_forecast.cli forecast --horizon 24 --weather-parquet data/clean/valais_clean.parquet --stations data/valais_stations.csv --classifier data/models/classifier/lstm_kan.h5 --regressor data/models/regressor/reg_lstm_kan.h5 --scaler data/models/scaler.joblib
```

## Configuration

Default paths and parameters are defined in `src/weather_forecast/config.py`. You can modify these or override via command-line arguments.

Key configuration:
- Data directories
- Model hyperparameters (batch size, epochs, learning rate)
- Interpolation settings (variogram mode)
- Forecast parameters (horizon, thresholds)

## Project Structure

```
src/weather_forecast/
├── cli.py          # Command-line interface
├── config.py       # Configuration and paths
├── download.py     # Data downloading utilities
├── clean.py        # Data cleaning and preprocessing
├── interpolate.py  # Spatial interpolation with KED
├── merge.py        # Variable merging into Parquet
├── classifier.py   # Precipitation classification models
├── regressor.py    # Precipitation regression models
├── forecast.py     # Forecasting pipeline
└── utils.py        # Utility functions
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] Add example notebooks demonstrating the pipeline
- [ ] Add sample data
- [ ] Implement additional neural network architectures (Transformers, CNNs)
- [ ] Add support for real-time data ingestion
- [ ] Create visualization tools for forecast results
- [ ] Add unit tests and integration tests
- [ ] Optimize performance for larger datasets
- [ ] Add Docker containerization
- [ ] Implement model serving with FastAPI
- [ ] Add hyperparameter tuning capabilities
- [ ] Support for additional weather variables
- [ ] Cross-validation for model evaluation
