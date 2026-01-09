# 🌍 Earth Tamagotchi

A machine learning project that forecasts global CO₂ concentration and temperature anomalies, providing a "Tamagotchi-style" visualization of Earth's health through an interactive Streamlit dashboard.

## Overview

Earth Tamagotchi is an end-to-end ML pipeline that:

1. **Ingests** historical climate data from NOAA (CO₂) and NASA (Temperature)
2. **Engineers** features using lag and rolling window statistics
3. **Trains** forecasting models using a two-stage approach (trend + residual) for both CO₂ and temperature
4. **Generates** 24-month autoregressive forecasts
5. **Monitors** prediction accuracy through hindcast evaluation
6. **Visualizes** results through an interactive Streamlit dashboard with an Earth Health Index (EHI)

The project uses **Hopsworks** as the feature store and model registry, enabling automated retraining and monitoring workflows.

## Features

### Data Pipeline
- **Historical Data Backfill**: One-time ingestion of complete historical datasets
- **Incremental Updates**: Monthly pipeline to fetch and process new data
- **Feature Engineering**: Lag features (1, 2, 3, 6, 12 months) and rolling means (3, 12 months)
- **Time-based Features**: Cyclical seasonality encoding and normalized year trends

### Machine Learning Models

#### CO₂ Model (Trend + Residual)
- **Trend Model**: Linear regression on time-based features (year, month, seasonality)
- **Residual Model**: Gradient Boosting Regressor on lag/rolling features
- **Bias Correction**: Post-processing adjustment for systematic errors
- **Year Normalization**: Consistent scaling across full historical range

#### Temperature Model (Trend + Residual)
- **Trend Model**: Linear regression on time-based features (year, month, seasonality)
- **Residual Model**: Gradient Boosting Regressor on lag/rolling features
- **Bias Correction**: Post-processing adjustment for systematic errors
- **Year Normalization**: Consistent scaling across full historical range
- **Same Architecture as CO₂**: Two-stage approach for capturing both long-term trends and short-term fluctuations

### Forecasting
- **24-Month Autoregressive Forecasts**: Multi-step ahead predictions
- **Automatic Backfilling**: One-time generation of historical predictions for hindcast evaluation
- **Monitoring**: Comparison of predictions vs actuals as data arrives

### Dashboard (`app.py`)
- **🌍 Earth Tamagotchi Tab**: 
  - Visual Earth representation based on health status
  - Earth Health Index (EHI) with colored progress bar
  - Mood (Good/Fair/Poor) and Trend (Improving/Stable/Worsening) indicators
  - 24-month EHI forecast trajectory
- **🌫️ CO₂ Forecast Tab**: 24-month CO₂ concentration predictions
- **🌡️ Temperature Forecast Tab**: 24-month temperature anomaly predictions
- **📊 Hindcast Tabs**: Prediction accuracy evaluation with MSE, MAE, R² metrics

### Earth Health Index (EHI)
EHI is calculated using a normalization formula:

```
EHI = 1 - [ ( (CO₂ - 280) / (500 - 280) ) + ( (Temp - 0) / (2 - 0) ) ] / 2
```

- **Range**: 0-100 (higher is better)
- **Parameters**: 
  - CO₂: 280 ppm (pre-industrial) to 500 ppm (dangerous threshold)
  - Temperature: 0°C (baseline) to 2°C (dangerous threshold)
- **Uses**: Predicted values at end of 24-month forecast



## Workflow

1. **Initial Setup** (One-time):
   - Run `feature_backfill.ipynb` to populate historical data
   - Run `training_pipeline.ipynb` to train and register models

2. **Monthly Updates** (Automated via GitHub Actions):
   - `feature_pipeline.ipynb`: Fetches new data and updates feature store
   - `batch_inference.ipynb`: Generates new forecasts and stores predictions

3. **Continuous Monitoring**:
   - Streamlit dashboard displays latest forecasts and hindcasts
   - Hindcast evaluation compares predictions with actuals as they arrive

## Data Sources

- **CO₂ Data**: [NOAA GML Global Monthly Mean CO₂](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt)
- **Temperature Data**: [NASA GISTEMP Global Land-Ocean Temperature Anomaly](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)

## License

This project is part of an KTH academic course (ID2223 - Scalable Machine Learning and Deep Learning).

---
