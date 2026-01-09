# 🌍 Earth Tamagotchi

A machine learning project that forecasts global CO₂ concentration and temperature anomalies, providing a "Tamagotchi-style" visualization of Earth's health through an interactive Streamlit dashboard.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technologies](#technologies)

## 🎯 Overview

Earth Tamagotchi is an end-to-end ML pipeline that:

1. **Ingests** historical climate data from NOAA (CO₂) and NASA (Temperature)
2. **Engineers** features using lag and rolling window statistics
3. **Trains** forecasting models using a two-stage approach (trend + residual) for both CO₂ and temperature
4. **Generates** 24-month autoregressive forecasts
5. **Monitors** prediction accuracy through hindcast evaluation
6. **Visualizes** results through an interactive Streamlit dashboard with an Earth Health Index (EHI)

The project uses **Hopsworks** as the feature store and model registry, enabling automated retraining and monitoring workflows.

## ✨ Features

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

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │
│  NOAA & NASA    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Backfill│  (One-time historical data)
│   Notebook      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Pipeline│  (Monthly updates)
│   Notebook      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hopsworks      │
│  Feature Store  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training        │
│ Pipeline        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hopsworks      │
│  Model Registry │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Inference │  (Monthly forecasts)
│   Notebook      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hopsworks      │
│  Predictions FG │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit App   │  (Public dashboard)
│     app.py      │
└─────────────────┘
```

### Workflow

1. **Initial Setup** (One-time):
   - Run `feature_backfill.ipynb` to populate historical data
   - Run `training_pipeline.ipynb` to train and register models

2. **Monthly Updates** (Automated via GitHub Actions):
   - `feature_pipeline.ipynb`: Fetches new data and updates feature store
   - `batch_inference.ipynb`: Generates new forecasts and stores predictions

3. **Continuous Monitoring**:
   - Streamlit dashboard displays latest forecasts and hindcasts
   - Hindcast evaluation compares predictions with actuals as they arrive

## 📁 Project Structure

```
atwEarthTamagotchi/
├── app.py                          # Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .github/
│   └── workflows/
│       └── earth-tamagotchi-pipeline.yml  # Automated monthly pipeline
├── notebooks/
│   ├── feature_backfill.ipynb     # One-time historical data ingestion
│   ├── feature_pipeline.ipynb     # Monthly data updates
│   ├── training_pipeline.ipynb     # Model training and registration
│   ├── batch_inference.ipynb      # Forecast generation and monitoring
│   ├── co2_model/                 # CO₂ model artifacts (local)
│   └── temp_model/                # Temperature model artifacts (local)
└── data/
    └── images/                    # Earth Tamagotchi mood images
        ├── excellent_*.png
        ├── good_*.png
        ├── fair_*.png
        └── poor_*.png
```

## 🚀 Setup

### Prerequisites

- Python 3.11+
- Hopsworks account and API key
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd atwEarthTamagotchi
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   HOPSWORKS_API_KEY=your_api_key_here
   ```

## 📖 Usage

### 1. Initial Data Backfill (One-time)

Run `notebooks/feature_backfill.ipynb` to:
- Fetch complete historical CO₂ data from NOAA GML
- Fetch complete historical temperature data from NASA GISS
- Engineer features (lags, rolling means, time features)
- Store data in Hopsworks feature groups

**Expected output**: 
- `global_co2` feature group (version 1)
- `global_temperature` feature group (version 1)

### 2. Model Training

Run `notebooks/training_pipeline.ipynb` to:
- Load features from Hopsworks
- Train CO₂ model (trend + residual)
- Train temperature model (trend + residual)
- Perform grid search for optimal feature combinations
- Save models to Hopsworks model registry

**Expected output**:
- `co2_trend_residual_model` in model registry
- `global_temperature_trend_residual_model` in model registry

### 3. Monthly Feature Updates

Run `notebooks/feature_pipeline.ipynb` to:
- Fetch latest CO₂ and temperature data
- Calculate features for new data points
- Insert new records into feature groups

**Note**: This runs automatically via GitHub Actions on the 1st, 6th, 12th, 18th, 24th, and 29th of each month.

### 4. Batch Inference (Forecasting)

Run `notebooks/batch_inference.ipynb` to:
- Load trained models from registry
- Generate 24-month autoregressive forecasts
- Store predictions in monitoring feature groups
- Create hindcast evaluations (predictions vs actuals)
- Generate visualization plots

**Expected output**:
- `co2_predictions` feature group (version 3)
- `temperature_predictions` feature group (version 3)
- Hindcast plots comparing predictions with actuals

**Note**: 
- Backfill runs automatically on first execution (if no historical predictions exist)
- Backfill only runs once and never replaces existing predictions

### 5. Streamlit Dashboard

Run the interactive dashboard:
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

**Features**:
- Real-time data from Hopsworks
- Interactive plots and metrics
- Earth Health Index visualization
- No API key required for users (uses server-side key)

## 🚢 Deployment

### Streamlit Cloud

1. **Push code to GitHub**

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository and set main file: `app.py`

3. **Configure Secrets**:
   - In Streamlit Cloud Settings → Secrets, add:
   ```toml
   HOPSWORKS_API_KEY = "your-api-key-here"
   ```

4. **Access your app**:
   - Your app will be available at `https://your-app-name.streamlit.app`
   - Share the link - no authentication required for viewers!

### GitHub Actions (Automated Pipeline)

The project includes a GitHub Actions workflow (`.github/workflows/earth-tamagotchi-pipeline.yml`) that:

- **Runs automatically** on the 1st, 6th, 12th, 18th, 24th, and 29th of each month
- **Can be triggered manually** via workflow_dispatch
- **Executes**:
  1. Feature pipeline (updates data)
  2. Batch inference (generates forecasts)
  3. Archives generated plots

**Setup**:
1. Add `HOPSWORKS_API_KEY` to GitHub Secrets (Settings → Secrets and variables → Actions)
2. The workflow will run automatically on schedule

## 🛠️ Technologies

- **Python 3.11+**: Core language
- **Hopsworks 4.6**: Feature store and model registry
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning (Linear Regression, Gradient Boosting)
- **XGBoost**: Gradient boosting for residual models (both CO₂ and temperature)
- **Matplotlib**: Visualization
- **Streamlit**: Interactive web dashboard
- **Great Expectations**: Data validation (optional)
- **Jupyter**: Notebook environment
- **GitHub Actions**: CI/CD automation

## 📊 Data Sources

- **CO₂ Data**: [NOAA GML Global Monthly Mean CO₂](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt)
- **Temperature Data**: [NASA GISTEMP Global Land-Ocean Temperature Anomaly](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)

## 🔍 Key Concepts

### Autoregressive Forecasting
Predictions use previous predictions as input for future steps, creating a multi-month forecast chain.

### Hindcast Evaluation
Historical predictions are compared against actual observed values to measure model accuracy over time.

### Feature Engineering
- **Lag Features**: Previous values (1, 2, 3, 6, 12 months ago)
- **Rolling Features**: Moving averages over windows (3, 12 months)
- **Time Features**: Year, month, cyclical seasonality (sin/cos), normalized trends

### Model Architecture

**CO₂ Model**:
- Two-stage approach: Trend captures long-term patterns, Residual captures short-term fluctuations
- Year normalization ensures consistent scaling across full historical range

**Temperature Model**:
- Two-stage approach: Trend captures long-term patterns, Residual captures short-term fluctuations
- Same architecture as CO₂ model for consistency
- Year normalization ensures consistent scaling across full historical range

## 📝 Notes

- **Backfill Safety**: The batch inference notebook includes safeguards to ensure backfill only runs once and never replaces existing predictions
- **Caching**: Streamlit app caches data for 1 hour to reduce API calls
- **Versioning**: Feature groups and models are versioned in Hopsworks for reproducibility
- **Monitoring**: Predictions are stored with `days_before_forecast_day` to track forecast horizon

## 🤝 Contributing

This is a course project. For questions or issues, please refer to the project documentation or contact the maintainers.

## 📄 License

This project is part of an academic course (ID2223 - Scalable Machine Learning and Deep Learning).

---

**Built with ❤️ for Earth monitoring and climate awareness**
