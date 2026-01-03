import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hopsworks
import os
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv
import numpy as np

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Earth Tamagotchi Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌍 Earth Tamagotchi Dashboard")
st.markdown("**Global CO₂ and Temperature Anomaly Forecasting**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Hopsworks API Key",
        type="password",
        help="Enter your Hopsworks API key. You can also set it as HOPSWORKS_API_KEY environment variable."
    )
    
    if not api_key:
        # Try to load from environment
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Please enter your Hopsworks API key")
        st.stop()

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def connect_to_hopsworks():
    """Connect to Hopsworks and get feature store"""
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project="EarthTamagotchi",
        api_key_value=api_key
    )
    fs = project.get_feature_store()
    return fs

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_co2_forecast(_fs):
    """Get CO2 predictions from monitoring feature group - all 24 months from most recent forecast"""
    try:
        co2_monitor_fg = _fs.get_feature_group(
            name='co2_predictions',
            version=3,
        )
        # Get ALL predictions (all 24 months, not just 1-month ahead)
        monitoring_df = co2_monitor_fg.read()
        
        if len(monitoring_df) > 0:
            monitoring_df['date'] = pd.to_datetime(monitoring_df['date'])
            # Get most recent forecast date
            latest_forecast_date = monitoring_df['forecast_date'].max()
            # Get all 24 months from the most recent forecast
            latest_forecast = monitoring_df[monitoring_df['forecast_date'] == latest_forecast_date].copy()
            # Sort by date to get chronological order (1 month ahead, 2 months ahead, etc.)
            latest_forecast = latest_forecast.sort_values('date')
            return latest_forecast, latest_forecast_date
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Error fetching CO2 forecast: {str(e)}")
        return pd.DataFrame(), None

@st.cache_data(ttl=3600)
def get_co2_hindcast(_fs):
    """Get CO2 hindcast (predictions vs actuals)"""
    try:
        # Get predictions
        co2_monitor_fg = _fs.get_feature_group(
            name='co2_predictions',
            version=3,
        )
        monitoring_df = co2_monitor_fg.filter(
            co2_monitor_fg.days_before_forecast_day == 1
        ).read()
        
        if len(monitoring_df) == 0:
            return pd.DataFrame()
        
        # Get actuals
        co2_fg = _fs.get_feature_group(name='global_co2', version=1)
        co2_actual_df = co2_fg.read()
        co2_actual_df['date'] = pd.to_datetime(co2_actual_df['date'])
        
        # Merge
        outcome_df = co2_actual_df[['date', 'average']].rename(columns={'average': 'actual_co2'})
        preds_df = monitoring_df[['date', 'predicted_co2']].drop_duplicates(subset=['date'], keep='last')
        hindcast_df = pd.merge(preds_df, outcome_df, on="date", how="inner")
        hindcast_df = hindcast_df.sort_values(by=['date'])
        
        return hindcast_df
    except Exception as e:
        st.error(f"Error fetching CO2 hindcast: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_temp_forecast(_fs):
    """Get Temperature predictions from monitoring feature group - all 24 months from most recent forecast"""
    try:
        temp_monitor_fg = _fs.get_feature_group(
            name='temperature_predictions',
            version=3,
        )
        # Get ALL predictions (all 24 months, not just 1-month ahead)
        monitoring_df = temp_monitor_fg.read()
        
        if len(monitoring_df) > 0:
            monitoring_df['date'] = pd.to_datetime(monitoring_df['date'])
            # Get most recent forecast date
            latest_forecast_date = monitoring_df['forecast_date'].max()
            # Get all 24 months from the most recent forecast
            latest_forecast = monitoring_df[monitoring_df['forecast_date'] == latest_forecast_date].copy()
            # Sort by date to get chronological order (1 month ahead, 2 months ahead, etc.)
            latest_forecast = latest_forecast.sort_values('date')
            return latest_forecast, latest_forecast_date
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Error fetching Temperature forecast: {str(e)}")
        return pd.DataFrame(), None

@st.cache_data(ttl=3600)
def get_temp_hindcast(_fs):
    """Get Temperature hindcast (predictions vs actuals)"""
    try:
        # Get predictions
        temp_monitor_fg = _fs.get_feature_group(
            name='temperature_predictions',
            version=3,
        )
        monitoring_df = temp_monitor_fg.filter(
            temp_monitor_fg.days_before_forecast_day == 1
        ).read()
        
        if len(monitoring_df) == 0:
            return pd.DataFrame()
        
        # Get actuals
        temp_fg = _fs.get_feature_group(name='global_temperature', version=1)
        temp_actual_df = temp_fg.read()
        temp_actual_df['date'] = pd.to_datetime(temp_actual_df['date'])
        
        # Merge
        outcome_df = temp_actual_df[['date', 'temp_anomaly']].rename(columns={'temp_anomaly': 'actual_temp_anomaly'})
        preds_df = monitoring_df[['date', 'predicted_temp_anomaly']].drop_duplicates(subset=['date'], keep='last')
        hindcast_df = pd.merge(preds_df, outcome_df, on="date", how="inner")
        hindcast_df = hindcast_df.sort_values(by=['date'])
        
        return hindcast_df
    except Exception as e:
        st.error(f"Error fetching Temperature hindcast: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_latest_actuals(_fs):
    """Get latest actual CO2 and temperature values"""
    try:
        # Get latest CO2
        co2_fg = _fs.get_feature_group(name='global_co2', version=1)
        co2_df = co2_fg.read()
        co2_df['date'] = pd.to_datetime(co2_df['date'])
        latest_co2 = co2_df.sort_values('date').iloc[-1]
        
        # Get latest temperature
        temp_fg = _fs.get_feature_group(name='global_temperature', version=1)
        temp_df = temp_fg.read()
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        latest_temp = temp_df.sort_values('date').iloc[-1]
        
        return {
            'co2_date': latest_co2['date'],
            'co2_value': latest_co2['average'],
            'temp_date': latest_temp['date'],
            'temp_value': latest_temp['temp_anomaly']
        }
    except Exception as e:
        st.error(f"Error fetching latest actuals: {str(e)}")
        return None

def compute_ehi_timeseries(co2_forecast, temp_forecast):
    """
    Compute EHI for each month in the forecast to create a time series.
    Returns a DataFrame with date and ehi columns.
    """
    if len(co2_forecast) == 0 or len(temp_forecast) == 0:
        return pd.DataFrame()
    
    # Normalization bounds
    CO2_MIN = 280
    CO2_MAX = 500
    TEMP_MIN = 0
    TEMP_MAX = 2
    
    # Align dates
    co2_forecast = co2_forecast.copy()
    temp_forecast = temp_forecast.copy()
    co2_forecast['date'] = pd.to_datetime(co2_forecast['date'])
    temp_forecast['date'] = pd.to_datetime(temp_forecast['date'])
    
    # Merge on date
    merged = pd.merge(
        co2_forecast[['date', 'predicted_co2']],
        temp_forecast[['date', 'predicted_temp_anomaly']],
        on='date',
        how='inner'
    )
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Compute EHI for each row
    co2_normalized = (merged['predicted_co2'] - CO2_MIN) / (CO2_MAX - CO2_MIN)
    co2_normalized = co2_normalized.clip(0, 1)
    
    temp_normalized = (merged['predicted_temp_anomaly'] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    temp_normalized = temp_normalized.clip(0, 1)
    
    ehi = 1 - (co2_normalized + temp_normalized) / 2
    ehi_100 = ehi * 100
    
    return pd.DataFrame({
        'date': merged['date'],
        'ehi': ehi_100
    })

def compute_ehi(co2_forecast, temp_forecast):
    """
    Compute Earth Health Index (EHI) using normalization formula.
    
    Formula: EHI = 1 - [ ( (CO2 - CO2_min) / (CO2_max - CO2_min) ) + ( (Temp - Temp_min) / (Temp_max - Temp_min) ) ] / 2
    
    Uses predicted values at end of 24-month forecast to reflect future trajectory.
    EHI ranges from 0-1 (or 0-100 when multiplied by 100), where:
    - Higher is better (lower CO2, lower temperature)
    - 1.0 (100) = best possible (CO2_min, Temp_min)
    - 0.0 (0) = worst possible (CO2_max, Temp_max)
    """
    if len(co2_forecast) == 0 or len(temp_forecast) == 0:
        return None, None, None
    
    # Get predicted values at end of 24-month forecast (future trajectory)
    co2_values = co2_forecast['predicted_co2'].values
    temp_values = temp_forecast['predicted_temp_anomaly'].values
    
    # Use the END value (24 months ahead) to reflect predicted future state
    co2_predicted = co2_values[-1]  # CO2 at end of forecast
    temp_predicted = temp_values[-1]  # Temperature at end of forecast
    
    # Normalization bounds
    CO2_MIN = 280  # Pre-industrial level (~280 ppm)
    CO2_MAX = 500  # Dangerous threshold
    TEMP_MIN = 0   # No anomaly (baseline)
    TEMP_MAX = 2   # Dangerous threshold (°C anomaly)
    
    # Normalize CO2 to 0-1 scale (0 = best, 1 = worst)
    co2_normalized = (co2_predicted - CO2_MIN) / (CO2_MAX - CO2_MIN)
    co2_normalized = max(0, min(1, co2_normalized))  # Clamp to [0, 1]
    
    # Normalize Temperature to 0-1 scale (0 = best, 1 = worst)
    temp_normalized = (temp_predicted - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    temp_normalized = max(0, min(1, temp_normalized))  # Clamp to [0, 1]
    
    # Calculate EHI: 1 - average of normalized values
    # Lower normalized values = higher EHI (better health)
    ehi = 1 - (co2_normalized + temp_normalized) / 2
    
    # Convert to 0-100 scale for display
    ehi_100 = ehi * 100
    
    # Calculate change rates for display
    co2_start = co2_values[0]
    co2_end = co2_values[-1]
    co2_change_rate = (co2_end - co2_start) / len(co2_values)
    
    temp_start = temp_values[0]
    temp_end = temp_values[-1]
    temp_change_rate = (temp_end - temp_start) / len(temp_values)
    
    # Determine trend (comparing first 6 months vs last 6 months)
    if len(co2_values) >= 12:
        co2_first_half = np.mean(co2_values[:6])
        co2_second_half = np.mean(co2_values[-6:])
        co2_trend = co2_second_half - co2_first_half
        
        temp_first_half = np.mean(temp_values[:6])
        temp_second_half = np.mean(temp_values[-6:])
        temp_trend = temp_second_half - temp_first_half
        
        # Overall trend: negative is good (decreasing), positive is bad (increasing)
        overall_trend = -(co2_trend / 10 + temp_trend * 10)  # Weighted combination
        
        if overall_trend > 0.1:
            trend = "improving"
        elif overall_trend < -0.1:
            trend = "worsening"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    return ehi_100, trend, {
        'co2_predicted': co2_predicted,
        'temp_predicted': temp_predicted,
        'co2_normalized': co2_normalized,
        'temp_normalized': temp_normalized,
        'co2_change_rate': co2_change_rate,
        'temp_change_rate': temp_change_rate,
        'co2_score': (1 - co2_normalized) * 50,  # Component score for display
        'temp_score': (1 - temp_normalized) * 50
    }

def get_mood_image_path(ehi, trend):
    """
    Get image path based on EHI and trend. Returns None if image doesn't exist.
    
    Mapping:
    - improving + ehi >= 70: excellent_improving.png
    - improving + ehi >= 50: good_improving.png
    - improving + ehi < 50: fair_improving.png
    - worsening + ehi >= 70: good_worsening.png
    - worsening + ehi >= 50: fair_worsening.png
    - worsening + ehi < 50: poor_worsening.png
    - stable + ehi >= 70: excellent_stable.png
    - stable + ehi >= 50: good_stable.png
    - stable + ehi < 50: fair_stable.png
    """
    import os
    base_path = "data/images"
    
    if ehi is None:
        return None
    
    # Map mood to image filename based on EHI and trend
    if trend == "improving":
        if ehi >= 70:
            filename = "excellent_improving.png"
        elif ehi >= 50:
            filename = "good_improving.png"
        else:
            filename = "fair_improving.png"
    elif trend == "worsening":
        if ehi >= 70:
            filename = "good_worsening.png"
        elif ehi >= 50:
            filename = "fair_worsening.png"
        else:
            filename = "poor_worsening.png"
    else:  # stable
        if ehi >= 70:
            filename = "excellent_stable.png"
        elif ehi >= 50:
            filename = "good_stable.png"
        else:
            filename = "fair_stable.png"
    
    image_path = os.path.join(base_path, filename)
    if os.path.exists(image_path):
        return image_path
    return None

def get_mood_color_and_text(ehi, trend):
    """Get color and mood text based on EHI and trend"""
    if ehi is None:
        return "gray", "Unknown"
    
    if trend == "improving":
        if ehi >= 70:
            return "green", "Excellent & Improving"
        elif ehi >= 50:
            return "lightgreen", "Good & Improving"
        else:
            return "yellow", "Fair & Improving"
    elif trend == "worsening":
        if ehi >= 70:
            return "orange", "Good but Worsening"
        elif ehi >= 50:
            return "darkorange", "Fair but Worsening"
        else:
            return "red", "Poor & Worsening"
    else:  # stable
        if ehi >= 70:
            return "green", "Excellent & Stable"
        elif ehi >= 50:
            return "yellow", "Good & Stable"
        else:
            return "orange", "Fair & Stable"

# Main app
if api_key:
    try:
        fs = connect_to_hopsworks()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌍 Earth Tamagotchi", 
            "🌫️ CO₂ Forecast", 
            "🌡️ Temperature Forecast", 
            "📊 CO₂ Hindcast", 
            "📊 Temperature Hindcast"
        ])
        
        with tab1:
            st.header("🌍 Earth Tamagotchi")
            st.markdown("**Earth Health Index (EHI) Dashboard**")
            
            # Get forecasts and latest actuals
            co2_forecast, co2_forecast_date = get_co2_forecast(fs)
            temp_forecast, temp_forecast_date = get_temp_forecast(fs)
            latest_actuals = get_latest_actuals(fs)
            
            # Compute EHI
            ehi, trend, ehi_details = compute_ehi(co2_forecast, temp_forecast)
            color, mood_text = get_mood_color_and_text(ehi, trend)
            image_path = get_mood_image_path(ehi, trend)
            
            # Display Earth Tamagotchi
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Display the Earth Tamagotchi image
                if image_path:
                    # Center the image using CSS
                    import base64
                    with open(image_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    st.markdown(
                        f'<div style="text-align: center;"><img src="data:image/png;base64,{img_data}" width="300" style="display: inline-block;"></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("⚠️ Image not found for current mood")
                st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {color};'>{mood_text}</div>", unsafe_allow_html=True)
                
                if ehi is not None:
                    st.markdown(f"<div style='text-align: center; font-size: 48px; font-weight: bold; margin: 20px 0;'>EHI: {ehi:.1f}/100</div>", unsafe_allow_html=True)
                    
                    # Progress bar for EHI
                    st.progress(ehi / 100)
                    
                    # Trend indicator
                    trend_arrow = "📈" if trend == "improving" else "📉" if trend == "worsening" else "➡️"
                    st.markdown(f"<div style='text-align: center; font-size: 20px;'>Trend: {trend_arrow} {trend.capitalize()}</div>", unsafe_allow_html=True)
            
            # Latest values
            st.markdown("---")
            st.subheader("📊 Latest Observed Values")
            
            if latest_actuals:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "CO₂ Concentration",
                        f"{latest_actuals['co2_value']:.2f} ppm",
                        help=f"Last updated: {pd.to_datetime(latest_actuals['co2_date']).strftime('%Y-%m-%d')}"
                    )
                with col2:
                    st.metric(
                        "Temperature Anomaly",
                        f"{latest_actuals['temp_value']:.2f} °C",
                        help=f"Last updated: {pd.to_datetime(latest_actuals['temp_date']).strftime('%Y-%m-%d')}"
                    )
            
            # EHI Details
            if ehi_details:
                st.markdown("---")
                st.subheader("🔍 EHI Components")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CO₂ (24mo ahead)", f"{ehi_details['co2_predicted']:.2f} ppm")
                with col2:
                    st.metric("CO₂ Normalized", f"{ehi_details['co2_normalized']:.3f}")
                with col3:
                    st.metric("Temp (24mo ahead)", f"{ehi_details['temp_predicted']:.2f} °C")
                with col4:
                    st.metric("Temp Normalized", f"{ehi_details['temp_normalized']:.3f}")
                
                # Show change rates
                st.markdown("**Change Rates:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CO₂ Change Rate", f"{ehi_details['co2_change_rate']:.3f} ppm/month")
                with col2:
                    st.metric("Temp Change Rate", f"{ehi_details['temp_change_rate']:.4f} °C/month")
                
                # Explanation
                with st.expander("ℹ️ How is EHI calculated?"):
                    st.markdown("""
                    **Earth Health Index (EHI)** uses a normalization formula:
                    
                    ```
                    EHI = 1 - [ ( (CO₂ - CO₂_min) / (CO₂_max - CO₂_min) ) 
                              + ( (Temp - Temp_min) / (Temp_max - Temp_min) ) ] / 2
                    ```
                    
                    **Parameters:**
                    - CO₂ bounds: 280 ppm (pre-industrial) to 500 ppm (dangerous threshold)
                    - Temperature bounds: 0°C (baseline) to 2°C (dangerous threshold)
                    - Uses predicted values at **end of 24-month forecast** (future trajectory)
                    
                    **Interpretation:**
                    - **EHI = 100**: Best possible (CO₂ = 280 ppm, Temp = 0°C)
                    - **EHI = 50**: Midpoint (CO₂ = 390 ppm, Temp = 1°C)
                    - **EHI = 0**: Worst possible (CO₂ = 500 ppm, Temp = 2°C)
                    
                    **Trend**: Compares first 6 months vs last 6 months of forecast
                    - **Improving**: Decreasing CO₂ and/or temperature trends
                    - **Stable**: Minimal change
                    - **Worsening**: Increasing CO₂ and/or temperature trends
                    
                    **Mood** combines EHI score with trend direction.
                    """)
            
            # Forecast summary
            if len(co2_forecast) > 0 and len(temp_forecast) > 0:
                st.markdown("---")
                st.subheader("📈 24-Month Forecast Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    co2_start = co2_forecast['predicted_co2'].iloc[0]
                    co2_end = co2_forecast['predicted_co2'].iloc[-1]
                    co2_change = co2_end - co2_start
                    st.metric(
                        "CO₂ Forecast",
                        f"{co2_end:.2f} ppm",
                        f"{co2_change:+.2f} ppm over 24 months"
                    )
                
                with col2:
                    temp_start = temp_forecast['predicted_temp_anomaly'].iloc[0]
                    temp_end = temp_forecast['predicted_temp_anomaly'].iloc[-1]
                    temp_change = temp_end - temp_start
                    st.metric(
                        "Temperature Forecast",
                        f"{temp_end:.2f} °C",
                        f"{temp_change:+.2f} °C over 24 months"
                    )
                
                # EHI Forecast Graph
                st.markdown("---")
                st.subheader("📊 EHI Forecast: 24-Month Trajectory")
                
                ehi_timeseries = compute_ehi_timeseries(co2_forecast, temp_forecast)
                if len(ehi_timeseries) > 0:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(
                        ehi_timeseries['date'],
                        ehi_timeseries['ehi'],
                        'o-',
                        color='#1EB182',
                        label='Predicted EHI',
                        linewidth=2,
                        markersize=6
                    )
                    # Add horizontal reference lines
                    ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Excellent (70)')
                    ax.axhline(y=50, color='yellow', linestyle='--', alpha=0.3, label='Good (50)')
                    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.3, label='Fair (30)')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Earth Health Index (EHI)', fontsize=12)
                    ax.set_title('EHI Forecast: Predicted Earth Health Over 24 Months', fontsize=14, fontweight='bold')
                    ax.set_ylim(0, 100)
                    ax.legend(loc='best', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    fig.autofmt_xdate()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show EHI change over forecast period
                    ehi_start = ehi_timeseries['ehi'].iloc[0]
                    ehi_end = ehi_timeseries['ehi'].iloc[-1]
                    ehi_change = ehi_end - ehi_start
                    st.metric(
                        "EHI Forecast Change",
                        f"{ehi_end:.1f}",
                        f"{ehi_change:+.1f} over 24 months (from {ehi_start:.1f})"
                    )
            
            if ehi is None:
                st.warning("⚠️ Cannot compute EHI: Forecast data not available. Run the batch inference pipeline first.")
        
        with tab2:
            st.header("CO₂ Concentration Forecast")
            st.markdown("**24-Month Autoregressive Prediction**")
            
            co2_forecast, co2_forecast_date = get_co2_forecast(fs)
            
            if co2_forecast_date:
                st.caption(f"📅 Forecast generated on: {pd.to_datetime(co2_forecast_date).strftime('%Y-%m-%d')}")
            
            if len(co2_forecast) > 0:
                # Plot forecast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(
                    co2_forecast['date'], 
                    co2_forecast['predicted_co2'], 
                    'o-', 
                    color='#1EB182', 
                    label='Predicted CO₂', 
                    linewidth=2,
                    markersize=6
                )
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('CO₂ Concentration (ppm)', fontsize=12)
                ax.set_title('CO₂ Forecast: 24-Month Autoregressive Prediction', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                with st.expander("View Forecast Data"):
                    st.dataframe(co2_forecast[['date', 'predicted_co2']].style.format({
                        'predicted_co2': '{:.2f}'
                    }))
            else:
                st.info("No CO₂ forecast data available yet. Run the batch inference pipeline to generate forecasts.")
        
        with tab3:
            st.header("Temperature Anomaly Forecast")
            st.markdown("**24-Month Autoregressive Prediction**")
            
            temp_forecast, temp_forecast_date = get_temp_forecast(fs)
            
            if temp_forecast_date:
                st.caption(f"📅 Forecast generated on: {pd.to_datetime(temp_forecast_date).strftime('%Y-%m-%d')}")
            
            if len(temp_forecast) > 0:
                # Plot forecast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(
                    temp_forecast['date'], 
                    temp_forecast['predicted_temp_anomaly'], 
                    'o-', 
                    color='#ff5f27', 
                    label='Predicted Temperature Anomaly', 
                    linewidth=2,
                    markersize=6
                )
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
                ax.set_title('Temperature Forecast: 24-Month Autoregressive Prediction', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                with st.expander("View Forecast Data"):
                    st.dataframe(temp_forecast[['date', 'predicted_temp_anomaly']].style.format({
                        'predicted_temp_anomaly': '{:.2f}'
                    }))
            else:
                st.info("No Temperature forecast data available yet. Run the batch inference pipeline to generate forecasts.")
        
        with tab4:
            st.header("CO₂ Hindcast: Predicted vs Actual")
            st.markdown("**1-Month Ahead Forecast Evaluation**")
            
            co2_hindcast = get_co2_hindcast(fs)
            
            if len(co2_hindcast) > 0:
                # Plot hindcast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(
                    co2_hindcast['date'], 
                    co2_hindcast['actual_co2'], 
                    'o-', 
                    color='blue', 
                    label='Actual CO₂', 
                    linewidth=2, 
                    markersize=6
                )
                ax.plot(
                    co2_hindcast['date'], 
                    co2_hindcast['predicted_co2'], 
                    's-', 
                    color='#ff5f27', 
                    label='Predicted CO₂ (1-month ahead)', 
                    linewidth=2, 
                    markersize=6
                )
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('CO₂ Concentration (ppm)', fontsize=12)
                ax.set_title('CO₂ Hindcast: Predicted vs Actual (1-Month Ahead Forecast)', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(co2_hindcast['actual_co2'], co2_hindcast['predicted_co2'])
                mae = mean_absolute_error(co2_hindcast['actual_co2'], co2_hindcast['predicted_co2'])
                r2 = r2_score(co2_hindcast['actual_co2'], co2_hindcast['predicted_co2'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae:.2f} ppm")
                with col3:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Show data table
                with st.expander("View Hindcast Data"):
                    st.dataframe(co2_hindcast.style.format({
                        'predicted_co2': '{:.2f}',
                        'actual_co2': '{:.2f}'
                    }))
            else:
                st.info("No CO₂ hindcast data available yet. This is normal - predictions need time to accumulate actual values for comparison.")
        
        with tab5:
            st.header("Temperature Hindcast: Predicted vs Actual")
            st.markdown("**1-Month Ahead Forecast Evaluation**")
            
            temp_hindcast = get_temp_hindcast(fs)
            
            if len(temp_hindcast) > 0:
                # Plot hindcast
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(
                    temp_hindcast['date'], 
                    temp_hindcast['actual_temp_anomaly'], 
                    'o-', 
                    color='blue', 
                    label='Actual Temperature Anomaly', 
                    linewidth=2, 
                    markersize=6
                )
                ax.plot(
                    temp_hindcast['date'], 
                    temp_hindcast['predicted_temp_anomaly'], 
                    's-', 
                    color='#ff5f27', 
                    label='Predicted Temperature Anomaly (1-month ahead)', 
                    linewidth=2, 
                    markersize=6
                )
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12)
                ax.set_title('Temperature Hindcast: Predicted vs Actual (1-Month Ahead Forecast)', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(temp_hindcast['actual_temp_anomaly'], temp_hindcast['predicted_temp_anomaly'])
                mae = mean_absolute_error(temp_hindcast['actual_temp_anomaly'], temp_hindcast['predicted_temp_anomaly'])
                r2 = r2_score(temp_hindcast['actual_temp_anomaly'], temp_hindcast['predicted_temp_anomaly'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae:.3f} °C")
                with col3:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Show data table
                with st.expander("View Hindcast Data"):
                    st.dataframe(temp_hindcast.style.format({
                        'predicted_temp_anomaly': '{:.2f}',
                        'actual_temp_anomaly': '{:.2f}'
                    }))
            else:
                st.info("No Temperature hindcast data available yet. This is normal - predictions need time to accumulate actual values for comparison.")
        
        # Footer
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {str(e)}")
        st.info("Please check your API key and ensure you have access to the EarthTamagotchi project.")

