import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hopsworks
import os
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

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

# Main app
if api_key:
    try:
        fs = connect_to_hopsworks()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌫️ CO₂ Forecast", 
            "🌡️ Temperature Forecast", 
            "📊 CO₂ Hindcast", 
            "📊 Temperature Hindcast"
        ])
        
        with tab1:
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
        
        with tab2:
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
        
        with tab3:
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
        
        with tab4:
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
        st.markdown("**Earth Tamagotchi** - Global Climate Forecasting Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {str(e)}")
        st.info("Please check your API key and ensure you have access to the EarthTamagotchi project.")

