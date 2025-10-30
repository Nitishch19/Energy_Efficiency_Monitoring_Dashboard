# =============================================================================
# ENERGY EFFICIENCY MONITORING DASHBOARD - SDG 9 & 11
# Updated with Excel Upload and Currency Selection Features
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime, timedelta

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Energy Efficiency Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",  # Use full width of browser
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
.main-header {
    color: #1f4e79;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}

.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}

.energy-alert {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
    border-radius: 5px;
}

.efficiency-tip {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-left: 4px solid #17a2b8;
    margin: 1rem 0;
    border-radius: 5px;
}

.savings-highlight {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CURRENCY SELECTION FEATURE (NEW)
# =============================================================================
st.sidebar.markdown("### 💰 Currency Settings")

# Let user choose between USD and INR
currency = st.sidebar.selectbox(
    "Select Currency", 
    ["USD ($)", "INR (₹)"],
    help="Choose your preferred currency for cost display"
)

# Set exchange rate (user can adjust if needed)
if currency == "INR (₹)":
    convert_rate = st.sidebar.number_input(
        "Exchange Rate (₹ per $)", 
        value=83.0,  # Default exchange rate
        min_value=1.0,
        max_value=200.0,
        step=0.5,
        help="Adjust the USD to INR exchange rate"
    )
else:
    convert_rate = 1.0  # No conversion for USD

# Helper function to convert USD to selected currency
def convert_currency(value_usd):
    """Convert USD value to selected currency (INR or USD)"""
    return value_usd * convert_rate if currency == "INR (₹)" else value_usd

# Helper function to format money with correct currency symbol
def fmt_money(amount):
    """Format amount with appropriate currency symbol"""
    if currency == "INR (₹)":
        return f"₹{amount:,.2f}"
    else:
        return f"${amount:,.2f}"

# =============================================================================
# EXCEL FILE UPLOAD FEATURE (NEW)
# =============================================================================
st.sidebar.markdown("### 📁 Data Upload")
st.sidebar.markdown("Upload your own Excel file or use sample data")

# File uploader widget - accepts .xlsx and .xls files
uploaded_file = st.sidebar.file_uploader(
    "Choose an Excel file",
    type=['xlsx', 'xls'],
    help="Upload Excel file with columns: date, building, floor, room, energy_kwh, cost_usd, carbon_kg, efficiency_percent"
)

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================
@st.cache_data
def load_energy_data(uploaded_file=None):
    """
    Load energy monitoring data from Excel file or use sample data
    
    Parameters:
    -----------
    uploaded_file : UploadedFile or None
        Excel file uploaded by user, or None to use sample data
        
    Returns:
    --------
    pandas.DataFrame
        Energy consumption data with required columns
    """
    
    # If user uploaded a file, read it
    if uploaded_file is not None:
        try:
            # Read Excel file - automatically detects the first sheet
            df = pd.read_excel(uploaded_file)
            
            # Validate required columns exist
            required_cols = ['date', 'building', 'floor', 'room', 'energy_kwh', 'cost_usd', 'carbon_kg', 'efficiency_percent']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: date, building, floor, room, energy_kwh, cost_usd, carbon_kg, efficiency_percent")
                return pd.DataFrame()  # Return empty dataframe
            
            # Success message
            st.sidebar.success(f"✅ Loaded {len(df)} records from Excel file")
            return df
            
        except Exception as e:
            st.error(f"❌ Error reading Excel file: {str(e)}")
            return pd.DataFrame()
    
    # If no file uploaded, use sample embedded data
    else:
        csv_data = """date,building,floor,room,energy_kwh,cost_usd,carbon_kg,efficiency_percent
2024-10-01,Office Building A,Ground Floor,Room 1,145.5,17.46,58.2,87.3
2024-10-01,Office Building A,Ground Floor,Room 2,162.3,19.48,64.9,85.1
2024-10-01,Office Building A,First Floor,Room 1,158.7,19.04,63.5,86.8
2024-10-01,Office Building A,First Floor,Room 2,143.2,17.18,57.3,88.5
2024-10-01,Office Building A,Second Floor,Room 1,139.8,16.78,55.9,89.2
2024-10-01,Residential Complex B,Ground Floor,Room 1,89.4,10.73,35.8,82.1
2024-10-01,Residential Complex B,Ground Floor,Room 2,94.6,11.35,37.8,81.7
2024-10-01,Residential Complex B,First Floor,Room 1,87.3,10.48,34.9,83.4
2024-10-01,Residential Complex B,First Floor,Room 2,92.1,11.05,36.8,82.9
2024-10-01,Commercial Plaza C,Ground Floor,Room 1,203.4,24.41,81.4,79.6
2024-10-01,Commercial Plaza C,Ground Floor,Room 2,198.7,23.84,79.5,80.3
2024-10-01,Commercial Plaza C,First Floor,Room 1,215.6,25.87,86.2,78.1
2024-10-01,Commercial Plaza C,First Floor,Room 2,207.9,24.95,83.2,79.0
2024-10-02,Office Building A,Ground Floor,Room 1,152.3,18.28,60.9,86.5
2024-10-02,Office Building A,Ground Floor,Room 2,167.8,20.14,67.1,84.3
2024-10-02,Office Building A,First Floor,Room 1,161.9,19.43,64.8,86.1
2024-10-02,Office Building A,First Floor,Room 2,148.6,17.83,59.4,87.8
2024-10-02,Residential Complex B,Ground Floor,Room 1,91.7,11.00,36.7,81.8
2024-10-02,Residential Complex B,Ground Floor,Room 2,96.2,11.54,38.5,81.4
2024-10-02,Residential Complex B,First Floor,Room 1,89.8,10.78,35.9,83.1
2024-10-02,Commercial Plaza C,Ground Floor,Room 1,208.1,24.97,83.2,79.2
2024-10-02,Commercial Plaza C,First Floor,Room 1,219.4,26.33,87.8,77.8
2024-10-03,Office Building A,Ground Floor,Room 1,147.2,17.66,58.9,87.8
2024-10-03,Office Building A,First Floor,Room 1,159.4,19.13,63.8,86.6
2024-10-03,Residential Complex B,Ground Floor,Room 1,88.9,10.67,35.6,82.5
2024-10-03,Commercial Plaza C,Ground Floor,Room 1,205.7,24.68,82.3,79.9
2024-10-04,Office Building A,Ground Floor,Room 1,149.8,17.98,59.9,87.1
2024-10-04,Residential Complex B,Ground Floor,Room 1,90.3,10.84,36.1,82.2
2024-10-04,Commercial Plaza C,Ground Floor,Room 1,202.9,24.35,81.2,80.4
2024-10-05,Office Building A,Ground Floor,Room 1,151.4,18.17,60.6,86.9
2024-10-05,Residential Complex B,Ground Floor,Room 1,92.1,11.05,36.8,81.9"""
        
        st.sidebar.info("📊 Using sample data. Upload Excel file to analyze your own data.")
        return pd.read_csv(io.StringIO(csv_data))

# =============================================================================
# LOAD APPLIANCE DATA
# =============================================================================
@st.cache_data
def load_appliance_data():
    """Load appliance-level consumption and efficiency data"""
    appliances_csv = """appliance,avg_daily_kwh,efficiency_rating,potential_savings_percent
Air Conditioner,45.6,82,25
LED Lights,1.8,95,8
Computer/Laptop,3.2,88,12
Refrigerator,8.4,79,18
Water Heater,28.3,75,30
Fan,1.5,91,5
Projector,4.2,85,15
Coffee Machine,6.7,73,22"""
    return pd.read_csv(io.StringIO(appliances_csv))

# =============================================================================
# PREDICTIVE MODEL TRAINING
# =============================================================================
@st.cache_data
def train_prediction_model(df):
    """
    Train Random Forest model to predict energy consumption
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical energy data
        
    Returns:
    --------
    tuple : (model, feature_columns)
        Trained model and list of feature column names
    """
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract time-based features
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['date'].dt.month  # 1-12
    
    # Encode categorical variables (building and floor) as dummy variables
    building_encoded = pd.get_dummies(df['building'], prefix='building')
    floor_encoded = pd.get_dummies(df['floor'], prefix='floor')
    
    # Combine all features into single dataframe
    features = pd.concat([
        df[['day_of_week', 'month', 'efficiency_percent']],
        building_encoded,
        floor_encoded
    ], axis=1)
    
    # Prepare training data
    X = features
    y = df['energy_kwh']
    
    # Split into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns.tolist()

# =============================================================================
# LOAD AND PROCESS DATA
# =============================================================================
# Load main energy data (from Excel or sample)
df = load_energy_data(uploaded_file)

# Only proceed if we have valid data
if not df.empty:
    # Add currency-converted cost column
    df['cost_local'] = df['cost_usd'].apply(convert_currency)
    
    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Load appliance reference data
    appliance_df = load_appliance_data()
    
    # =============================================================================
    # SIDEBAR NAVIGATION
    # =============================================================================
    st.sidebar.title("🏢 Energy Dashboard Navigation")
    pages = [
        "🏠 Overview", 
        "📊 Room Analysis", 
        "🔄 Building Comparison", 
        "📈 Energy Predictions", 
        "💡 Efficiency Recommendations", 
        "ℹ️ About SDGs"
    ]
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # =============================================================================
    # PAGE 1: OVERVIEW
    # =============================================================================
    if selected_page == "🏠 Overview":
        st.markdown("<h1 class='main-header'>⚡ Energy Efficiency Monitoring Dashboard</h1>", unsafe_allow_html=True)
        
        # Info box explaining dashboard purpose
        st.markdown("""
        <div class='energy-alert'>
        <strong>SDG 9 & 11 Alignment:</strong> This dashboard monitors energy consumption across buildings, rooms, and floors 
        to optimize infrastructure efficiency (SDG 9) and create sustainable communities (SDG 11). 
        Track real-time usage, predict future needs, and implement energy-saving strategies.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate overall metrics
        total_energy = df['energy_kwh'].sum()
        total_cost = df['cost_usd'].sum()
        total_carbon = df['carbon_kg'].sum()
        avg_efficiency = df['efficiency_percent'].mean()
        
        # Display key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Total Energy</h4>
                <h2 style='color: #e74c3c;'>{total_energy:,.1f} kWh</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Use currency conversion for cost display
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Total Cost</h4>
                <h2 style='color: #f39c12;'>{fmt_money(convert_currency(total_cost))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Carbon Footprint</h4>
                <h2 style='color: #95a5a6;'>{total_carbon:,.1f} kg CO₂</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Avg Efficiency</h4>
                <h2 style='color: #27ae60;'>{avg_efficiency:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            # Calculate potential savings (assume 20% possible)
            potential_savings = total_cost * 0.2
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Potential Savings</h4>
                <h2 style='color: #2c5aa0;'>{fmt_money(convert_currency(potential_savings))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Energy consumption by building
            building_energy = df.groupby('building')['energy_kwh'].sum().reset_index()
            fig1 = px.bar(
                building_energy,
                x='building',
                y='energy_kwh',
                title="Energy Consumption by Building",
                color='energy_kwh',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Cost by building (using converted currency)
            building_cost = df.groupby('building')['cost_local'].sum().reset_index()
            fig2 = px.bar(
                building_cost,
                x='building',
                y='cost_local',
                title=f"Cost by Building ({currency})",
                color='cost_local',
                color_continuous_scale='greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Daily energy trend
        daily_trend = df.groupby('date')['energy_kwh'].sum().reset_index()
        fig3 = px.line(
            daily_trend,
            x='date',
            y='energy_kwh',
            title="Daily Energy Consumption Trend",
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Appliance efficiency overview
        st.markdown("### 🔌 Appliance Efficiency Overview")
        fig4 = px.bar(
            appliance_df,
            x='appliance',
            y='avg_daily_kwh',
            color='efficiency_rating',
            title="Average Daily Consumption by Appliance",
            color_continuous_scale='RdYlGn'
        )
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)
    
    # =============================================================================
    # PAGE 2: ROOM ANALYSIS
    # =============================================================================
    elif selected_page == "📊 Room Analysis":
        st.markdown("<h1 class='main-header'>📊 Room-wise Energy Analysis</h1>", unsafe_allow_html=True)
        
        # Filters for selecting specific room
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_building = st.selectbox("Select Building", df['building'].unique())
        
        with col2:
            available_floors = df[df['building'] == selected_building]['floor'].unique()
            selected_floor = st.selectbox("Select Floor", available_floors)
        
        with col3:
            available_rooms = df[
                (df['building'] == selected_building) & 
                (df['floor'] == selected_floor)
            ]['room'].unique()
            selected_room = st.selectbox("Select Room", available_rooms)
        
        # Filter data for selected room
        room_data = df[
            (df['building'] == selected_building) & 
            (df['floor'] == selected_floor) & 
            (df['room'] == selected_room)
        ].sort_values('date')
        
        if not room_data.empty:
            latest_data = room_data.iloc[-1]  # Get most recent record
            
            # Room overview section
            st.markdown(f"### 🏠 {selected_room} Analysis - {selected_building}, {selected_floor}")
            
            # Display key metrics for this room
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Latest Energy", f"{latest_data['energy_kwh']:.1f} kWh")
            
            with col2:
                st.metric("Daily Cost", fmt_money(latest_data['cost_local']))
            
            with col3:
                st.metric("Carbon Impact", f"{latest_data['carbon_kg']:.1f} kg CO₂")
            
            with col4:
                st.metric("Efficiency", f"{latest_data['efficiency_percent']:.1f}%")
            
            # Trends and analysis
            st.markdown("### 📈 Trends")
            col1, col2 = st.columns(2)
            
            with col1:
                # Energy consumption trend
                fig1 = px.line(
                    room_data,
                    x='date',
                    y='energy_kwh',
                    title=f"{selected_room} Energy Consumption Trend",
                    markers=True
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Efficiency trend
                fig2 = px.line(
                    room_data,
                    x='date',
                    y='efficiency_percent',
                    title=f"{selected_room} Efficiency Trend",
                    markers=True,
                    color_discrete_sequence=['green']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Cost trend chart
            fig3 = px.bar(
                room_data,
                x='date',
                y='cost_local',
                title=f"Daily Cost Trend - {selected_room} ({currency})",
                color='cost_local',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Energy analysis
            st.markdown("### 📊 Detailed Analysis")
            
            avg_daily = room_data['energy_kwh'].mean()
            max_daily = room_data['energy_kwh'].max()
            min_daily = room_data['energy_kwh'].min()
            
            # Alert if energy usage is high
            if latest_data['energy_kwh'] > avg_daily * 1.2:
                st.markdown("""
                <div class='energy-alert'>
                ⚠️ <strong>High Energy Usage Alert:</strong> This room is consuming above average energy. 
                Consider checking appliance settings and usage patterns.
                </div>
                """, unsafe_allow_html=True)
            
            # Room-specific recommendations
            st.markdown("### 💡 Room Optimization Recommendations")
            
            if latest_data['efficiency_percent'] < 80:
                st.markdown("""
                <div class='efficiency-tip'>
                🔧 <strong>Low Efficiency Detected:</strong>
                <ul>
                    <li>Check for air leaks around windows and doors</li>
                    <li>Upgrade to LED lighting if not already installed</li>
                    <li>Set air conditioning to optimal temperature (24-26°C)</li>
                    <li>Use power strips to eliminate standby power consumption</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Data table
            st.markdown("### 📋 Historical Data")
            st.dataframe(
                room_data[['date', 'energy_kwh', 'cost_local', 'carbon_kg', 'efficiency_percent']].rename(
                    columns={'cost_local': f'Cost ({currency})'}
                ),
                use_container_width=True, 
                hide_index=True
            )
    
    # =============================================================================
    # PAGE 3: BUILDING COMPARISON
    # =============================================================================
    elif selected_page == "🔄 Building Comparison":
        st.markdown("<h1 class='main-header'>🔄 Building Performance Comparison</h1>", unsafe_allow_html=True)
        
        # Building comparison metrics
        building_summary = df.groupby('building').agg({
            'energy_kwh': ['sum', 'mean'],
            'cost_usd': 'sum',
            'carbon_kg': 'sum',
            'efficiency_percent': 'mean'
        }).round(2)
        
        building_summary.columns = ['Total Energy', 'Avg Daily Energy', 'Total Cost', 'Total Carbon', 'Avg Efficiency']
        building_summary = building_summary.reset_index()
        
        # Add currency-converted cost column
        building_summary['Cost Local'] = building_summary['Total Cost'].apply(convert_currency)
        
        # Performance ranking
        st.markdown("### 🏆 Building Performance Ranking")
        
        # Calculate overall performance score
        building_summary['Performance Score'] = (
            building_summary['Avg Efficiency'] * 0.6 - 
            (building_summary['Avg Daily Energy'] / building_summary['Avg Daily Energy'].max()) * 40
        ).round(1)
        
        building_summary = building_summary.sort_values('Performance Score', ascending=False)
        building_summary['Rank'] = range(1, len(building_summary) + 1)
        
        # Display summary table with converted currency
        display_summary = building_summary[['Rank', 'building', 'Total Energy', 'Cost Local', 'Total Carbon', 'Avg Efficiency', 'Performance Score']]
        display_summary = display_summary.rename(columns={'Cost Local': f'Total Cost ({currency})'})
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                building_summary,
                x='building',
                y='Total Energy',
                title="Total Energy Consumption Comparison",
                color='Total Energy',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                building_summary,
                x='building',
                y='Avg Efficiency',
                title="Average Efficiency Comparison",
                color='Avg Efficiency',
                color_continuous_scale='greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Floor-wise comparison
        st.markdown("### 🏢 Floor-wise Energy Distribution")
        
        floor_data = df.groupby(['building', 'floor'])['energy_kwh'].sum().reset_index()
        
        fig3 = px.sunburst(
            floor_data,
            path=['building', 'floor'],
            values='energy_kwh',
            title="Energy Consumption Hierarchy: Building → Floor"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # =============================================================================
    # PAGE 4: ENERGY PREDICTIONS
    # =============================================================================
    elif selected_page == "📈 Energy Predictions":
        st.markdown("<h1 class='main-header'>📈 Energy Consumption Predictions</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='efficiency-tip'>
        <strong>Predictive Analytics:</strong> Using machine learning to forecast future energy requirements 
        based on historical consumption patterns, building characteristics, and efficiency ratings.
        </div>
        """, unsafe_allow_html=True)
        
        # Train prediction model
        model, feature_names = train_prediction_model(df)
        
        # Prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔮 Future Consumption Prediction")
            
            pred_building = st.selectbox("Building", df['building'].unique(), key="pred_building")
            pred_floor = st.selectbox("Floor", df['floor'].unique(), key="pred_floor")
            pred_efficiency = st.slider("Expected Efficiency (%)", 70.0, 100.0, 85.0, key="pred_eff")
            pred_days = st.slider("Prediction Period (days)", 1, 30, 7, key="pred_days")
            
            if st.button("Generate Prediction"):
                # Create prediction features
                predictions = []
                base_date = df['date'].max()
                
                for i in range(pred_days):
                    future_date = base_date + timedelta(days=i+1)
                    
                    # Create feature vector
                    features = pd.DataFrame({
                        'day_of_week': [future_date.dayofweek],
                        'month': [future_date.month],
                        'efficiency_percent': [pred_efficiency]
                    })
                    
                    # Add building and floor dummies
                    for building in df['building'].unique():
                        features[f'building_{building}'] = [1 if building == pred_building else 0]
                    
                    for floor in df['floor'].unique():
                        features[f'floor_{floor}'] = [1 if floor == pred_floor else 0]
                    
                    # Ensure all feature columns are present
                    for col in feature_names:
                        if col not in features.columns:
                            features[col] = [0]
                    
                    # Reorder columns to match training
                    features = features[feature_names]
                    
                    # Make prediction
                    pred_energy = model.predict(features)[0]
                    pred_cost = pred_energy * 0.12  # $0.12 per kWh
                    pred_carbon = pred_energy * 0.4  # 0.4 kg CO2 per kWh
                    
                    predictions.append({
                        'date': future_date,
                        'predicted_energy_kwh': pred_energy,
                        'predicted_cost_usd': pred_cost,
                        'predicted_carbon_kg': pred_carbon
                    })
                
                pred_df = pd.DataFrame(predictions)
                
                # Display predictions
                fig = px.line(
                    pred_df,
                    x='date',
                    y='predicted_energy_kwh',
                    title=f"Energy Consumption Forecast - {pred_building}, {pred_floor}",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics with currency conversion
                total_pred_energy = pred_df['predicted_energy_kwh'].sum()
                total_pred_cost = pred_df['predicted_cost_usd'].sum()
                total_pred_carbon = pred_df['predicted_carbon_kg'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Energy", f"{total_pred_energy:.1f} kWh")
                with col2:
                    st.metric("Predicted Cost", fmt_money(convert_currency(total_pred_cost)))
                with col3:
                    st.metric("Predicted Carbon", f"{total_pred_carbon:.1f} kg CO₂")
        
        with col2:
            st.markdown("### 📊 Historical vs Predicted Trends")
            
            # Show recent trend
            recent_data = df.groupby('date')['energy_kwh'].sum().reset_index().tail(10)
            
            fig = px.line(
                recent_data,
                x='date',
                y='energy_kwh',
                title="Recent Energy Consumption Trend",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Energy saving scenarios with currency
            st.markdown("### 💰 Potential Savings Scenarios")
            
            scenarios = pd.DataFrame({
                'Scenario': ['Current', '5% Improvement', '10% Improvement', '15% Improvement'],
                'Daily Energy (kWh)': [150, 142.5, 135, 127.5],
                'Monthly Cost': [540, 513, 486, 459],
                'Annual Savings': [0, 324, 648, 972]
            })
            
            # Convert cost columns to selected currency
            scenarios['Monthly Cost'] = scenarios['Monthly Cost'].apply(convert_currency)
            scenarios['Annual Savings'] = scenarios['Annual Savings'].apply(convert_currency)
            scenarios = scenarios.rename(columns={
                'Monthly Cost': f'Monthly Cost ({currency})',
                'Annual Savings': f'Annual Savings ({currency})'
            })
            
            st.dataframe(scenarios, use_container_width=True, hide_index=True)
    
    # =============================================================================
    # PAGE 5: EFFICIENCY RECOMMENDATIONS
    # =============================================================================
    elif selected_page == "💡 Efficiency Recommendations":
        st.markdown("<h1 class='main-header'>💡 Energy Efficiency Recommendations</h1>", unsafe_allow_html=True)
        
        # Appliance-specific recommendations
        st.markdown("### 🔌 Appliance Optimization Guide")
        
        for _, appliance in appliance_df.iterrows():
            efficiency_status = "🟢 Excellent" if appliance['efficiency_rating'] > 90 else "🟡 Good" if appliance['efficiency_rating'] > 80 else "🔴 Needs Improvement"
            
            with st.expander(f"{appliance['appliance']} - {efficiency_status} ({appliance['efficiency_rating']}% efficient)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Daily Consumption", f"{appliance['avg_daily_kwh']:.1f} kWh")
                    st.metric("Efficiency Rating", f"{appliance['efficiency_rating']}%")
                
                with col2:
                    st.metric("Potential Savings", f"{appliance['potential_savings_percent']}%")
                    potential_kwh_savings = appliance['avg_daily_kwh'] * appliance['potential_savings_percent'] / 100
                    potential_cost_savings = potential_kwh_savings * 0.12 * 30  # Monthly savings
                    st.metric("Monthly Savings Potential", fmt_money(convert_currency(potential_cost_savings)))
                
                # Specific recommendations based on appliance type
                if appliance['appliance'] == 'Air Conditioner':
                    st.markdown("""
                    **Optimization Tips:**
                    - Set temperature to 24-26°C for optimal efficiency
                    - Use programmable timers to avoid cooling empty rooms
                    - Regular filter cleaning improves efficiency by 5-15%
                    - Consider upgrading to inverter AC for 30-50% energy savings
                    """)
                elif appliance['appliance'] == 'Water Heater':
                    st.markdown("""
                    **Optimization Tips:**
                    - Lower water heater temperature to 120°F (49°C)
                    - Insulate water heater and hot water pipes
                    - Use timer switches for electric water heaters
                    - Consider solar water heating systems
                    """)
                elif appliance['appliance'] == 'Refrigerator':
                    st.markdown("""
                    **Optimization Tips:**
                    - Keep refrigerator temperature at 37-40°F (3-4°C)
                    - Ensure proper door seals to prevent cold air leakage
                    - Keep refrigerator full but not overcrowded
                    - Clean coils regularly for better heat dissipation
                    """)
                elif appliance['appliance'] == 'LED Lights':
                    st.markdown("""
                    **Optimization Tips:**
                    - Install motion sensors in less frequently used areas
                    - Use daylight sensors to adjust lighting automatically
                    - Replace any remaining incandescent bulbs with LEDs
                    - Consider task lighting instead of overhead lighting
                    """)
        
        # Building-wide recommendations
        st.markdown("### 🏢 Building-wide Efficiency Strategies")
        
        recommendations = {
            "Immediate Actions (0-1 month)": [
                "🔌 Implement smart power strips to eliminate phantom loads",
                "🌡️ Optimize HVAC schedules based on occupancy patterns",
                "💡 Install occupancy sensors for lighting control",
                "📊 Set up real-time energy monitoring alerts"
            ],
            "Short-term Improvements (1-6 months)": [
                "🪟 Seal air leaks around windows, doors, and ducts",
                "🔄 Upgrade to high-efficiency appliances when replacements are needed",
                "⚡ Install LED lighting throughout the building",
                "🏢 Implement building automation system for optimal control"
            ],
            "Long-term Investments (6+ months)": [
                "☀️ Consider solar panel installation for renewable energy",
                "🔋 Implement energy storage systems for peak shaving",
                "🏗️ Upgrade building insulation and windows",
                "🌱 Install green roof or living walls for natural cooling"
            ]
        }
        
        for timeframe, actions in recommendations.items():
            with st.expander(timeframe):
                for action in actions:
                    st.markdown(f"- {action}")
        
        # Cost-benefit analysis with currency conversion
        st.markdown("### 💰 Investment Payback Analysis")
        
        investments = pd.DataFrame({
            'Improvement': ['LED Upgrade', 'Smart Thermostats', 'Insulation', 'Solar Panels', 'Energy Management System'],
            'Initial Cost': [2000, 1500, 5000, 25000, 8000],
            'Annual Savings': [600, 400, 800, 3000, 1200],
            'Payback Period (years)': [3.3, 3.8, 6.3, 8.3, 6.7],
            'CO₂ Reduction (kg/year)': [1500, 1000, 2000, 7500, 3000]
        })
        
        # Convert currency columns
        investments['Initial Cost'] = investments['Initial Cost'].apply(convert_currency)
        investments['Annual Savings'] = investments['Annual Savings'].apply(convert_currency)
        investments = investments.rename(columns={
            'Initial Cost': f'Initial Cost ({currency})',
            'Annual Savings': f'Annual Savings ({currency})'
        })
        
        fig = px.scatter(
            investments,
            x=f'Initial Cost ({currency})',
            y=f'Annual Savings ({currency})',
            size='CO₂ Reduction (kg/year)',
            color='Payback Period (years)',
            title="Energy Efficiency Investment Analysis",
            hover_data=['Improvement']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(investments, use_container_width=True, hide_index=True)
    
    # =============================================================================
    # PAGE 6: ABOUT SDGs
    # =============================================================================
    elif selected_page == "ℹ️ About SDGs":
        st.markdown("<h1 class='main-header'>ℹ️ SDG Alignment & Impact</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Sustainable Development Goals Alignment
        
        This Energy Efficiency Monitoring System directly contributes to multiple UN Sustainable Development Goals:
        """)
        
        # SDG 9: Industry, Innovation and Infrastructure
        st.markdown("""
        <div class='efficiency-tip'>
        <h4>🏗️ SDG 9: Industry, Innovation and Infrastructure</h4>
        
        <strong>How this project contributes:</strong>
        <ul>
            <li><strong>9.4:</strong> Upgrade infrastructure and retrofit industries for sustainability and resource efficiency</li>
            <li><strong>9.b:</strong> Support domestic technology development and innovation in developing countries</li>
            <li><strong>9.c:</strong> Increase access to ICT and provide universal internet access</li>
        </ul>
        
        <strong>Specific Impact:</strong>
        - Real-time monitoring enables data-driven infrastructure optimization
        - Predictive analytics help plan future infrastructure needs
        - IoT-based system promotes technological innovation in building management
        - Reduces resource consumption through efficient energy management
        </div>
        """, unsafe_allow_html=True)
        
        # SDG 11: Sustainable Cities and Communities  
        st.markdown("""
        <div class='savings-highlight'>
        <h4>🏙️ SDG 11: Sustainable Cities and Communities</h4>
        
        <strong>How this project contributes:</strong>
        <ul>
            <li><strong>11.6:</strong> Reduce per capita environmental impact of cities, including air quality and waste management</li>
            <li><strong>11.b:</strong> Implement policies for inclusion, resource efficiency, and climate change adaptation</li>
            <li><strong>11.c:</strong> Support least developed countries in sustainable and resilient building construction</li>
        </ul>
        
        <strong>Specific Impact:</strong>
        - Reduces carbon emissions through optimized energy consumption
        - Promotes sustainable building practices and energy-efficient communities  
        - Provides tools for urban planners to make informed decisions
        - Creates awareness about energy consumption patterns in communities
        </div>
        """, unsafe_allow_html=True)
        
        # Additional SDG connections
        st.markdown("""
        ### 🔗 Secondary SDG Connections
        
        **SDG 7: Affordable and Clean Energy**
        - Promotes energy efficiency and conservation
        - Supports transition to sustainable energy systems
        - Reduces overall energy demand through optimization
        
        **SDG 12: Responsible Consumption and Production**  
        - Enables responsible energy consumption patterns
        - Promotes resource efficiency in buildings
        - Supports sustainable procurement of energy-efficient appliances
        
        **SDG 13: Climate Action**
        - Reduces greenhouse gas emissions through energy optimization
        - Provides data for climate impact assessment
        - Supports building resilience against climate change impacts
        """)
        
        # Impact metrics
        st.markdown("### 📊 Project Impact Metrics")
        
        impact_data = pd.DataFrame({
            'Impact Area': ['Energy Savings', 'Cost Reduction', 'CO₂ Reduction', 'Efficiency Improvement'],
            'Current Achievement': ['15%', '12%', '18%', '8%'],
            'Target (1 Year)': ['25%', '20%', '30%', '15%'],
            'SDG Contribution': ['SDG 7, 11', 'SDG 9, 11', 'SDG 11, 13', 'SDG 9, 12']
        })
        
        st.dataframe(impact_data, use_container_width=True, hide_index=True)
        
        # Technical implementation
        st.markdown("""
        ### 🛠️ Technical Implementation Framework
        
        **Data Collection Layer:**
        - IoT sensors for real-time energy monitoring
        - Smart meters for appliance-level consumption tracking
        - Environmental sensors for context-aware analysis
        
        **Analytics Layer:**
        - Machine learning models for consumption prediction
        - Statistical analysis for pattern recognition
        - Anomaly detection for identifying inefficiencies
        
        **Visualization Layer:**
        - Interactive dashboards for stakeholder engagement
        - Mobile applications for real-time monitoring
        - Reporting tools for regulatory compliance
        
        **Action Layer:**
        - Automated control systems for energy optimization
        - Alert systems for immediate response to anomalies
        - Recommendation engines for continuous improvement
        """)
        
        # Call to action
        st.markdown("""
        <div class='energy-alert'>
        <h4>🚀 Next Steps for Implementation</h4>
        
        <strong>Phase 1:</strong> Pilot deployment in select buildings
        <br><strong>Phase 2:</strong> Scale to entire campus/complex
        <br><strong>Phase 3:</strong> Integration with smart city infrastructure
        <br><strong>Phase 4:</strong> Community-wide adoption and knowledge sharing
        
        <br><br>
        <strong>Join the sustainable energy revolution! 🌱⚡</strong>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
**Energy Efficiency Monitoring Dashboard** | 
**SDG 9:** Industry, Innovation and Infrastructure | 
**SDG 11:** Sustainable Cities and Communities | 
Built with ❤️ for a sustainable future | 
**Currency:** {currency}
""")