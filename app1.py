import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

# Set page config
st.set_page_config(
    page_title="MTA Congestion Relief Zone Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        font-size: 1rem !important;
        color: #555 !important;
    }
    .stMetric div {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    .stPlotlyChart {
        border-radius: 10px;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
        background-image: none;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .st-b7 {
        color: #333;
    }
    .st-c0 {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Page title and description
st.title("MTA Congestion Relief Zone Explorer")
st.markdown("""
Explore traffic patterns entering Manhattan's Congestion Relief Zone. Discover trends, compare entry points, 
and analyze traffic flow over time through interactive visualizations.
""")

# Load the data with caching and error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
        # Convert date to datetime
        df['Toll Date'] = pd.to_datetime(df['Toll Date'])
        df['Toll Hour'] = pd.to_datetime(df['Toll Hour'])
        # Extract date and time components
        df['date'] = df['Toll Date'].dt.date
        df['hour'] = df['Hour of Day']
        df['day_of_week'] = df['Toll Date'].dt.day_name()
        df['month'] = df['Toll Date'].dt.month_name()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load the data with a progress indicator
with st.spinner('Loading data... Please wait while we prepare the dashboard.'):
    df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data. Please check the data file and try again.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Exploration Filters")

# Date range selection
date_range = st.sidebar.date_input(
    "Select date range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Ensure we have a valid date range
if len(date_range) != 2:
    date_range = (df['date'].min(), df['date'].max())

# Time range slider
time_range = st.sidebar.slider(
    "Select hours to analyze",
    0, 23, (6, 20),
    help="Select the time window to analyze each day"
)

# Vehicle class filter
vehicle_classes = sorted(df['Vehicle Class'].unique())
selected_vehicles = st.sidebar.multiselect(
    "Filter vehicle classes",
    options=vehicle_classes,
    default=vehicle_classes,
    help="Select which vehicle classes to include in the analysis"
)

# Entry point selection
entry_points = sorted(df['Detection Group'].unique())
selected_points = st.sidebar.multiselect(
    "Filter entry points",
    options=entry_points,
    default=entry_points,
    help="Select which entry points to analyze"
)

# Filter data based on selections
filtered_df = df[
    (df['date'] >= date_range[0]) & 
    (df['date'] <= date_range[1]) & 
    (df['hour'] >= time_range[0]) & 
    (df['hour'] <= time_range[1]) &
    (df['Vehicle Class'].isin(selected_vehicles)) &
    (df['Detection Group'].isin(selected_points))
]

# Show data summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Summary**")
st.sidebar.markdown(f"📅 Date Range: {date_range[0]} to {date_range[1]}")
st.sidebar.markdown(f"⏰ Time Window: {time_range[0]}:00 - {time_range[1]}:00")
st.sidebar.markdown(f"🚗 Vehicle Classes: {len(selected_vehicles)} selected")
st.sidebar.markdown(f"📍 Entry Points: {len(selected_points)} selected")

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Spatial Analysis", "📊 Temporal Patterns", "🔍 Comparative Analysis", "📈 Trend Explorer"])

with tab1:
    st.header("Spatial Traffic Patterns")
    st.markdown("Visualize how traffic volume at each entry point evolves across days and hours.")
    
    # Map coordinates for the entry points
    coordinates = {
        "Brooklyn Bridge": (40.7061, -73.9969),
        "West Side Highway at 60th St": (40.7714, -73.9900),
        "West 60th St": (40.7700, -73.9850),
        "Queensboro Bridge": (40.7553, -73.9500),
        "Queens Midtown Tunnel": (40.7407, -73.9711),
        "Lincoln Tunnel": (40.7570, -74.0027),
        "Holland Tunnel": (40.7270, -74.0119),
        "FDR Drive at 60th St": (40.7600, -73.9580),
        "East 60th St": (40.7610, -73.9630),
        "Williamsburg Bridge": (40.7117, -73.970),
        "Manhattan Bridge": (40.7075, -73.9903),
        "Hugh L. Carey Tunnel": (40.7017, -74.0132)
    }
    center_lat, center_lon = 40.7380, -73.9855
    
    # Calculate traffic by entry point
    entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
    entry_traffic['percentage'] = (entry_traffic['CRZ Entries'] / entry_traffic['CRZ Entries'].sum() * 100).round(1)
    
    # Filter to selected date range, selected entry points, AND selected hours
    animated_df = df[
        (df['date'] >= date_range[0]) &
        (df['date'] <= date_range[1]) &
        (df['Detection Group'].isin(selected_points)) &
        (df['hour'] >= time_range[0]) &
        (df['hour'] <= time_range[1])
    ].copy()
    
    # Add coordinates
    animated_df['lat'] = animated_df['Detection Group'].map(lambda x: coordinates.get(x, (None, None))[0])
    animated_df['lon'] = animated_df['Detection Group'].map(lambda x: coordinates.get(x, (None, None))[1])
    animated_df.dropna(subset=['lat', 'lon'], inplace=True)
    
    # Check if there's any data after filtering
    if animated_df.empty:
        st.warning("No data available for the selected filters. Please adjust your filters.")
        st.stop()
    
    # Combine date and hour into one animation timestamp
    animated_df['timestamp'] = animated_df['Toll Hour'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Group by timestamp and entry point
    grouped = (
        animated_df
        .groupby(['timestamp', 'Detection Group', 'lat', 'lon'], as_index=False)['CRZ Entries']
        .sum()
    )
    
    # Use median traffic for discrete color mapping
    median_traffic = grouped['CRZ Entries'].median()
    grouped['color'] = grouped['CRZ Entries'].apply(lambda x: 'red' if x > median_traffic else 'green')
    
    # Build animation
    frames = []
    timestamps = grouped['timestamp'].unique()
    
    # Initialize figure
    initial_data = grouped[grouped['timestamp'] == timestamps[0]]
    fig_anim = go.Figure(
        data=[
            go.Scattermapbox(
                lat=initial_data['lat'],
                lon=initial_data['lon'],
                mode='markers',
                marker=dict(
                    size=initial_data['CRZ Entries'] / grouped['CRZ Entries'].max() * 25 + 8,
                    color=initial_data['color'],
                    opacity=0.8
                ),
                text=initial_data.apply(
                    lambda row: f"{row['Detection Group']}<br>Traffic: {row['CRZ Entries']:,}", axis=1
                ),
                hoverinfo='text'
            )
        ],
        layout=dict(
            mapbox=dict(
                style="open-street-map",
                zoom=11,
                center=dict(lat=center_lat, lon=center_lon)
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
            title=f"Traffic Flow Animation ({time_range[0]}:00-{time_range[1]}:00)",
            showlegend=False,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "▶️",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}],
                        "label": "⏸",
                        "method": "animate"
                    }
                ],
                "type": "buttons",
                "direction": "left",
                "pad": {"r": 10, "t": 30},
                "x": 0,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 10},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[t], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                        "label": t,
                        "method": "animate"
                    } for t in timestamps
                ]
            }]
        )
    )
    
    # Add frames
    for t in timestamps:
        frame_data = grouped[grouped['timestamp'] == t]
        frames.append(go.Frame(
            data=[
                go.Scattermapbox(
                    lat=frame_data['lat'],
                    lon=frame_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=frame_data['CRZ Entries'] / grouped['CRZ Entries'].max() * 25 + 8,
                        color=frame_data['color'],
                        opacity=0.8
                    ),
                    text=frame_data.apply(
                        lambda row: f"{row['Detection Group']}<br>Traffic: {row['CRZ Entries']:,}", axis=1
                    ),
                    hoverinfo='text'
                )
            ],
            name=t
        ))
    
    fig_anim.frames = frames
    st.plotly_chart(fig_anim, use_container_width=True)
    
    # Entry point comparison charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Entry Points")
        top_n = st.slider("Show top N entry points", 3, 12, 5)
        top_entries = entry_traffic.nlargest(top_n, 'CRZ Entries')
        
        fig_top = px.bar(
            top_entries,
            x='CRZ Entries',
            y='Detection Group',
            orientation='h',
            color='CRZ Entries',
            color_continuous_scale='Blues',
            labels={'CRZ Entries': 'Traffic Volume', 'Detection Group': 'Entry Point'},
            height=400
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.subheader("Entry Point Distribution")
        fig_pie = px.pie(
            entry_traffic,
            values='CRZ Entries',
            names='Detection Group',
            hole=0.4,
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        
with tab2:
    st.header("Temporal Traffic Patterns")
    st.markdown("Analyze how traffic flows change throughout the day, week, and month.")
    
    # Hourly patterns
    st.subheader("Hourly Traffic Patterns")
    hourly = filtered_df.groupby(['hour', 'Detection Group'])['CRZ Entries'].sum().reset_index()
    
    fig_hourly = px.line(
        hourly,
        x='hour',
        y='CRZ Entries',
        color='Detection Group',
        line_shape='spline',
        labels={'hour': 'Hour of Day', 'CRZ Entries': 'Traffic Volume'},
        height=500
    )
    fig_hourly.update_layout(hovermode='x unified')
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Day of week patterns
    st.subheader("Day of Week Patterns")
    dow = filtered_df.groupby(['day_of_week', 'Detection Group'])['CRZ Entries'].sum().reset_index()
    dow['day_of_week'] = pd.Categorical(dow['day_of_week'], 
                                      categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                                      ordered=True)
    
    fig_dow = px.bar(
        dow,
        x='day_of_week',
        y='CRZ Entries',
        color='Detection Group',
        barmode='group',
        labels={'day_of_week': 'Day of Week', 'CRZ Entries': 'Traffic Volume'},
        height=500
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Heatmap of hour vs day of week
    st.subheader("Hourly vs Weekly Heatmap")
    heatmap_data = filtered_df.groupby(['hour', 'day_of_week'])['CRZ Entries'].sum().unstack()
    heatmap_data = heatmap_data.reindex(columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Day of Week", y="Hour", color="Traffic Volume"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.header("Comparative Analysis")
    st.markdown("Compare traffic patterns across different dimensions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Class Comparison")
        vehicle_data = filtered_df.groupby('Vehicle Class')['CRZ Entries'].sum().reset_index()
        
        fig_vehicle = px.bar(
            vehicle_data,
            x='Vehicle Class',
            y='CRZ Entries',
            color='Vehicle Class',
            labels={'CRZ Entries': 'Traffic Volume'},
            height=400
        )
        st.plotly_chart(fig_vehicle, use_container_width=True)
        
    with col2:
        st.subheader("Entry Point vs Vehicle Class")
        cross_data = filtered_df.groupby(['Detection Group', 'Vehicle Class'])['CRZ Entries'].sum().reset_index()
        
        fig_cross = px.sunburst(
            cross_data,
            path=['Detection Group', 'Vehicle Class'],
            values='CRZ Entries',
            height=500
        )
        st.plotly_chart(fig_cross, use_container_width=True)
    
    st.subheader("Entry Point Correlation Matrix")
    # Create pivot table for correlation
    corr_data = filtered_df.pivot_table(
        index='date',
        columns='Detection Group',
        values='CRZ Entries',
        aggfunc='sum'
    ).corr()
    
    fig_corr = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        x=corr_data.columns,
        y=corr_data.index,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.header("Trend Explorer")
    st.markdown("Analyze how traffic patterns have changed over time.")
    
    # Time series with different aggregations
    st.subheader("Daily Traffic Trends")
    agg_method = st.radio("Aggregation method", ['Sum', 'Average'], horizontal=True)
    
    time_data = filtered_df.groupby('date')['CRZ Entries'].sum().reset_index()
    if agg_method == 'Average':
        time_data = filtered_df.groupby('date')['CRZ Entries'].mean().reset_index()
    
    fig_time = px.line(
        time_data,
        x='date',
        y='CRZ Entries',
        labels={'date': 'Date', 'CRZ Entries': f'{agg_method} Traffic Volume'},
        height=500
    )
    
    # Add trendline
    fig_time.add_trace(go.Scatter(
        x=time_data['date'],
        y=time_data['CRZ Entries'].rolling(7, min_periods=1).mean(),
        name='7-day Moving Average',
        line=dict(color='red', dash='dash')
    ))
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Decomposition of trends - with error handling
    st.subheader("Trend Decomposition")
    st.markdown("Break down traffic patterns into overall trend, weekly seasonality, and residual components.")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Create a daily time series with proper frequency
        daily_series = filtered_df.groupby('date')['CRZ Entries'].sum()
        daily_series = daily_series.asfreq('D').fillna(method='ffill')  # Forward fill missing days
        
        # Ensure we have enough data points (at least 2 full periods)
        if len(daily_series) >= 14:  # Need at least 2 weeks for weekly seasonality
            decomposition = seasonal_decompose(daily_series, model='additive', period=7)
            
            # Plot results
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=daily_series.index,
                y=decomposition.trend,
                name='Trend',
                line=dict(color='blue')
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_series.index,
                y=decomposition.seasonal,
                name='Seasonal (Weekly)',
                line=dict(color='green')
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_series.index,
                y=decomposition.resid,
                name='Residual',
                line=dict(color='red')
            ))
            
            fig_trend.update_layout(
                height=500,
                title="Time Series Decomposition",
                xaxis_title="Date",
                yaxis_title="Traffic Volume",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Not enough data for decomposition. Need at least 2 weeks of data.")
            
    except Exception as e:
        st.error(f"Could not perform time series decomposition: {str(e)}")
        st.info("Try selecting a longer date range with complete daily data.")

# Key metrics at the bottom
st.markdown("---")
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Vehicles", f"{filtered_df['CRZ Entries'].sum():,}")
with col2:
    st.metric("Average Daily Traffic", f"{filtered_df.groupby('date')['CRZ Entries'].sum().mean():,.0f}")
with col3:
    st.metric("Peak Hour", filtered_df.groupby('hour')['CRZ Entries'].sum().idxmax())
with col4:
    st.metric("Busiest Entry Point", entry_traffic.loc[entry_traffic['CRZ Entries'].idxmax(), 'Detection Group'])

# Data download option
st.sidebar.markdown("---")
st.sidebar.markdown("### Download Data")
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name=f"congestion_data_{date_range[0]}_to_{date_range[1]}.csv",
    mime='text/csv'
)