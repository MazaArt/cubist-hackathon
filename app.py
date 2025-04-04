import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="MTA Congestion Relief Zone Traffic Visualization",
    page_icon="🚗",
    layout="wide"
)

# Page title
st.title("MTA Congestion Relief Zone Traffic Flow")
st.markdown("Visualizing traffic flow into the Congestion Relief Zone by entry points")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
    # Convert date to datetime
    df['Toll Date'] = pd.to_datetime(df['Toll Date'])
    df['Toll Hour'] = pd.to_datetime(df['Toll Hour'])
    # Extract date and time components
    df['date'] = df['Toll Date'].dt.date
    df['hour'] = df['Hour of Day']
    return df

# Load the data with a progress indicator
with st.spinner('Loading data... This may take a moment.'):
    try:
        df = load_data()
        st.success('Data loaded successfully!')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Display basic info about the dataset
st.subheader("Dataset Information")
st.write(f"Total records: {len(df):,}")
st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Show unique entry points
entry_points = df['Detection Region'].unique()
st.write(f"Entry points: {len(entry_points)}")
st.write(f"Entry locations: {', '.join(sorted(entry_points))}")

# Date and time selection
st.sidebar.header("Filter Data")
selected_date = st.sidebar.date_input(
    "Select date",
    min_value=df['date'].min(),
    max_value=df['date'].max(),
    value=df['date'].min()
)

# Time range slider
time_range = st.sidebar.slider(
    "Select time range (hours)",
    0, 23, (0, 23)
)

# Filter data based on selections
filtered_df = df[
    (df['date'] == selected_date) & 
    (df['hour'] >= time_range[0]) & 
    (df['hour'] <= time_range[1])
]

# Create a map of entry points
st.subheader("Traffic Flow by Entry Point")

# Map coordinates for the entry points based on Detection Region
coordinates = {
    "Brooklyn": (40.6782, -73.9442),
    "West Side Highway": (40.7713, -73.9971),
    "West 60th St": (40.7697, -73.9829),
    "Queens": (40.7282, -73.7949),
    "New Jersey": (40.7589, -74.0337),
    "FDR Drive": (40.7681, -73.9682),
    "East 60th St": (40.7615, -73.9656)
}

# Get traffic by entry point
entry_traffic = filtered_df.groupby('Detection Region')['CRZ Entries'].sum().reset_index()
total_traffic = entry_traffic['CRZ Entries'].sum()
entry_traffic['percentage'] = (entry_traffic['CRZ Entries'] / total_traffic * 100).round(1)

# Create the map
fig = go.Figure()

# Central point for Manhattan Congestion Zone
center_lat, center_lon = 40.7580, -73.9855

# Add the central zone
fig.add_trace(go.Scattermapbox(
    lat=[center_lat],
    lon=[center_lon],
    mode='markers',
    marker=dict(size=30, color='red', opacity=0.5),
    text=["Manhattan Congestion Zone"],
    name="Congestion Zone"
))

# Add entry points with arrows pointing to the center
for _, row in entry_traffic.iterrows():
    location = row['Detection Region']
    
    # Skip if we don't have coordinates
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Calculate arrow path
    # Start from the entry point and go 80% toward the center
    mid_lat = lat + 0.8 * (center_lat - lat)
    mid_lon = lon + 0.8 * (center_lon - lon)
    
    # Line width scaled by percentage of traffic
    line_width = 1 + (row['percentage'] / 5)
    
    # Add the arrow line
    fig.add_trace(go.Scattermapbox(
        lat=[lat, mid_lat],
        lon=[lon, mid_lon],
        mode='lines',
        line=dict(width=line_width, color='blue'),
        name=location,
        text=f"{location}: {row['percentage']}% ({row['CRZ Entries']:,} entries)"
    ))
    
    # Add the entry point marker
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=[f"{row['percentage']}%"],
        textposition="top center",
        name=location
    ))

# Update the layout
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=10.5,
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# Display traffic summary
st.subheader("Traffic Summary")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric("Total Traffic", f"{total_traffic:,} vehicles")
    
    # Display traffic by entry point
    st.dataframe(
        entry_traffic.sort_values('CRZ Entries', ascending=False),
        column_config={
            'Detection Region': 'Entry Point',
            'CRZ Entries': 'Traffic Volume',
            'percentage': 'Percentage (%)'
        },
        hide_index=True
    )

with summary_col2:
    # Traffic volume by hour
    hourly_traffic = filtered_df.groupby('hour')['CRZ Entries'].sum().reset_index()
    fig_hourly = px.line(
        hourly_traffic, 
        x='hour', 
        y='CRZ Entries',
        title='Traffic Volume by Hour',
        labels={'hour': 'Hour of Day', 'CRZ Entries': 'Number of Vehicles'}
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# Add vehicle class breakdown
st.subheader("Traffic by Vehicle Class")
vehicle_class_traffic = filtered_df.groupby('Vehicle Class')['CRZ Entries'].sum().reset_index()
vehicle_class_traffic['percentage'] = (vehicle_class_traffic['CRZ Entries'] / vehicle_class_traffic['CRZ Entries'].sum() * 100).round(1)
vehicle_class_traffic = vehicle_class_traffic.sort_values('CRZ Entries', ascending=False)

fig_vehicle = px.pie(
    vehicle_class_traffic,
    values='CRZ Entries',
    names='Vehicle Class',
    title='Traffic Distribution by Vehicle Class'
)
st.plotly_chart(fig_vehicle, use_container_width=True)

# Add time animation
st.subheader("Traffic Flow Over Time")
st.write("Use the slider to see how traffic changes throughout the day")

# Create time slider
selected_hour = st.slider("Hour of day", 0, 23, 8)

# Filter data for the selected hour
hourly_data = df[
    (df['date'] == selected_date) & 
    (df['hour'] == selected_hour)
]

# Group by entry point
hourly_entry_traffic = hourly_data.groupby('Detection Region')['CRZ Entries'].sum().reset_index()
hourly_total = hourly_entry_traffic['CRZ Entries'].sum()
hourly_entry_traffic['percentage'] = (hourly_entry_traffic['CRZ Entries'] / hourly_total * 100).round(1)

# Create hourly map
hourly_fig = go.Figure()

# Add the central zone
hourly_fig.add_trace(go.Scattermapbox(
    lat=[center_lat],
    lon=[center_lon],
    mode='markers',
    marker=dict(size=30, color='red', opacity=0.5),
    text=[f"Manhattan Congestion Zone<br>Total: {hourly_total:,} vehicles at {selected_hour}:00"],
    name="Congestion Zone"
))

# Add entry points with arrows pointing to the center
for _, row in hourly_entry_traffic.iterrows():
    location = row['Detection Region']
    
    # Skip if we don't have coordinates
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Calculate arrow path
    mid_lat = lat + 0.8 * (center_lat - lat)
    mid_lon = lon + 0.8 * (center_lon - lon)
    
    # Line width scaled by percentage of traffic
    line_width = 1 + (row['percentage'] / 5)
    
    # Add the arrow line
    hourly_fig.add_trace(go.Scattermapbox(
        lat=[lat, mid_lat],
        lon=[lon, mid_lon],
        mode='lines',
        line=dict(width=line_width, color='blue'),
        name=location,
        text=f"{location}: {row['percentage']}% ({row['CRZ Entries']:,} entries)"
    ))
    
    # Add the entry point marker
    hourly_fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=[f"{row['percentage']}%"],
        textposition="top center",
        name=location
    ))

# Update the layout
hourly_fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=10.5,
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    title=f"Traffic Flow at {selected_hour}:00"
)

st.plotly_chart(hourly_fig, use_container_width=True) 