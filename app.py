import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from helper import haversine_distance
from sidebar import create_sidebar
from nyc_map import create_map
from constants import COORDINATES, CENTER_LON, CENTER_LAT


# Set page config
st.set_page_config(
    page_title="MTA Congestion Relief Zone Traffic Visualization",
    page_icon="🚗",
    layout="wide"
)

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
    
    df_2 = pd.read_csv('data/subway_ridership_diff.csv.gz')
    df_2['datehour'] = pd.to_datetime(df_2['datehour'])
    return df, df_2

# Load subway station data
@st.cache_data
def load_subway_stations():
    try:
        subway_df = pd.read_csv("subway_stations.csv")
        return subway_df
    except Exception as e:
        st.error(f"Error loading subway data: {e}")
        return pd.DataFrame()

@st.cache_data
def map_stations_to_ports(subway_stations):
    # Create a new column to store the closest port
    subway_stations = subway_stations.copy()
    subway_stations['closest_port'] = ""
    subway_stations['distance_km'] = 0.0
    
    # For each station, find the closest port
    for idx, station in subway_stations.iterrows():
        min_distance = float('inf')
        closest_port = ""
        
        for port, (port_lat, port_lon) in COORDINATES.items():
            distance = haversine_distance(
                station['latitude'], station['longitude'],
                port_lat, port_lon
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_port = port
        
        subway_stations.at[idx, 'closest_port'] = closest_port
        subway_stations.at[idx, 'distance_km'] = min_distance
    
    return subway_stations


# Page title
st.title("MTA Congestion Relief Zone Traffic Flow")
st.markdown("Visualizing traffic flow into the Congestion Relief Zone by entry points")


# Load the data with a progress indicator
with st.spinner('Loading data... This may take a moment.'):
    try:
        df, before_after_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Display basic info about the dataset
st.subheader("Dataset Information")
st.write(f"Total records: {len(df):,}")
st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Show unique entry points
entry_points = df['Detection Region'].unique()

# Date and time selection
# Create the Sidebar
st.sidebar.header("Filter Data")
entry_traffic, show_subway_stations, filtered_df, selected_date_range = create_sidebar(df)


# Filter traffic data based on selected points
filtered_traffic = entry_traffic[entry_traffic['Detection Group'].isin(st.session_state.selected_points)]

# Calculate total traffic for selected points only
selected_total = filtered_traffic['CRZ Entries'].sum()
filtered_traffic['percentage'] = (filtered_traffic['CRZ Entries'] / selected_total * 100).round(1)

fig = create_map(entry_traffic, filtered_traffic)

time_flow_tab, before_after_tab = st.tabs(['Time Flow Map', 'Before-After Subway Map'])

with time_flow_tab:
    # @Will Write your changes here
    print("TODO")
    
with before_after_tab:
    before_after_df 
    # my changes
    print("TODO")

# Display the map
st.plotly_chart(fig, use_container_width=True)

# Display traffic summary
st.subheader("Traffic Summary")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric("Total Traffic", f"{filtered_traffic['CRZ Entries'].sum():,} vehicles")
    
    # Display traffic by detection group
    st.dataframe(
        filtered_traffic.sort_values('CRZ Entries', ascending=False),
        column_config={
            'Detection Group': 'Entry Point',
            'CRZ Entries': 'Traffic Volume',
            'percentage': 'Percentage (%)'
        },
        hide_index=True
    )

with summary_col2:
    # Traffic volume by hour
    # Filter data for selected points
    hourly_filtered_df = filtered_df[filtered_df['Detection Group'].isin(st.session_state.selected_points)]
    hourly_traffic = hourly_filtered_df.groupby('hour')['CRZ Entries'].sum().reset_index()
    fig_hourly = px.line(
        hourly_traffic, 
        x='hour', 
        y='CRZ Entries',
        title='Traffic Volume by Hour',
        labels={'hour': 'Hour of Day', 'CRZ Entries': 'Number of Vehicles'}
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# Add vehicle class breakdown
# st.subheader("Traffic by Vehicle Class")
# vehicle_class_traffic = filtered_df.groupby('Vehicle Class')['CRZ Entries'].sum().reset_index()
# vehicle_class_traffic['percentage'] = (vehicle_class_traffic['CRZ Entries'] / vehicle_class_traffic['CRZ Entries'].sum() * 100).round(1)
# vehicle_class_traffic = vehicle_class_traffic.sort_values('CRZ Entries', ascending=False)

# fig_vehicle = px.pie(
#     vehicle_class_traffic,
#     values='CRZ Entries',
#     names='Vehicle Class',
#     title='Traffic Distribution by Vehicle Class'
# )
# st.plotly_chart(fig_vehicle, use_container_width=True)

# Add time animation
st.subheader("Traffic Flow Over Time")
st.write("Use the slider to see how traffic changes throughout the day")

# Create time slider
selected_hour = st.slider("Hour of day", 0, 23, 8)

# Filter data for the selected hour
hourly_data = df[
    (df['date'] == selected_date_range[0]) & 
    (df['hour'] == selected_hour)
]

# Group by entry point
hourly_entry_traffic = hourly_data.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
hourly_total = hourly_entry_traffic['CRZ Entries'].sum()
hourly_entry_traffic['percentage'] = (hourly_entry_traffic['CRZ Entries'] / hourly_total * 100).round(1)

# Create hourly map
hourly_fig = go.Figure()

# Add entry points with arrows pointing to the center
for _, row in hourly_entry_traffic.iterrows():
    location = row['Detection Group']
    
    # Skip if we don't have coordinates
    if location not in COORDINATES:
        continue
        
    lat, lon = COORDINATES[location]
    
    # Calculate arrow path
    mid_lat = lat + 0.8 * (CENTER_LAT - lat)
    mid_lon = lon + 0.8 * (CENTER_LON - lon)
    
    # Line width scaled by percentage of traffic
    line_width = 1 + (row['percentage'] / 5)
    
    # Add the entry point marker
    hourly_fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=dict(
            size=10 + (row['CRZ Entries'] / hourly_total * 30),  # Scale size based on entries
            color='darkblue'
        ),
        name=location
    ))

# Update the layout
hourly_fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=11,
        center=dict(lat=CENTER_LAT, lon=CENTER_LON)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    title=f"Traffic Flow at {selected_hour}:00"
)

st.plotly_chart(hourly_fig, use_container_width=True) 