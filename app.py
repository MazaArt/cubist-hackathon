import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import math

# Function to calculate distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

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

# Load subway station data
@st.cache_data
def load_subway_stations():
    try:
        subway_df = pd.read_csv("subway_stations.csv")
        return subway_df
    except Exception as e:
        st.error(f"Error loading subway data: {e}")
        return pd.DataFrame()

# Map coordinates for the entry points based on Detection Group
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

# Define colors for each port of entry
port_colors = {
    "Brooklyn Bridge": "#FF5733",
    "West Side Highway at 60th St": "#33FF57",
    "West 60th St": "#5733FF",
    "Queensboro Bridge": "#FF33A1",
    "Queens Midtown Tunnel": "#33A1FF",
    "Lincoln Tunnel": "#A133FF",
    "Holland Tunnel": "#FFD733",
    "FDR Drive at 60th St": "#33FFD7",
    "East 60th St": "#FF8333",
    "Williamsburg Bridge": "#8333FF",
    "Manhattan Bridge": "#33FFA1",
    "Hugh L. Carey Tunnel": "#A1FF33"
}

# Function to find nearest port of entry for each subway station
@st.cache_data
def map_stations_to_ports(subway_stations, coordinates):
    # Create a new column to store the closest port
    subway_stations = subway_stations.copy()
    subway_stations['closest_port'] = ""
    subway_stations['distance_km'] = 0.0
    
    # For each station, find the closest port
    for idx, station in subway_stations.iterrows():
        min_distance = float('inf')
        closest_port = ""
        
        for port, (port_lat, port_lon) in coordinates.items():
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

# Load the data with a progress indicator
with st.spinner('Loading data... This may take a moment.'):
    try:
        df = load_data()
        subway_stations = load_subway_stations()
        
        # Map subway stations to nearest port of entry
        if not subway_stations.empty:
            subway_stations = map_stations_to_ports(subway_stations, coordinates)
            
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

# Initialize session state for selected points if not exists
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = set()

# Get traffic by detection group
entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
total_traffic = entry_traffic['CRZ Entries'].sum()
entry_traffic['percentage'] = (entry_traffic['CRZ Entries'] / total_traffic * 100).round(1)

# Create the map
fig = go.Figure()

# Central point for Manhattan Congestion Zone
center_lat, center_lon = 40.7580, -73.9855

# Add entry points with arrows pointing to the center
for _, row in entry_traffic.iterrows():
    location = row['Detection Group']
    
    # Skip if we don't have coordinates
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Use the port's color instead of red/blue for consistency
    marker_color = port_colors[location]
    
    # Add the entry point marker
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(size=15, color=marker_color),
        text=[f"{location}: {row['percentage']}% ({row['CRZ Entries']:,} entries)"],
        textposition="top center",
        name=location,
        hoverinfo='text'  # Only show the text on hover
    ))

# Add subway stations to the map
if not subway_stations.empty:
    # For each port of entry, add its subway stations with matching color
    for port in coordinates.keys():
        # Filter stations for this port
        port_stations = subway_stations[subway_stations['closest_port'] == port]
        
        if not port_stations.empty:
            fig.add_trace(go.Scattermapbox(
                lat=port_stations['latitude'],
                lon=port_stations['longitude'],
                mode='markers',
                marker=dict(size=5, color=port_colors[port], opacity=0.7),
                text=port_stations.apply(lambda x: f"{x['station_complex']} (nearest to {x['closest_port']})", axis=1),
                name=f"Stations near {port}",
                hoverinfo='text'
            ))

# Update the layout
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=11.3,
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Display the map
st.plotly_chart(fig, use_container_width=True)

# Add selection buttons in a more compact layout
st.write("Select entry points to focus on:")
cols = st.columns(4)  # Create 4 columns
for i, location in enumerate(sorted(entry_traffic['Detection Group'].unique())):
    col = cols[i % 4]  # Distribute buttons across columns
    is_selected = location in st.session_state.selected_points
    if col.button(
        f"{'✓' if is_selected else ''} {location}",
        key=f"btn_{location}",
        type="primary" if is_selected else "secondary"
    ):
        if is_selected:
            st.session_state.selected_points.remove(location)
        else:
            st.session_state.selected_points.add(location)
        st.rerun()

# Filter traffic data based on selected points
if st.session_state.selected_points:
    filtered_traffic = entry_traffic[entry_traffic['Detection Group'].isin(st.session_state.selected_points)]
else:
    filtered_traffic = entry_traffic

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
hourly_entry_traffic = hourly_data.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
hourly_total = hourly_entry_traffic['CRZ Entries'].sum()
hourly_entry_traffic['percentage'] = (hourly_entry_traffic['CRZ Entries'] / hourly_total * 100).round(1)

# Create hourly map
hourly_fig = go.Figure()

# Add entry points with arrows pointing to the center
for _, row in hourly_entry_traffic.iterrows():
    location = row['Detection Group']
    
    # Skip if we don't have coordinates
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Use the port's color instead of red/blue for consistency
    marker_color = port_colors[location]
    
    # Add the entry point marker
    hourly_fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(size=15, color=marker_color),
        text=[f"{row['percentage']}%"],
        textposition="top center",
        name=location
    ))

# Update the layout
hourly_fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=11.3,
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    title=f"Traffic Flow at {selected_hour}:00"
)

st.plotly_chart(hourly_fig, use_container_width=True)

# Add subway station information section
if not subway_stations.empty:
    st.subheader("Subway Stations")
    
    # Summary of stations by nearest port of entry
    port_counts = subway_stations['closest_port'].value_counts().reset_index()
    port_counts.columns = ['Port of Entry', 'Number of Stations']
    
    st.write("### Stations by Nearest Port of Entry")
    
    # Create horizontal bar chart
    fig_ports = px.bar(
        port_counts,
        y='Port of Entry',
        x='Number of Stations',
        orientation='h',
        color='Port of Entry',
        color_discrete_map=port_colors,
        title='Number of Subway Stations by Nearest Port of Entry'
    )
    st.plotly_chart(fig_ports, use_container_width=True)
    
    # Allow filtering subway stations by port of entry
    port_filter = st.selectbox(
        "Filter by Port of Entry:",
        options=["All"] + sorted(coordinates.keys())
    )
    
    if port_filter != "All":
        filtered_subway_stations = subway_stations[subway_stations['closest_port'] == port_filter]
        st.write(f"Showing {len(filtered_subway_stations)} stations closest to {port_filter}")
    else:
        filtered_subway_stations = subway_stations
        st.write(f"Showing all {len(subway_stations)} stations")
    
    # Allow selecting a specific station
    selected_station = st.selectbox(
        "Select a subway station to highlight:",
        options=["None"] + filtered_subway_stations['station_complex'].sort_values().tolist()
    )
    
    if selected_station != "None":
        # Filter to show only the selected station
        station_data = subway_stations[subway_stations['station_complex'] == selected_station]
        
        # Show details about the selected station
        st.write(f"### Station Details: {selected_station}")
        st.write(f"Station ID: {station_data['station_complex_id'].values[0]}")
        st.write(f"Location: ({station_data['latitude'].values[0]}, {station_data['longitude'].values[0]})")
        st.write(f"Nearest Port of Entry: {station_data['closest_port'].values[0]}")
        st.write(f"Distance to Port: {station_data['distance_km'].values[0]:.2f} km")
        
        # Create a map focusing on the selected station
        st.write("### Station Location")
        station_map = go.Figure()
        
        # Add the station
        station_map.add_trace(go.Scattermapbox(
            lat=[station_data['latitude'].values[0]],
            lon=[station_data['longitude'].values[0]],
            mode='markers',
            marker=dict(size=15, color=port_colors[station_data['closest_port'].values[0]]),
            text=[selected_station],
            name='Selected Station'
        ))
        
        # Add the nearest port
        port_name = station_data['closest_port'].values[0]
        port_lat, port_lon = coordinates[port_name]
        
        station_map.add_trace(go.Scattermapbox(
            lat=[port_lat],
            lon=[port_lon],
            mode='markers',
            marker=dict(size=20, color=port_colors[port_name], symbol='square'),
            text=[f"Port: {port_name}"],
            name='Nearest Port'
        ))
        
        # Add a line connecting them
        station_map.add_trace(go.Scattermapbox(
            lat=[station_data['latitude'].values[0], port_lat],
            lon=[station_data['longitude'].values[0], port_lon],
            mode='lines',
            line=dict(width=2, color=port_colors[port_name]),
            name='Path to Nearest Port'
        ))
        
        station_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=12,
                center=dict(
                    lat=(station_data['latitude'].values[0] + port_lat) / 2,
                    lon=(station_data['longitude'].values[0] + port_lon) / 2
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(station_map, use_container_width=True)
    
    # Show table with filtered subway stations
    st.write("### Station List")
    st.dataframe(
        filtered_subway_stations.sort_values('station_complex'),
        column_config={
            'station_complex': 'Station Name',
            'station_complex_id': 'Station ID',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'closest_port': 'Nearest Port of Entry',
            'distance_km': 'Distance (km)'
        },
        hide_index=True
    ) 