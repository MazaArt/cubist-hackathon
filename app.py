import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from helper import haversine_distance

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
    df_2['datehour'] = pd.to_datetime(df['datehour'])
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
st.sidebar.header("Filter Data")
selected_date_range = st.sidebar.date_input(
    "Select date range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Ensure we have a valid date range
if len(selected_date_range) != 2:
    selected_date_range = (df['date'].min(), df['date'].max())

# Time range slider
time_range = st.sidebar.slider(
    "Select time range (hours)",
    0, 23, (0, 23)
)

# Filter data based on selections
filtered_df = df[
    (df['date'] >= selected_date_range[0]) & 
    (df['date'] <= selected_date_range[1]) & 
    (df['hour'] >= time_range[0]) & 
    (df['hour'] <= time_range[1])
]

# Create a map of entry points
st.subheader("Traffic Flow by Entry Point")
entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
total_traffic = entry_traffic['CRZ Entries'].sum()
entry_traffic['percentage'] = (entry_traffic['CRZ Entries'] / total_traffic * 100).round(1)

# Initialize session state for selected points if not exists
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())

# Add custom CSS for smaller buttons
st.markdown("""
    <style>
    .stButton > button {
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        height: auto;
        min-height: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Entry point selection in sidebar
st.sidebar.header("Entry Points")
st.sidebar.write("Select entry points to focus on:")

# Add select all and deselect all buttons in a row with smaller text
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Select All", key="select_all", use_container_width=True):
        st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())
        st.rerun()
with col2:
    if st.sidebar.button("Deselect All", key="deselect_all", use_container_width=True):
        st.session_state.selected_points = set()
        st.rerun()

# Add selection buttons in the sidebar with smaller text
for location in sorted(entry_traffic['Detection Group'].unique()):
    is_selected = location in st.session_state.selected_points
    if st.sidebar.button(
        f"{'✓ ' if is_selected else ''}{location}",
        key=f"btn_{location}",
        type="primary" if is_selected else "secondary",
        use_container_width=True
    ):
        if is_selected:
            st.session_state.selected_points.remove(location)
        else:
            st.session_state.selected_points.add(location)
        st.rerun()

# Add some spacing after the buttons
st.sidebar.markdown("---")

# Filter traffic data based on selected points
filtered_traffic = entry_traffic[entry_traffic['Detection Group'].isin(st.session_state.selected_points)]

# Calculate total traffic for selected points only
selected_total = filtered_traffic['CRZ Entries'].sum()
filtered_traffic['percentage'] = (filtered_traffic['CRZ Entries'] / selected_total * 100).round(1)


# Create the map
fig = go.Figure()

# Central point for Manhattan Congestion Zone
center_lat, center_lon = 40.7380, -73.9855

# Add entry points with arrows pointing to the center
for _, row in entry_traffic.iterrows():
    location = row['Detection Group']
    
    # Skip if we don't have coordinates
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Determine marker color based on selection
    marker_color = 'green' if location in st.session_state.selected_points else 'darkblue'

    hover_text = f"{location}: {filtered_traffic[filtered_traffic['Detection Group'] == location]['percentage'].iloc[0]}%" \
    if location in st.session_state.selected_points else location
    
    # Add the entry point marker
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=dict(size=12, color=marker_color),  # Reduced marker size
        text=[hover_text],
        textposition="top center",
        name=location,
        hoverinfo='text'
    ))

# Update the layout
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=11,
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=400,  # Reduced map height
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

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
    if location not in coordinates:
        continue
        
    lat, lon = coordinates[location]
    
    # Calculate arrow path
    mid_lat = lat + 0.8 * (center_lat - lat)
    mid_lon = lon + 0.8 * (center_lon - lon)
    
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
        center=dict(lat=center_lat, lon=center_lon)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    title=f"Traffic Flow at {selected_hour}:00"
)

st.plotly_chart(hourly_fig, use_container_width=True) 