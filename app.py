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

# Get traffic by detection group
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
st.sidebar.write("Filter the entry points shown on the map and charts.")
st.sidebar.markdown("---")

# Multiselect instead of individual buttons
all_locations = sorted(entry_traffic['Detection Group'].unique())
selected = st.sidebar.multiselect(
    "Choose entry points:",
    options=all_locations,
    default=list(st.session_state.selected_points)
)

# Update session state
st.session_state.selected_points = set(selected)

# Add select all and deselect all buttons in a row with smaller text
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Select All", use_container_width=True):
        st.session_state.selected_points = set(all_locations)
        st.rerun()
with col2:
    if st.button("Deselect All", use_container_width=True):
        st.session_state.selected_points = set()
        st.rerun()



# Filter traffic data based on selected points
filtered_traffic = entry_traffic[entry_traffic['Detection Group'].isin(st.session_state.selected_points)]

# Calculate total traffic for selected points only
selected_total = filtered_traffic['CRZ Entries'].sum()
filtered_traffic['percentage'] = (filtered_traffic['CRZ Entries'] / selected_total * 100).round(1)

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

st.subheader("Animated Traffic Flow Over Time")
st.markdown("Visualize how traffic volume at each entry point evolves across days and hours.")

# 🚧 Handle empty selection
if not st.session_state.selected_points:
    st.warning("No entry points selected. Please choose one or more from the sidebar to see traffic animation.")
    st.stop()  # Or skip this block entirely

# Filter to selected date range and selected entry points
animated_df = df[
    (df['date'] >= selected_date_range[0]) &
    (df['date'] <= selected_date_range[1]) &
    (df['Detection Group'].isin(st.session_state.selected_points))
].copy()

# Add coordinates
animated_df['lat'] = animated_df['Detection Group'].map(lambda x: coordinates.get(x, (None, None))[0])
animated_df['lon'] = animated_df['Detection Group'].map(lambda x: coordinates.get(x, (None, None))[1])
animated_df.dropna(subset=['lat', 'lon'], inplace=True)

# Combine date and hour into one animation timestamp
animated_df['timestamp'] = animated_df['Toll Hour'].dt.strftime('%Y-%m-%d %H:%M')

# Group by timestamp and entry point
grouped = (
    animated_df
    .groupby(['timestamp', 'Detection Group', 'lat', 'lon'], as_index=False)['CRZ Entries']
    .sum()
)

# Normalize CRZ Entries for manual color mapping
max_traffic = grouped['CRZ Entries'].max()
min_traffic = grouped['CRZ Entries'].min()

# Map traffic to color: green (low) to red (high)
def traffic_to_color(value):
    scale = (value - min_traffic) / (max_traffic - min_traffic + 1e-5)
    r = int(255 * scale)
    g = int(255 * (1 - scale))
    return f'rgb({r},{g},0)'

grouped['color'] = grouped['CRZ Entries'].apply(traffic_to_color)

# Build animation using go.Figure for full color control
from plotly.graph_objects import Figure, Scattermapbox, Frame

frames = []
timestamps = grouped['timestamp'].unique()

# Initialize figure
initial_data = grouped[grouped['timestamp'] == timestamps[0]]
fig = Figure(
    data=[
        Scattermapbox(
            lat=initial_data['lat'],
            lon=initial_data['lon'],
            mode='markers',
            marker=dict(
                size=initial_data['CRZ Entries'] / max_traffic * 25 + 8,
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
            style="open-street-map",  # or dynamic `mapbox_style`
            zoom=11,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title="Animated Traffic Flow Over Time",
        showlegend=False,

        # 🎮 Play/Pause buttons — move them higher (y=0.15)
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
            "pad": {"r": 10, "t": 30},  # top padding between button and plot
            "x": 0,
            "xanchor": "left",
            "y": 0,  # moved slightly above the slider
            "yanchor": "top"
        }],

        # 🕓 Slider — keep it lower (y=0)
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
            "y": 0,  # keep at bottom
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

# Add frames for animation
for t in timestamps:
    frame_data = grouped[grouped['timestamp'] == t]
    frames.append(Frame(
        data=[
            Scattermapbox(
                lat=frame_data['lat'],
                lon=frame_data['lon'],
                mode='markers',
                marker=dict(
                    size=frame_data['CRZ Entries'] / max_traffic * 25 + 8,
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

fig.frames = frames

# Show animated chart
st.plotly_chart(fig, use_container_width=True)
