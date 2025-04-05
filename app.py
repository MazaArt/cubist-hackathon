import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import math
import colorsys
import random
import time
import os
import json

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
vehicle_types = sorted(df['Vehicle Class'].unique())

# Date and time selection
st.sidebar.header("Filter Data")
selected_date_range = st.sidebar.date_input(
    "Select date range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

st.sidebar.subheader("Vehicle Types")
selected_vehicle_types = []

for vehicle_type in vehicle_types:
    if st.sidebar.checkbox(vehicle_type, value=True):
        selected_vehicle_types.append(vehicle_type)

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
    (df['hour'] <= time_range[1]) &
    (df['Vehicle Class'].isin(selected_vehicle_types))
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

if 'selected_points' not in st.session_state:
    st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())

st.sidebar.header("Entry Points")
st.sidebar.write("Select entry points to focus on:")

# Add select all and deselect all buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Select All", key="select_all", use_container_width=True):
        st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())
        st.rerun()
with col2:
    if st.sidebar.button("Deselect All", key="deselect_all", use_container_width=True):
        st.session_state.selected_points = set()
        st.rerun()

# Use multiselect with the current session state
selected_points = st.sidebar.multiselect(
    "Choose entry points:",
    options=sorted(entry_traffic['Detection Group'].unique()),
    default=list(st.session_state.selected_points)
)

# Update session state with new selection
st.session_state.selected_points = set(selected_points)

# Add some spacing after the selection
st.sidebar.markdown("---")

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

origin, time_flow_tab, before_after_tab = st.tabs(['Time Flow Map', 'Time Flow Map With Subway and Bus', 'Before-After Subway Map'])

with origin:
    pass

with time_flow_tab:
    # Read the processed rideshare data
    @st.cache_data
    def load_rideshare_data():
        rideshare_df = pd.read_csv("eda/processed_rideshare.csv.gz")
        return rideshare_df
    
    try:
        with st.spinner('Loading rideshare data...'):
            rideshare_df = load_rideshare_data()
            
            # Load bus data
            @st.cache_data
            def load_bus_data():
                # Use the preprocessed file instead of the original
                try:
                    # Check if processed file exists
                    if not os.path.exists('processed_bus_data.csv'):
                        st.warning("Processed bus data not found. Run preprocess_bus_data.py first.")
                        return pd.DataFrame()  # Return empty DataFrame
                        
                    # Load the much smaller preprocessed file
                    bus_df = pd.read_csv("processed_bus_data.csv")
                    
                    # Convert date column back to datetime
                    bus_df['date'] = pd.to_datetime(bus_df['date']).dt.date
                    
                    # Load bus route coordinates
                    with open('bus_route_coordinates.json', 'r') as f:
                        bus_routes = json.load(f)
                        
                    return bus_df, bus_routes
                    
                except Exception as e:
                    st.error(f"Error loading processed bus data: {e}")
                    return pd.DataFrame(), {}  # Return empty DataFrame and dict
            
            # Load bus data
            with st.spinner('Loading bus data...'):
                bus_df, bus_routes = load_bus_data()
            
            # Extract unique station complexes and their locations
            station_complexes = rideshare_df[['station_complex', 'latitude', 'longitude']].drop_duplicates()
            
            # Display information about the stations
            st.write(f"Total unique station complexes: {len(station_complexes)}")
            
            # Save the unique stations to a new CSV
            station_complexes.to_csv("station_locations.csv", index=False)
            
            # Function to calculate distance between two coordinates
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate the great circle distance between two points in kilometers"""
                from math import radians, cos, sin, asin, sqrt
                
                # Convert decimal degrees to radians
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                
                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                r = 6371  # Radius of earth in kilometers
                return c * r
            
            # Assign each station to its closest entry point
            entry_points = list(coordinates.keys())
            
            # Create a new column to store the closest entry point for each station
            station_complexes['closest_entry'] = None
            station_complexes['entry_distance'] = float('inf')
            
            # Calculate closest entry point for each station
            for idx, station in station_complexes.iterrows():
                min_distance = float('inf')
                closest_entry = None
                
                for entry in entry_points:
                    entry_lat, entry_lon = coordinates[entry]
                    distance = haversine_distance(
                        station['latitude'], station['longitude'],
                        entry_lat, entry_lon
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_entry = entry
                
                station_complexes.at[idx, 'closest_entry'] = closest_entry
                station_complexes.at[idx, 'entry_distance'] = min_distance
            
            # Create a color mapping for each entry point
            import colorsys
            import random
            
            # Generate colors for each entry point - UPDATED to ensure contrast
            base_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
            ]
            # Shuffle the colors to prevent similar colors being next to each other
            random.shuffle(base_colors)
            entry_colors = dict(zip(entry_points, base_colors[:len(entry_points)]))
            
            # Group stations by their closest entry
            entry_station_groups = station_complexes.groupby('closest_entry')
            
            # Add time-based visualization section
            st.subheader("Traffic Volume Over Time")
            
            # Load population data if available
            @st.cache_data
            def load_population_data():
                try:
                    with open('entry_point_populations.json', 'r') as f:
                        return json.load(f)
                except Exception as e:
                    st.warning(f"Population data not available. Run generate_population_data.py first. Error: {e}")
                    return {}
            
            # Load population data
            population_data = load_population_data()
            
            # Add date and time sliders
            unique_dates = sorted(pd.to_datetime(rideshare_df['datehour']).dt.date.unique())
            
            if len(unique_dates) > 0:
                # Find the closest date to 2025-02-03 in the available dates
                default_date = datetime.strptime("2025-02-03", "%Y-%m-%d").date()
                if default_date not in unique_dates:
                    default_date = unique_dates[0]  # Fallback to first date if 2025-02-03 not available
                
                selected_date = st.date_input(
                    "Select date",
                    value=default_date,
                    min_value=unique_dates[0],
                    max_value=unique_dates[-1]
                )
                
                # Initialize session state for animation if it doesn't exist
                if 'animating' not in st.session_state:
                    st.session_state.animating = False
                    st.session_state.current_hour = 8  # Start at 8am
                
                # Format date for filtering
                date_str = selected_date.strftime("%Y-%m-%d")
                
                # Get all rideshare data for this date
                date_rideshare = rideshare_df[rideshare_df['datehour'].str.startswith(date_str)]
                
                # Get all congestion zone data for this date
                date_congestion = df[df['date'] == selected_date]
                
                # Create columns for time controls
                time_col1, time_col2 = st.columns([3, 1])
                
                with time_col1:
                    if st.session_state.animating:
                        # If animating, use the current hour from session state
                        selected_hour = st.session_state.current_hour
                        # Display a slider but make it non-interactive during animation
                        st.slider("Hour of day (animating...)", 0, 23, selected_hour, disabled=True)
                    else:
                        # Regular interactive slider when not animating
                        selected_hour = st.slider("Select hour of day", 0, 23, 8)  # Default to 8 AM
                
                with time_col2:
                    # Animation control button
                    if st.session_state.animating:
                        if st.button("Stop Animation", type="primary"):
                            st.session_state.animating = False
                            st.rerun()
                    else:
                        if st.button("Animate Through Day", type="primary"):
                            st.session_state.animating = True
                            st.session_state.current_hour = 0  # Start at midnight
                            st.rerun()
                
                # Filter data for the selected hour
                hour_str = f"{date_str} {selected_hour:02d}:00:00"
                rideshare_filtered = date_rideshare[date_rideshare['datehour'] == hour_str]
                congestion_filtered = date_congestion[date_congestion['hour'] == selected_hour]
                
                # Get station traffic volumes for the selected time
                station_traffic = rideshare_filtered.groupby('station_complex')['ridership'].sum().reset_index()
                station_traffic_dict = dict(zip(station_traffic['station_complex'], station_traffic['ridership'])) if not station_traffic.empty else {}
                
                # Get congestion zone entry traffic
                entry_traffic = congestion_filtered.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
                entry_traffic_dict = dict(zip(entry_traffic['Detection Group'], entry_traffic['CRZ Entries'])) if not entry_traffic.empty else {}
                
                # Display debug info to check if data is changing
                if st.checkbox("Show debug info", value=False):
                    st.write(f"Hour: {selected_hour}:00")
                    st.write(f"Entry points with traffic data: {len(entry_traffic_dict)}")
                    if entry_traffic_dict:
                        st.write("Sample entry point traffic:")
                        for i, (entry, volume) in enumerate(list(entry_traffic_dict.items())[:3]):
                            st.write(f"  {entry}: {volume}")
                    st.write(f"Stations with traffic data: {len(station_traffic_dict)}")
                    if station_traffic_dict:
                        st.write("Sample station traffic:")
                        for i, (station, volume) in enumerate(list(station_traffic_dict.items())[:3]):
                            st.write(f"  {station}: {volume}")
                
                # Get max values for scaling
                max_station_traffic = max(station_traffic_dict.values()) if station_traffic_dict else 1
                max_entry_traffic = max(entry_traffic_dict.values()) if entry_traffic_dict else 1
                total_riders = sum(station_traffic_dict.values())
                total_vehicles = sum(entry_traffic_dict.values())
                
                # Create traffic flow map with time dimension
                st.subheader(f"Traffic at {selected_date} {selected_hour:02d}:00")
                
                # Add display toggles
                toggle_col1, toggle_col2, toggle_col3, toggle_col4 = st.columns(4)
                with toggle_col1:
                    show_regions = st.checkbox("Show Entry Point Regions", value=True)
                with toggle_col2:
                    show_subway = st.checkbox("Show Subway Stations", value=True)
                with toggle_col3:
                    show_buses = st.checkbox("Show Top Bus Routes", value=True)
                with toggle_col4:
                    show_population = st.checkbox("Show Population Data", value=False)
                
                # Create a new flow map for traffic visualization
                traffic_map = go.Figure()
                
                # Create regions using approximate Voronoi diagram approach (only if checkbox is checked)
                if show_regions:
                    # Reduce grid size for better performance
                    grid_size = 75  # Reduced from 100
                    
                    # Generate a grid of points covering NYC
                    lat_min, lat_max = 40.55, 40.90  # NYC latitude range
                    lon_min, lon_max = -74.05, -73.85  # NYC longitude range
                    
                    lat_grid = np.linspace(lat_min, lat_max, grid_size)
                    lon_grid = np.linspace(lon_min, lon_max, grid_size)
                    
                    # Create a grid of points
                    grid_points = []
                    for lat in lat_grid:
                        for lon in lon_grid:
                            grid_points.append((lat, lon))
                    
                    # Assign each grid point to the closest entry point
                    grid_assignments = {}
                    for entry in entry_points:
                        grid_assignments[entry] = []
                    
                    for lat, lon in grid_points:
                        min_distance = float('inf')
                        closest_entry = None
                        
                        for entry in entry_points:
                            if entry in coordinates:
                                entry_lat, entry_lon = coordinates[entry]
                                distance = haversine_distance(lat, lon, entry_lat, entry_lon)
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_entry = entry
                        
                        if closest_entry:
                            grid_assignments[closest_entry].append((lat, lon))
                    
                    # Add regions to the map
                    for entry, points in grid_assignments.items():
                        if not points:
                            continue
                            
                        # Extract lat/lon arrays
                        lats = [p[0] for p in points]
                        lons = [p[1] for p in points]
                        
                        # Add the region as a scatter density map
                        traffic_map.add_trace(go.Densitymapbox(
                            lat=lats,
                            lon=lons,
                            z=[1] * len(lats),  # Uniform density
                            radius=20,  # Increased from 15 to make regions larger
                            colorscale=[[0, entry_colors[entry]], [1, entry_colors[entry]]],
                            showscale=False,
                            hoverinfo='none',
                            opacity=0.5,  # Increased from 0.3
                            name=f"{entry} Region"
                        ))
                
                # Add bus routes first (lowest layer) if enabled
                if show_buses and not bus_df.empty:
                    # Filter bus data for the selected hour
                    hour_str = f"{date_str} {selected_hour:02d}:00:00"
                    bus_filtered = bus_df[bus_df['datehour'] == hour_str]
                    
                    # Get top 25 busiest bus routes for this hour
                    top_bus_routes = bus_filtered.groupby('bus_route')['ridership'].sum().reset_index()
                    top_bus_routes = top_bus_routes.sort_values('ridership', ascending=False).head(25)
                    
                    # Show debug info about displayed routes
                    if st.checkbox("Show bus route debug info", value=False):
                        st.write(f"Available coordinate routes: {len(bus_routes)}")
                        st.write(f"Routes with ridership data for this hour: {len(top_bus_routes)}")
                        
                        # Count routes that can be displayed (have both coordinates and data)
                        displayable_routes = [r for r in top_bus_routes['bus_route'] if r in bus_routes]
                        st.write(f"Routes that can be displayed: {len(displayable_routes)}")
                        st.write(f"Bus routes with coordinates: {', '.join(sorted(bus_routes.keys()))}")
                        st.write(f"Top bus routes for this hour: {', '.join(top_bus_routes['bus_route'].tolist())}")
                    
                    # Add top bus routes to map
                    bus_route_added_to_legend = False  # Track if we've added a bus route to the legend yet
                    
                    for _, route_row in top_bus_routes.iterrows():
                        route_name = route_row['bus_route']
                        ridership = route_row['ridership']
                        
                        # Skip if we don't have coordinates for this route
                        if route_name not in bus_routes:
                            continue
                            
                        # Get route coordinates
                        route_coords = bus_routes[route_name]
                        lats = [coord[0] for coord in route_coords]
                        lons = [coord[1] for coord in route_coords]
                        
                        # Calculate line width based on ridership
                        max_ridership = top_bus_routes['ridership'].max()
                        line_width = 1 + (ridership / max_ridership * 4) if max_ridership > 0 else 1
                        
                        # Add bus route as a line
                        traffic_map.add_trace(go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode='lines',
                            line=dict(
                                width=line_width,
                                color='#FF5722'  # Distinct orange color for bus routes
                            ),
                            name="Bus Routes" if not bus_route_added_to_legend else "",  # Only show "Bus Routes" once in legend
                            hovertext=f"Bus {route_name}: {int(ridership):,} riders",
                            hoverinfo='text',
                            showlegend=not bus_route_added_to_legend,
                            opacity=0.35  # More transparent
                        ))
                        
                        # Update flag to avoid adding more bus routes to legend
                        if not bus_route_added_to_legend:
                            bus_route_added_to_legend = True
                        
                        # Add bus route markers at endpoints (but don't show in legend)
                        traffic_map.add_trace(go.Scattermapbox(
                            lat=[lats[0], lats[-1]],
                            lon=[lons[0], lons[-1]],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color='#FF5722',
                                opacity=0.7
                            ),
                            hovertext=[f"Bus {route_name} Start", f"Bus {route_name} End"],
                            hoverinfo='text',
                            showlegend=False
                        ))
                
                # Add traffic-scaled entry points (no arrows)
                for entry in entry_points:
                    if entry in coordinates:
                        lat, lon = coordinates[entry]
                        traffic_volume = entry_traffic_dict.get(entry, 0)
                        
                        # Add the entry point marker with size scaled by traffic
                        size = 10  # Larger minimum size for ports
                        if max_entry_traffic > 0 and traffic_volume > 0:
                            size = 10 + (traffic_volume / max_entry_traffic * 40)
                        
                        # Simple approach with same color as related stations but larger size
                        traffic_map.add_trace(go.Scattermapbox(
                            lat=[lat],
                            lon=[lon],
                            mode='markers+text',
                            marker=dict(
                                size=size,
                                color=entry_colors[entry],  # Same color as associated stations
                                opacity=1.0
                            ),
                            text=[f"{entry}"],  # Simple text label
                            textposition="top center",
                            hovertext=[f"{entry}: {traffic_volume:,} vehicles"],
                            name=f"{entry}",
                            hoverinfo='text'
                        ))
                
                # Add traffic-scaled stations with distinct appearance
                if show_subway:
                    # Track if we've added a station to the legend for each entry point
                    entry_added_to_legend = set()
                    
                    for entry, group in entry_station_groups:
                        if entry in entry_colors:
                            # Create lists for coordinates and sizes
                            lats = []
                            lons = []
                            sizes = []
                            hover_texts = []
                            
                            for _, station in group.iterrows():
                                station_name = station['station_complex']
                                traffic_volume = station_traffic_dict.get(station_name, 0)
                                
                                # Scale size based on traffic (very small for stations)
                                size = 3  # Minimum size
                                if max_station_traffic > 0 and traffic_volume > 0:
                                    size = 3.5 + (traffic_volume / max_station_traffic * 10)  # Much smaller max size
                                
                                lats.append(station['latitude'])
                                lons.append(station['longitude'])
                                sizes.append(size)
                                hover_texts.append(f"{station_name}: {traffic_volume:,} riders")
                            
                            # Determine if this entry should be added to legend
                            show_in_legend = entry not in entry_added_to_legend
                            legend_name = entry if show_in_legend else ""
                            
                            # Add the stations with guaranteed supported properties
                            traffic_map.add_trace(go.Scattermapbox(
                                lat=lats,
                                lon=lons,
                                mode='markers',
                                marker=dict(
                                    size=sizes,
                                    color=entry_colors[entry],  # Same color as the associated port
                                    opacity=0.7
                                ),
                                text=hover_texts,
                                name=legend_name,  # Just show the entry point name without "Stations near"
                                hoverinfo='text',
                                showlegend=show_in_legend
                            ))
                            
                            # Mark this entry as added to legend
                            entry_added_to_legend.add(entry)
                
                # Add population data visualization if enabled
                if show_population and population_data:
                    # Create a dataframe for the population data
                    pop_data = []
                    for entry in entry_points:
                        if entry in population_data:
                            pop_data.append({
                                'Entry Point': entry,
                                'Population': population_data[entry]['total_population'],
                                'Borough': population_data[entry]['borough'],
                                'Color': entry_colors.get(entry, '#333333')
                            })
                    
                    if pop_data:
                        pop_df = pd.DataFrame(pop_data)
                        
                        # Create a population bar chart
                        pop_chart = px.bar(
                            pop_df, 
                            x='Entry Point', 
                            y='Population',
                            color='Entry Point',
                            color_discrete_map={entry: color for entry, color in zip(pop_df['Entry Point'], pop_df['Color'])},
                            title='Estimated Population by Entry Point Region',
                            labels={'Population': 'Estimated Population', 'Entry Point': 'Entry Point'}
                        )
                        
                        # Customize the layout
                        pop_chart.update_layout(
                            xaxis=dict(tickangle=-45),
                            yaxis=dict(title='Population'),
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=100)
                        )
                        
                        # Display the chart
                        st.plotly_chart(pop_chart, use_container_width=True)
                        
                        # Add annotations to the map showing population
                        for entry in entry_points:
                            if entry in coordinates and entry in population_data:
                                lat, lon = coordinates[entry]
                                pop = population_data[entry]['total_population']
                                
                                # Calculate a position offset from the port (further away from Manhattan center)
                                center_lat, center_lon = 40.7380, -73.9855  # Manhattan center
                                
                                # Calculate direction away from center
                                dir_lat = lat - center_lat
                                dir_lon = lon - center_lon
                                
                                # Normalize and apply larger offset (move further away from Manhattan)
                                dist = np.sqrt(dir_lat**2 + dir_lon**2)
                                if dist > 0:
                                    offset_lat = 0.018 * (dir_lat/dist)  # About 1.8km away (increased from 1.2km)
                                    offset_lon = 0.018 * (dir_lon/dist)
                                else:
                                    offset_lat, offset_lon = 0.018, 0  # Default offset if same as center
                                
                                # Calculate circle position (offset from the port location)
                                circle_lat = lat + offset_lat
                                circle_lon = lon + offset_lon
                                
                                # Normalize population to get circle size
                                max_pop = max(population_data[e]['total_population'] for e in population_data)
                                rel_size = pop / max_pop
                                
                                # Create population circle
                                size = 20 + (rel_size * 80)  # Scale from 20 to 100
                                
                                # Use circles instead of bars - more reliable in Scattermapbox
                                traffic_map.add_trace(go.Scattermapbox(
                                    lat=[circle_lat],
                                    lon=[circle_lon],
                                    mode='markers',
                                    marker=dict(
                                        size=size,
                                        color='rgb(128, 0, 128)',  # Purple
                                        opacity=0.5  # Reduced from 0.7 to make it more transparent
                                    ),
                                    hovertext=f"{entry} Region: {pop:,} people",
                                    hoverinfo='text',
                                    showlegend=False
                                ))
                        
                        # Add population info box
                        st.info("""
                            **Population Data Information**
                            
                            The population estimates shown are approximations based on neighborhoods near each entry point.
                            Purple circles on the map indicate the estimated population in the region around each entry point.
                            
                            The circles are positioned in the areas associated with each entry point.
                            Circle size corresponds to relative population - larger circles indicate higher populations.
                        """)
                
                # Update the layout
                traffic_map.update_layout(
                    mapbox=dict(
                        style="carto-positron",
                        zoom=11,
                        center=dict(lat=40.7380, lon=-73.9855)
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=600,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        itemsizing="constant"
                    )
                )
                
                # Display the traffic map
                traffic_map_placeholder = st.empty()
                traffic_map_placeholder.plotly_chart(traffic_map, use_container_width=True)
                
                # Add some metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Total Vehicle Entries", f"{total_vehicles:,}")
                
                with metrics_col2:
                    st.metric("Total Subway Riders", f"{total_riders:,}")
                
                # Show busiest entry points and stations
                busy_col1, busy_col2 = st.columns(2)
                
                with busy_col1:
                    st.subheader("Busiest Entry Points")
                    top_entries = entry_traffic.sort_values('CRZ Entries', ascending=False).head(5) if not entry_traffic.empty else pd.DataFrame(columns=['Detection Group', 'CRZ Entries'])
                    
                    st.dataframe(
                        top_entries,
                        column_config={
                            'Detection Group': 'Entry Point',
                            'CRZ Entries': 'Vehicles'
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                with busy_col2:
                    st.subheader("Busiest Subway Stations")
                    top_stations = station_traffic.sort_values('ridership', ascending=False).head(5) if not station_traffic.empty else pd.DataFrame(columns=['station_complex', 'ridership'])
                    
                    st.dataframe(
                        top_stations,
                        column_config={
                            'station_complex': 'Station',
                            'ridership': 'Riders'
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Animation logic
                if st.session_state.animating:
                    # Increment the hour for next iteration
                    st.session_state.current_hour = (st.session_state.current_hour + 1) % 24
                    time.sleep(1.0)  # Wait 1 second between hours
                    st.rerun()
                
                # Add line chart section for traffic trends
                st.subheader("Traffic Trends Throughout the Day")
                
                # Get the selected entry points from the sidebar
                selected_entry_points = list(st.session_state.selected_points)
                
                if selected_entry_points:
                    # Get hourly traffic data for each selected entry point
                    hourly_data = []
                    
                    for hour in range(24):
                        # Get vehicle entries for this hour
                        hour_str = f"{date_str} {hour:02d}:00:00"
                        hour_congestion = date_congestion[date_congestion['hour'] == hour]
                        hour_rideshare = date_rideshare[date_rideshare['datehour'] == hour_str]
                        hour_bus = bus_df[bus_df['datehour'] == hour_str]
                        
                        # Filter by selected entry points
                        hour_entry_traffic = hour_congestion[hour_congestion['Detection Group'].isin(selected_entry_points)]
                        
                        # Get stations associated with selected entry points
                        selected_stations = station_complexes[
                            station_complexes['closest_entry'].isin(selected_entry_points)
                        ]['station_complex'].unique()
                        
                        # Filter subway ridership by selected stations
                        hour_station_traffic = hour_rideshare[
                            hour_rideshare['station_complex'].isin(selected_stations)
                        ]
                        
                        # Filter top 10 bus routes for the selected region
                        # This is a simplification - ideally you would have actual geographic associations
                        top_routes = bus_df.groupby('bus_route')['ridership'].sum().nlargest(10).index
                        hour_bus_traffic = hour_bus[hour_bus['bus_route'].isin(top_routes)]
                        
                        # Sum up traffic
                        total_entries = hour_entry_traffic['CRZ Entries'].sum()
                        total_subway = hour_station_traffic['ridership'].sum()
                        total_bus = hour_bus_traffic['ridership'].sum()
                        
                        hourly_data.append({
                            'hour': hour,
                            'vehicle_entries': total_entries,
                            'subway_ridership': total_subway,
                            'bus_ridership': total_bus
                        })
                    
                    # Create a DataFrame for plotting
                    hourly_df = pd.DataFrame(hourly_data)
                    
                    # Create line chart
                    line_chart = go.Figure()
                    
                    # Add vehicle entries line - primary y-axis
                    line_chart.add_trace(go.Scatter(
                        x=hourly_df['hour'],
                        y=hourly_df['vehicle_entries'],
                        mode='lines+markers',
                        name='Vehicle Entries',
                        line=dict(color='rgb(49, 130, 189)', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add subway ridership line - secondary y-axis
                    line_chart.add_trace(go.Scatter(
                        x=hourly_df['hour'],
                        y=hourly_df['subway_ridership'],
                        mode='lines+markers',
                        name='Subway Ridership',
                        line=dict(color='rgb(204, 84, 94)', width=3),
                        marker=dict(size=8),
                        yaxis="y2"  # Use secondary y-axis
                    ))
                    
                    # Add bus ridership line - also on secondary y-axis
                    line_chart.add_trace(go.Scatter(
                        x=hourly_df['hour'],
                        y=hourly_df['bus_ridership'],
                        mode='lines+markers',
                        name='Bus Ridership',
                        line=dict(color='rgb(255, 87, 34)', width=3, dash='dot'),  # Orange dotted line
                        marker=dict(size=8, symbol='square'),
                        yaxis="y2"  # Use secondary y-axis
                    ))
                    
                    # Add a vertical line for the current selected hour
                    line_chart.add_vline(
                        x=selected_hour,
                        line_width=2,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Current: {selected_hour}:00",
                        annotation_position="top right"
                    )
                    
                    # Update layout for better appearance with dual y-axes
                    line_chart.update_layout(
                        title=f"Traffic Trends for Selected Entry Points ({len(selected_entry_points)} selected)",
                        xaxis_title="Hour of Day",
                        yaxis=dict(
                            title=dict(
                                text="Vehicle Entries",
                                font=dict(color="rgb(49, 130, 189)")
                            ),
                            tickfont=dict(color="rgb(49, 130, 189)")
                        ),
                        yaxis2=dict(
                            title=dict(
                                text="Public Transit Ridership",
                                font=dict(color="rgb(204, 84, 94)")
                            ),
                            tickfont=dict(color="rgb(204, 84, 94)"),
                            anchor="x",
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        hovermode="x unified",
                        margin=dict(l=20, r=50, t=60, b=20),
                        height=400
                    )
                    
                    # Set x-axis to show all hours
                    line_chart.update_xaxes(
                        tickvals=list(range(24)),
                        ticktext=[f"{h}:00" for h in range(24)],
                        tickangle=45
                    )
                    
                    # Display the chart
                    st.plotly_chart(line_chart, use_container_width=True)
                    
                    # Display explanation
                    st.info("""
                        This chart shows traffic trends throughout the day for the entry points selected in the sidebar.
                        - Blue line: Total vehicle entries through selected entry points
                        - Red line: Total subway ridership at stations near the selected entry points
                        - Orange dotted line: Total bus ridership for top 10 routes
                        - Dashed line: Current selected hour
                    """)
                else:
                    st.warning("Please select at least one entry point in the sidebar to see traffic trends.")
            
            else:
                st.error("No date data available in the rideshare dataset")
    except Exception as e:
        st.error(f"Error processing rideshare data: {e}")
    
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

# Add time animation
# Add date selector for hourly visualization


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