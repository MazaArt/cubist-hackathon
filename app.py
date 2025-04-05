import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import json
import random
from plotly.graph_objects import Figure, Scattermapbox, Frame

# Constants
COORDINATES = {
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
CENTER_LAT, CENTER_LON = 40.7380, -73.9855
BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
]

# Page Configuration
def configure_page():
    st.set_page_config(
        page_title="MTA Congestion Relief Zone Traffic Visualization",
        page_icon="🚗",
        layout="wide"
    )
    st.title("MTA Congestion Relief Zone Traffic Flow")
    st.markdown("Visualizing traffic flow into the Congestion Relief Zone by entry points")

# Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv")
    df['Toll Date'] = pd.to_datetime(df['Toll Date'])
    df['Toll Hour'] = pd.to_datetime(df['Toll Hour'])
    df['date'] = df['Toll Date'].dt.date
    df['hour'] = df['Hour of Day']
    return df

@st.cache_data
def load_rideshare_data():
    return pd.read_csv("processed_rideshare.csv")

@st.cache_data
def load_bus_data():
    if not os.path.exists('processed_bus_data.csv'):
        st.warning("Processed bus data not found. Run preprocess_bus_data.py first.")
        return pd.DataFrame(), {}
        
    bus_df = pd.read_csv("processed_bus_data.csv")
    bus_df['date'] = pd.to_datetime(bus_df['date']).dt.date
    
    with open('bus_route_coordinates.json', 'r') as f:
        bus_routes = json.load(f)
        
    return bus_df, bus_routes

@st.cache_data
def load_population_data():
    try:
        with open('entry_point_populations.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Population data not available. Error: {e}")
        return {}

# Utility Functions
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371  # Earth radius in km

def traffic_to_color(value, min_val, max_val):
    scale = (value - min_val) / (max_val - min_val + 1e-5)
    r = int(255 * scale)
    g = int(255 * (1 - scale))
    return f'rgb({r},{g},0)'

# Visualization Components
def create_main_map(entry_traffic, selected_points):
    fig = go.Figure()
    
    for _, row in entry_traffic.iterrows():
        location = row['Detection Group']
        if location not in COORDINATES:
            continue
            
        lat, lon = COORDINATES[location]
        marker_color = 'green' if location in selected_points else 'darkblue'
        
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(size=12, color=marker_color),
            text=[location],
            textposition="top center",
            name=location,
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=11,
            center=dict(lat=CENTER_LAT, lon=CENTER_LON)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_traffic_map(entry_traffic, selected_points, rideshare_df, bus_df, bus_routes, population_data, selected_date, selected_hour):
    """Create the interactive traffic map visualization with subway, bus, and population data"""
    # Generate colors for each entry point
    random.shuffle(BASE_COLORS)
    entry_colors = dict(zip(COORDINATES.keys(), BASE_COLORS[:len(COORDINATES)]))
    
    # Process station data
    station_complexes = rideshare_df[['station_complex', 'latitude', 'longitude']].drop_duplicates()
    station_complexes['closest_entry'] = None
    station_complexes['entry_distance'] = float('inf')
    
    # Assign each station to its closest entry point
    for idx, station in station_complexes.iterrows():
        min_distance = float('inf')
        closest_entry = None
        
        for entry in COORDINATES:
            entry_lat, entry_lon = COORDINATES[entry]
            distance = haversine_distance(
                station['latitude'], station['longitude'],
                entry_lat, entry_lon
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_entry = entry
        
        station_complexes.at[idx, 'closest_entry'] = closest_entry
        station_complexes.at[idx, 'entry_distance'] = min_distance
    
    # Group stations by their closest entry
    entry_station_groups = station_complexes.groupby('closest_entry')
    
    # Filter data for selected date and hour
    date_str = selected_date.strftime("%Y-%m-%d")
    hour_str = f"{date_str} {selected_hour:02d}:00:00"
    
    rideshare_filtered = rideshare_df[rideshare_df['datehour'] == hour_str]
    station_traffic = rideshare_filtered.groupby('station_complex')['ridership'].sum().reset_index()
    station_traffic_dict = dict(zip(station_traffic['station_complex'], station_traffic['ridership'])) if not station_traffic.empty else {}
    
    # Get max values for scaling
    max_station_traffic = max(station_traffic_dict.values()) if station_traffic_dict else 1
    total_riders = sum(station_traffic_dict.values())
    
    # Create the map figure
    traffic_map = go.Figure()
    
    # Add regions using approximate Voronoi diagram approach
    if st.session_state.get('show_regions', True):
        grid_size = 75  # Reduced for better performance
        
        # Generate a grid of points covering NYC
        lat_min, lat_max = 40.55, 40.90
        lon_min, lon_max = -74.05, -73.85
        
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        # Assign grid points to closest entry point
        grid_assignments = {entry: [] for entry in COORDINATES}
        
        for lat in lat_grid:
            for lon in lon_grid:
                min_distance = float('inf')
                closest_entry = None
                
                for entry in COORDINATES:
                    entry_lat, entry_lon = COORDINATES[entry]
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
                
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            
            traffic_map.add_trace(go.Densitymapbox(
                lat=lats,
                lon=lons,
                z=[1] * len(lats),
                radius=20,
                colorscale=[[0, entry_colors[entry]], [1, entry_colors[entry]]],
                showscale=False,
                hoverinfo='none',
                opacity=0.5,
                name=f"{entry} Region"
            ))
    
    # Add bus routes if enabled and data exists
    if st.session_state.get('show_buses', True) and not bus_df.empty:
        bus_filtered = bus_df[bus_df['datehour'] == hour_str]
        top_bus_routes = bus_filtered.groupby('bus_route')['ridership'].sum().reset_index()
        top_bus_routes = top_bus_routes.sort_values('ridership', ascending=False).head(25)
        
        bus_route_added_to_legend = False
        
        for _, route_row in top_bus_routes.iterrows():
            route_name = route_row['bus_route']
            ridership = route_row['ridership']
            
            if route_name not in bus_routes:
                continue
                
            route_coords = bus_routes[route_name]
            lats = [coord[0] for coord in route_coords]
            lons = [coord[1] for coord in route_coords]
            
            max_ridership = top_bus_routes['ridership'].max()
            line_width = 1 + (ridership / max_ridership * 4) if max_ridership > 0 else 1
            
            traffic_map.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=line_width, color='#FF5722'),
                name="Bus Routes" if not bus_route_added_to_legend else "",
                hovertext=f"Bus {route_name}: {int(ridership):,} riders",
                hoverinfo='text',
                showlegend=not bus_route_added_to_legend,
                opacity=0.35
            ))
            
            if not bus_route_added_to_legend:
                bus_route_added_to_legend = True
            
            # Add bus route markers at endpoints
            traffic_map.add_trace(go.Scattermapbox(
                lat=[lats[0], lats[-1]],
                lon=[lons[0], lons[-1]],
                mode='markers',
                marker=dict(size=6, color='#FF5722', opacity=0.7),
                hovertext=[f"Bus {route_name} Start", f"Bus {route_name} End"],
                hoverinfo='text',
                showlegend=False
            ))
    
    # Add entry points with traffic volume scaling
    for entry in COORDINATES:
        if entry in selected_points:
            lat, lon = COORDINATES[entry]
            traffic_volume = entry_traffic[entry_traffic['Detection Group'] == entry]['CRZ Entries'].sum()
            
            size = 10 + (traffic_volume / entry_traffic['CRZ Entries'].max() * 40) if not entry_traffic.empty else 10
            
            traffic_map.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers+text',
                marker=dict(size=size, color=entry_colors[entry], opacity=1.0),
                text=[f"{entry}"],
                textposition="top center",
                hovertext=[f"{entry}: {traffic_volume:,} vehicles"],
                name=f"{entry}",
                hoverinfo='text'
            ))
    
    # Add subway stations if enabled
    if st.session_state.get('show_subway', True):
        entry_added_to_legend = set()
        
        for entry, group in entry_station_groups:
            if entry in entry_colors and entry in selected_points:
                lats, lons, sizes, hover_texts = [], [], [], []
                
                for _, station in group.iterrows():
                    station_name = station['station_complex']
                    traffic_volume = station_traffic_dict.get(station_name, 0)
                    
                    size = 3.5 + (traffic_volume / max_station_traffic * 10) if max_station_traffic > 0 else 3
                    
                    lats.append(station['latitude'])
                    lons.append(station['longitude'])
                    sizes.append(size)
                    hover_texts.append(f"{station_name}: {traffic_volume:,} riders")
                
                show_in_legend = entry not in entry_added_to_legend
                legend_name = entry if show_in_legend else ""
                
                traffic_map.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(size=sizes, color=entry_colors[entry], opacity=0.7),
                    text=hover_texts,
                    name=legend_name,
                    hoverinfo='text',
                    showlegend=show_in_legend
                ))
                
                entry_added_to_legend.add(entry)
    
    # Add population data if enabled
    if st.session_state.get('show_population', False) and population_data:
        for entry in selected_points:
            if entry in COORDINATES and entry in population_data:
                lat, lon = COORDINATES[entry]
                pop = population_data[entry]['total_population']
                
                # Calculate position offset from the port
                dir_lat = lat - CENTER_LAT
                dir_lon = lon - CENTER_LON
                
                dist = np.sqrt(dir_lat**2 + dir_lon**2)
                if dist > 0:
                    offset_lat = 0.018 * (dir_lat/dist)
                    offset_lon = 0.018 * (dir_lon/dist)
                else:
                    offset_lat, offset_lon = 0.018, 0
                
                circle_lat = lat + offset_lat
                circle_lon = lon + offset_lon
                
                max_pop = max(population_data[e]['total_population'] for e in population_data)
                rel_size = pop / max_pop
                size = 20 + (rel_size * 80)
                
                traffic_map.add_trace(go.Scattermapbox(
                    lat=[circle_lat],
                    lon=[circle_lon],
                    mode='markers',
                    marker=dict(size=size, color='rgb(128, 0, 128)', opacity=0.5),
                    hovertext=f"{entry} Region: {pop:,} people",
                    hoverinfo='text',
                    showlegend=False
                ))
    
    # Update the layout
    traffic_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=11,
            center=dict(lat=CENTER_LAT, lon=CENTER_LON)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemsizing="constant"
        ),
        title=f"Traffic at {selected_date} {selected_hour:02d}:00"
    )
    
    return traffic_map

def create_animation(grouped, max_traffic, min_traffic):
    timestamps = grouped['timestamp'].unique()
    frames = []

    initial_data = grouped[grouped['timestamp'] == timestamps[0]]
    fig = Figure(
        data=[
            Scattermapbox(
                lat=initial_data['lat'],
                lon=initial_data['lon'],
                mode='markers',
                marker=dict(
                    size=initial_data['CRZ Entries'] / max_traffic * 25 + 8,
                    color=initial_data['CRZ Entries'].apply(lambda x: traffic_to_color(x, min_traffic, max_traffic)),
                    opacity=0.8
                ),
                text=initial_data.apply(lambda row: f"{row['Detection Group']}<br>Traffic: {row['CRZ Entries']:,}", axis=1),
                hoverinfo='text'
            )
        ],
        layout=dict(
            mapbox=dict(style="open-street-map", zoom=11, center=dict(lat=CENTER_LAT, lon=CENTER_LON)),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
            title="Animated Traffic Flow Over Time",
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
                        color=frame_data['CRZ Entries'].apply(lambda x: traffic_to_color(x, min_traffic, max_traffic)),
                        opacity=0.8
                    ),
                    text=frame_data.apply(lambda row: f"{row['Detection Group']}<br>Traffic: {row['CRZ Entries']:,}", axis=1),
                    hoverinfo='text'
                )
            ],
            name=t
        ))

    fig.frames = frames
    return fig

# Main App
def main():
    configure_page()
    
    # Load data with progress indicators
    with st.spinner('Loading data... This may take a moment.'):
        try:
            df = load_data()
            entry_points = df['Detection Region'].unique()
            vehicle_types = sorted(df['Vehicle Class'].unique())
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # Sidebar filters
    st.sidebar.header("Filter Data")
    
    # Date range selection
    selected_date_range = st.sidebar.date_input(
        "Select date range",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )
    
    # Vehicle type selection
    st.sidebar.subheader("Vehicle Types")
    selected_vehicle_types = [
        vt for vt in vehicle_types 
        if st.sidebar.checkbox(vt, value=True)
    ]
    
    # Time range selection
    time_range = st.sidebar.slider(
        "Select time range (hours)",
        0, 23, (0, 23)
    )
    
    # Filter data
    filtered_df = df[
        (df['date'] >= selected_date_range[0]) & 
        (df['date'] <= selected_date_range[1]) & 
        (df['hour'] >= time_range[0]) & 
        (df['hour'] <= time_range[1]) &
        (df['Vehicle Class'].isin(selected_vehicle_types))
    ]
    
    # Entry point selection
    st.sidebar.header("Entry Points")
    entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
    
    # Initialize session state for selected points
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())
    
    # Selection controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="select_all"):
            st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())
            st.rerun()
    with col2:
        if st.button("Deselect All", key="deselect_all"):
            st.session_state.selected_points = set()
            st.rerun()
    
    selected_points = st.sidebar.multiselect(
        "Choose entry points:",
        options=sorted(entry_traffic['Detection Group'].unique()),
        default=list(st.session_state.selected_points)
    )
    st.session_state.selected_points = set(selected_points)
    
    # Main content
    st.subheader("Traffic Flow by Entry Point")
    st.plotly_chart(create_main_map(entry_traffic, st.session_state.selected_points), use_container_width=True)
    
    # Tab layout
    origin, time_flow_tab, before_after_tab = st.tabs(['Time Flow Map', 'Time Flow Map With Subway and Bus', 'Before-After Subway Map'])
    
    with time_flow_tab:
        try:
            with st.spinner('Loading additional data...'):
                rideshare_df = load_rideshare_data()
                bus_df, bus_routes = load_bus_data()
                population_data = load_population_data()
                
                    # Choose default date
                unique_dates = sorted(pd.to_datetime(rideshare_df['datehour']).dt.date.unique())
                default_date = datetime.strptime("2025-02-03", "%Y-%m-%d").date()
                if default_date not in unique_dates:
                    default_date = unique_dates[0]

                selected_date = st.date_input("Select date", value=default_date, min_value=unique_dates[0], max_value=unique_dates[-1])

                # Choose hour
                if 'animating' not in st.session_state:
                    st.session_state.animating = False
                    st.session_state.current_hour = 8

                time_col1, time_col2 = st.columns([3, 1])
                with time_col1:
                    if st.session_state.animating:
                        selected_hour = st.session_state.current_hour
                        st.slider("Hour (animating...)", 0, 23, selected_hour, disabled=True)
                    else:
                        selected_hour = st.slider("Select hour", 0, 23, 8)

                with time_col2:
                    if st.session_state.animating:
                        if st.button("Stop Animation", type="primary"):
                            st.session_state.animating = False
                            st.rerun()
                    else:
                        if st.button("Animate Through Day", type="primary"):
                            st.session_state.animating = True
                            st.session_state.current_hour = 0
                            st.rerun()

                # Create traffic map
                entry_traffic_filtered = filtered_df[filtered_df['date'] == selected_date]
                entry_traffic_hour = entry_traffic_filtered[entry_traffic_filtered['hour'] == selected_hour]
                traffic_map = create_traffic_map(
                    entry_traffic=entry_traffic_hour.groupby('Detection Group')['CRZ Entries'].sum().reset_index(),
                    selected_points=st.session_state.selected_points,
                    rideshare_df=rideshare_df,
                    bus_df=bus_df,
                    bus_routes=bus_routes,
                    population_data=population_data,
                    selected_date=selected_date,
                    selected_hour=selected_hour
                )
                st.plotly_chart(traffic_map, use_container_width=True)

                # Animate
                if st.session_state.animating:
                    st.session_state.current_hour = (st.session_state.current_hour + 1) % 24
                    time.sleep(1)
                    st.rerun()

                pass
                
        except Exception as e:
            st.error(f"Error processing rideshare data: {e}")
    
    with before_after_tab:
        st.write("TODO: Before-After Subway Map implementation")
    
    # Traffic summary and other visualizations
    st.subheader("Traffic Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        total_traffic = filtered_df[filtered_df['Detection Group'].isin(st.session_state.selected_points)]['CRZ Entries'].sum()
        st.metric("Total Traffic", f"{total_traffic:,} vehicles")
        
        # Breakdown by detection group
        filtered_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
        filtered_traffic = filtered_traffic[filtered_traffic['Detection Group'].isin(st.session_state.selected_points)]
        filtered_traffic['percentage'] = (filtered_traffic['CRZ Entries'] / total_traffic * 100).round(1)
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
        hourly_traffic = filtered_df[
            filtered_df['Detection Group'].isin(st.session_state.selected_points)
        ].groupby('hour')['CRZ Entries'].sum().reset_index()
        
        fig_hourly = px.line(
            hourly_traffic,
            x='hour',
            y='CRZ Entries',
            title='Traffic Volume by Hour',
            labels={'hour': 'Hour of Day', 'CRZ Entries': 'Number of Vehicles'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Animated Traffic Section
    st.subheader("Animated Traffic Flow Over Time")
    st.markdown("Visualize how traffic volume at each entry point evolves across days and hours.")

    if not st.session_state.selected_points:
        st.warning("No entry points selected. Please choose one or more from the sidebar.")
        return

    animated_df = df[
        (df['date'] >= selected_date_range[0]) &
        (df['date'] <= selected_date_range[1]) &
        (df['Detection Group'].isin(st.session_state.selected_points))
    ].copy()

    # Attach coordinates
    animated_df['lat'] = animated_df['Detection Group'].map(lambda x: COORDINATES.get(x, (None, None))[0])
    animated_df['lon'] = animated_df['Detection Group'].map(lambda x: COORDINATES.get(x, (None, None))[1])
    animated_df.dropna(subset=['lat', 'lon'], inplace=True)
    animated_df['timestamp'] = animated_df['Toll Hour'].dt.strftime('%Y-%m-%d %H:%M')

    # Group by timestamp and entry point
    grouped = (
        animated_df
        .groupby(['timestamp', 'Detection Group', 'lat', 'lon'], as_index=False)['CRZ Entries']
        .sum()
    )

    if grouped.empty:
        st.warning("No data available for the selected filters to generate animation.")
        return

    max_traffic = grouped['CRZ Entries'].max()
    min_traffic = grouped['CRZ Entries'].min()

    animated_fig = create_animation(grouped, max_traffic, min_traffic)
    st.plotly_chart(animated_fig, use_container_width=True)

if __name__ == "__main__":
    main()