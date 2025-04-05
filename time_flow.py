import random
from constants import COORDINATES, BASE_COLORS, CENTER_LAT, CENTER_LON
from helper import haversine_distance
import plotly.graph_objects as go
import streamlit as st
import numpy as np

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
