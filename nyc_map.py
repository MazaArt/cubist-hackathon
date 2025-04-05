import plotly.graph_objects as go
import streamlit as st
from constants import COORDINATES

def create_map(entry_traffic, filtered_traffic):
    # Create the map
    fig = go.Figure()

    # Central point for Manhattan Congestion Zone
    center_lat, center_lon = 40.7380, -73.9855

    # Add entry points with arrows pointing to the center
    for _, row in entry_traffic.iterrows():
        location = row['Detection Group']
        
        # Skip if we don't have coordinates
        if location not in COORDINATES:
            continue
            
        lat, lon = COORDINATES[location]
        
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
    return fig