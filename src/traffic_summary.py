from plotly.graph_objects import Figure, Scattermapbox, Frame
import plotly.graph_objects as go
from src.constants import COORDINATES, CENTER_LON, CENTER_LAT
from src.helper import traffic_to_color, add_congestion_region_overlay

def create_main_map(entry_traffic, selected_points):
    fig = go.Figure()
    
    for _, row in entry_traffic.iterrows():
        location = row['Detection Group']
        if location not in COORDINATES:
            continue
            
        lat, lon = COORDINATES[location]
        marker_color = 'green' if location in selected_points else 'darkblue'
        
        # Create hover text with percentage for selected points
        hover_text = location
        if location in selected_points:
            entries = row['CRZ Entries']
            percentage = row['percentage']
            hover_text = f"{location}<br>{percentage:.1f}%"
        
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(size=12, color=marker_color),
            text=[hover_text],
            textposition="top center",
            name=location,
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=11,
            center=dict(lat=CENTER_LAT, lon=CENTER_LON)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig = add_congestion_region_overlay(fig)
    return fig

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
            mapbox=dict(style="carto-positron", zoom=11, center=dict(lat=CENTER_LAT, lon=CENTER_LON)),
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