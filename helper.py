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

import plotly.graph_objects as go
from constants import REGION_BOUNDARY

def add_congestion_region_overlay(fig, fill_color='rgba(255, 0, 0, 0.2)', line_color='red', line_width=2):
    points = list(REGION_BOUNDARY.values())

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    
    lats.append(lats[0])
    lons.append(lons[0])
    
    new_fig = go.Figure()
    
    new_fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=line_width, color=line_color),
        fill='toself',
        fillcolor=fill_color,
        name='Congestion Relief Region',
        hoverinfo='none'
    ))
    
    for trace in fig.data:
        new_fig.add_trace(trace)
    
    new_fig.update_layout(fig.layout)
    
    return new_fig 