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
