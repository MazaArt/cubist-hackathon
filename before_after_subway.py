import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def create_subway_diff_map(df, selected_date, selected_hour):
    """
    Create an interactive map of subway ridership differences, with circles at each station.
    Green circles for positive differences, red for negative. Circle size is proportional to
    the absolute magnitude of the difference.
    """
    # Filter the DataFrame to the selected date/hour
    date_str = selected_date.strftime("%Y-%m-%d")
    hour_str = f"{date_str} {selected_hour:02d}:00:00"
    df_filtered = df[df["datehour"] == pd.to_datetime(hour_str)]

    # Group by station (and its coordinates) to get one point per station
    station_diffs = df_filtered
    # Handle edge cases (e.g., no data)
    if station_diffs.empty:
        st.warning("No data available for this date and hour.")
        return go.Figure()

    # Find the max absolute diff to scale the marker sizes
    max_abs_diff = station_diffs["ridership_diff"].abs().max()
    if pd.isna(max_abs_diff) or max_abs_diff == 0:
        max_abs_diff = 1

    # Build the map
    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=station_diffs["latitude"],
            lon=station_diffs["longitude"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                # Make size reflect magnitude of difference
                size=station_diffs["ridership_diff"].abs() / max_abs_diff * 20 + 5,
                # Green if diff >= 0, else red
                color=station_diffs["ridership_diff"].apply(lambda x: "green" if x >= 0 else "red"),
                opacity=0.7
            ),
            # Hover text shows station name and difference
            text=station_diffs.apply(
                lambda row: f"{row['station_complex']}: {row['ridership_diff']:+,} Riders",
                axis=1,
            ),
            hoverinfo="text",
            name="Subway Diff"
        )
    )

    # Configure map layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=10,  
            center=dict(lat=40.75, lon=-73.95),  # approximate center on NYC
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        title=f"Subway Ridership Difference — {date_str} {selected_hour:02d}:00",
    )

    return fig

def create_before_after_visualization(subway_ridership_diff_df):
    # Let the user select a date and hour
    unique_dates = sorted(subway_ridership_diff_df["datehour"].dt.date.unique())
    selected_date = st.selectbox("Select a date", unique_dates)

    # Filter the DataFrame so that the hour choices only come from the selected date
    filtered_for_date = subway_ridership_diff_df[subway_ridership_diff_df["datehour"].dt.date == selected_date]
    unique_hours = sorted(filtered_for_date["datehour"].dt.hour.unique())
    selected_hour = st.selectbox("Select an hour (0–23)", unique_hours)

    # Build the map
    fig = create_subway_diff_map(subway_ridership_diff_df, pd.to_datetime(selected_date), selected_hour)
    return fig
    
    
