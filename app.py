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
from constants import COORDINATES, CENTER_LAT, CENTER_LON, BASE_COLORS
import time
from time_flow import create_traffic_map
from traffic_summary import create_main_map, create_animation
from before_after_subway import create_before_after_visualization

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
    df = pd.read_csv("data/MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv.gz")
    df['Toll Date'] = pd.to_datetime(df['Toll Date'])
    df['Toll Hour'] = pd.to_datetime(df['Toll Hour'])
    df['date'] = df['Toll Date'].dt.date
    df['hour'] = df['Hour of Day']
    return df

@st.cache_data
def load_rideshare_data():
    return pd.read_csv("data/processed_rideshare.csv.gz")

@st.cache_data
def load_bus_data():
    if not os.path.exists('data/processed_bus_data.csv.gz'):
        st.warning("Processed bus data not found. Run preprocess_bus_data.py first.")
        return pd.DataFrame(), {}
        
    bus_df = pd.read_csv('data/processed_bus_data.csv.gz')
    bus_df['date'] = pd.to_datetime(bus_df['date']).dt.date
    
    with open('data/bus_route_coordinates.json', 'r') as f:
        bus_routes = json.load(f)
        
    return bus_df, bus_routes

@st.cache_data
def load_subway_diff_data():
    df = pd.read_csv('data/subway_ridership_diff.csv.gz')
    df["datehour"] = pd.to_datetime(df["datehour"])
    return df

@st.cache_data
def load_population_data():
    try:
        with open('data/entry_point_populations.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Population data not available. Error: {e}")
        return {}

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
    
    # Time range selection
    time_range = st.sidebar.slider(
        "Select time range (hours)",
        0, 23, (0, 23)
    )

    # Use multiselect with the current session state
    available_points = sorted(df['Detection Group'].unique())
    # Initialize session state for selected points if not exists
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = set(available_points)

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

    # Entry Points Selection
    st.sidebar.header("Entry Points")

    # Add select all and deselect all buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Select All", key="select_all", use_container_width=True):
            st.session_state.selected_points = set(available_points)
            st.rerun()
    with col2:
        if st.sidebar.button("Deselect All", key="deselect_all", use_container_width=True):
            st.session_state.selected_points = set()
            st.rerun()

    last_selected_points = st.session_state.selected_points
    # Update session state with new selection
    st.session_state.selected_points = st.sidebar.multiselect(
        "Choose entry points:",
        available_points,
        st.session_state.selected_points
    )

    if last_selected_points != st.session_state.selected_points:
        st.rerun()

    # Add some spacing after the selection
    st.sidebar.markdown("---")

    # Vehicle Types Selection
    st.sidebar.header("Vehicle Types")
    selected_vehicle_types = []

    for vehicle_type in vehicle_types:
        if st.sidebar.checkbox(vehicle_type, value=True, key=f"vehicle_type_{vehicle_type}"):
            selected_vehicle_types.append(vehicle_type)

    # Add some spacing after vehicle types
    st.sidebar.markdown("---")

    # Filter data based on selections
    filtered_df = df[
        (df['date'] >= selected_date_range[0]) & 
        (df['date'] <= selected_date_range[-1]) & 
        (df['hour'] >= time_range[0]) & 
        (df['hour'] <= time_range[1]) &
        (df['Vehicle Class'].isin(selected_vehicle_types))
    ]
    # Get traffic by detection group
    if len(selected_vehicle_types) == 0:
        st.warning("No vehicle types selected. Please select at least one vehicle type to see traffic data.")
        entry_traffic = pd.DataFrame({'Detection Group': list(COORDINATES.keys()), 'CRZ Entries': [0] * len(COORDINATES)})
    else:
        entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()
    # Filter traffic data based on selected points
    filtered_traffic = entry_traffic[entry_traffic['Detection Group'].isin(st.session_state.selected_points)]
    # Calculate total traffic for selected points only
    selected_total = filtered_traffic['CRZ Entries'].sum()
    filtered_traffic.loc['percentage'] = (filtered_traffic['CRZ Entries'] / max(selected_total, 1) * 100).round(1)
    total_traffic = entry_traffic['CRZ Entries'].sum()
    entry_traffic['percentage'] = (entry_traffic['CRZ Entries'] / max(total_traffic, 1) * 100).round(1)
    
    # Main content
    st.subheader("Traffic Flow by Entry Point")
    st.plotly_chart(create_main_map(entry_traffic, st.session_state.selected_points), use_container_width=True)
    
    # Tab layout
    traffic_summary, time_flow_subway_bus_tab, before_after_tab = st.tabs(['Traffic Summary', 'Time Flow Map With Subway and Bus', 'Before-After Subway Map'])
    with traffic_summary:
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
            (df['date'] <= selected_date_range[-1]) &
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
        
    with time_flow_subway_bus_tab:
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
                    time.sleep(.5)
                    st.rerun()

                pass
                
        except Exception as e:
            st.error(f"Error processing rideshare data: {e}")
    
    with before_after_tab:
        with st.spinner('Loading additional data...'):
            subway_ridership_diff_df = load_subway_diff_data()
            before_after_map = create_before_after_visualization(subway_ridership_diff_df)
            st.plotly_chart(before_after_map, use_container_width=True)

if __name__ == "__main__":
    main()