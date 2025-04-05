# sidebar.py
import streamlit as st
import pandas as pd
from src.constants import COORDINATES

def render_sidebar(df):
    """
    Renders all sidebar widgets for filtering data:
      - date range
      - time range
      - entry points
      - vehicle types
    Updates streamlit session state as needed.
    Returns:
      selected_date_range, time_range, selected_points, selected_vehicle_types, filtered_df, entry_traffic
    """

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

    # Available entry points
    available_points = sorted(df['Detection Group'].unique())

    # Initialize session state for selected_points if not exists
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = set(available_points)

    # Add custom CSS for smaller buttons (optional)
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
    col1, col2 = st.sidebar.columns(2)

    # "Select All" and "Deselect All"
    with col1:
        if st.sidebar.button("Select All", key="select_all", use_container_width=True):
            st.session_state.selected_points = set(available_points)
            st.experimental_rerun()

    # with col2:
    #     if st.sidebar.button("Deselect All", key="deselect_all", use_container_width=True):
    #         st.session_state.selected_points = set()
    #         st.experimental_rerun()

    # Multi-select for entry points
    selected_points = st.sidebar.multiselect(
        "Choose entry points:",
        options=available_points,
        default=list(st.session_state.selected_points),
        key="entry_points_select"
    )

    # Update session_state
    st.session_state.selected_points = set(selected_points)

    # Add some spacing
    st.sidebar.markdown("---")

    # Vehicle Types Selection
    vehicle_types = sorted(df['Vehicle Class'].unique())
    st.sidebar.header("Vehicle Types")
    selected_vehicle_types = []
    for vehicle_type in vehicle_types:
        if st.sidebar.checkbox(vehicle_type, value=True, key=f"vehicle_type_{vehicle_type}"):
            selected_vehicle_types.append(vehicle_type)

    # --- Filter the DataFrame based on user selections ---
    filtered_df = df[
        (df['date'] >= selected_date_range[0]) &
        (df['date'] <= selected_date_range[-1]) &
        (df['hour'] >= time_range[0]) &
        (df['hour'] <= time_range[1]) &
        (df['Vehicle Class'].isin(selected_vehicle_types))
    ]

    # If no vehicle types are selected, build an empty traffic DataFrame with 0
    if len(selected_vehicle_types) == 0:
        st.warning("No vehicle types selected. Please select at least one vehicle type.")
        entry_traffic = pd.DataFrame(
            {'Detection Group': list(COORDINATES.keys()), 'CRZ Entries': [0]*len(COORDINATES)}
        )
    else:
        entry_traffic = filtered_df.groupby('Detection Group')['CRZ Entries'].sum().reset_index()

    return (
        selected_date_range,
        time_range,
        st.session_state.selected_points,
        selected_vehicle_types,
        filtered_df,
        entry_traffic
    )
