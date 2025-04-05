import streamlit as st


def create_date_filter(df):
    selected_date_range = st.sidebar.date_input(
        "Select date range",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )

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
        (df['hour'] <= time_range[1])
    ]
    return filtered_df, selected_date_range

def create_entrypoint_selection_buttons(entry_traffic):
    # Add select all and deselect all buttons in a row with smaller text
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Select All", key="select_all", use_container_width=True):
            st.session_state.selected_points = set(entry_traffic['Detection Group'].unique())
            st.rerun()
    with col2:
        if st.sidebar.button("Deselect All", key="deselect_all", use_container_width=True):
            st.session_state.selected_points = set()
            st.rerun()

    # Add selection buttons in the sidebar with smaller text
    for location in sorted(entry_traffic['Detection Group'].unique()):
        is_selected = location in st.session_state.selected_points
        if st.sidebar.button(
            f"{'✓ ' if is_selected else ''}{location}",
            key=f"btn_{location}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            if is_selected:
                st.session_state.selected_points.remove(location)
            else:
                st.session_state.selected_points.add(location)
            st.rerun()
    return

def create_sidebar(df):
    filtered_df, selected_date_range = create_date_filter(df)
    # Create a map of entry points
    st.subheader("Traffic Flow by Entry Point")
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

    # Entry point selection in sidebar
    st.sidebar.header("Entry Points")
    st.sidebar.write("Select entry points to focus on:")
    create_entrypoint_selection_buttons(entry_traffic)
    
    show_subway_stations = st.sidebar.toggle("Show Subway Stations", value=True)
    # Add some spacing after the buttons
    st.sidebar.markdown("---")
    return entry_traffic, show_subway_stations, filtered_df, selected_date_range