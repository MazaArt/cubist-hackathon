# MTA Congestion Relief Zone Traffic Visualization

This Streamlit application visualizes traffic flow into the Manhattan Congestion Relief Zone based on the MTA dataset.

## Features

- Interactive map showing traffic flow from different entry points
- Traffic percentage visualization with varying arrow widths
- Date and time filtering via sidebar controls
- Hourly traffic volume chart
- Summary table of traffic by entry point

## Setup and Running

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the data file is in the same directory:
   - `MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv`

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. The app will open in your default web browser, typically at `http://localhost:8501`

## About the Data

The dataset provides the number of crossings into the Congestion Relief Zone by crossing location and vehicle class, in 10-minute intervals. This data should not be used for revenue calculations, as entries do not include information about exemption statuses, payment methods, and repeat entries.