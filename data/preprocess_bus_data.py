import pandas as pd
import time
import os

print("Starting bus data preprocessing...")
start_time = time.time()

# Check if the processed file already exists
if os.path.exists('processed_bus_data.csv'):
    print("Processed bus data already exists. Delete the file if you want to regenerate it.")
    exit()

# Load the raw bus data
print("Loading raw bus data...")
bus_df = pd.read_csv("MTA_Bus_Hourly_Ridership__Beginning_2025.csv")
print(f"Raw data shape: {bus_df.shape}")

# Convert timestamps and extract components
print("Converting timestamps...")
bus_df['datetime'] = pd.to_datetime(bus_df['transit_timestamp'])
bus_df['date'] = bus_df['datetime'].dt.date
bus_df['hour'] = bus_df['datetime'].dt.hour
bus_df['datehour'] = bus_df['datetime'].dt.strftime('%Y-%m-%d %H:00:00')
bus_df['ridership'] = bus_df['ridership'].astype(float)

# Keep only necessary columns
bus_df = bus_df[['datehour', 'date', 'hour', 'bus_route', 'ridership']]

# Aggregate data by date, hour, and route
print("Aggregating data...")
aggregated_df = bus_df.groupby(['datehour', 'date', 'hour', 'bus_route'])['ridership'].sum().reset_index()

# Get top 25 bus routes by total ridership
print("Identifying top bus routes...")
top_routes = bus_df.groupby('bus_route')['ridership'].sum().nlargest(25).index.tolist()

# Filter for only the top routes
filtered_df = aggregated_df[aggregated_df['bus_route'].isin(top_routes)]

# Save the processed data
print("Saving processed data...")
filtered_df.to_csv('processed_bus_data.csv', index=False)

# Calculate reduction in size
raw_size = bus_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
processed_size = filtered_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
reduction = (1 - processed_size/raw_size) * 100

print(f"Preprocessing complete!")
print(f"Original data: {len(bus_df):,} rows, {raw_size:.2f} MB")
print(f"Processed data: {len(filtered_df):,} rows, {processed_size:.2f} MB")
print(f"Size reduction: {reduction:.1f}%")
print(f"Total processing time: {time.time() - start_time:.2f} seconds")
print(f"Top bus routes kept: {', '.join(top_routes)}")

# Define and save bus route coordinates
bus_routes = {
    # Top 10 routes from previous analysis
    "Q58": [(40.7387, -73.8962), (40.7287, -73.8730), (40.7187, -73.8562)],  # Queens
    "M15+": [(40.7022, -73.9974), (40.7267, -73.9816), (40.7511, -73.9707)],  # Manhattan
    "B6": [(40.6342, -73.9583), (40.6401, -73.9320), (40.6442, -73.9031)],    # Brooklyn
    "Q27": [(40.7111, -73.8301), (40.7255, -73.8193), (40.7367, -73.8124)],   # Queens
    "Q65": [(40.7653, -73.8315), (40.7391, -73.8310), (40.7145, -73.8292)],   # Queens
    "Q25": [(40.7619, -73.8682), (40.7491, -73.8401), (40.7340, -73.8212)],   # Queens
    "M15": [(40.7090, -73.9865), (40.7240, -73.9796), (40.7507, -73.9685)],   # Manhattan
    "M101": [(40.7415, -73.9910), (40.7733, -73.9583), (40.8044, -73.9388)],  # Manhattan
    "B41": [(40.5839, -73.9481), (40.6244, -73.9550), (40.6633, -73.9760)],   # Brooklyn
    "Q10": [(40.6657, -73.8120), (40.6811, -73.8211), (40.7034, -73.8350)],   # Queens
    
    # Additional routes (11-25)
    "Q46": [(40.7274, -73.8162), (40.7375, -73.7960), (40.7387, -73.7727)],   # Queens
    "Q23": [(40.7060, -73.8436), (40.7219, -73.8458), (40.7385, -73.8521)],   # Queens
    "B35": [(40.6061, -73.9724), (40.6386, -73.9620), (40.6425, -73.9462)],   # Brooklyn
    "Q44+": [(40.7483, -73.8232), (40.7220, -73.7991), (40.6965, -73.7778)],  # Queens
    "M86+": [(40.7788, -73.9770), (40.7845, -73.9580), (40.7792, -73.9410)],  # Manhattan
    "Q66": [(40.7584, -73.9310), (40.7618, -73.9129), (40.7525, -73.8902)],   # Queens
    "B1": [(40.5902, -73.9925), (40.6013, -73.9744), (40.6088, -73.9601)],    # Brooklyn
    "BX12+": [(40.8299, -73.9116), (40.8309, -73.8841), (40.8316, -73.8624)], # Bronx
    "B38": [(40.6888, -73.9796), (40.6761, -73.9594), (40.6603, -73.9511)],   # Brooklyn
    "M4": [(40.7582, -73.9929), (40.7871, -73.9692), (40.8108, -73.9501)],    # Manhattan
    "Q33": [(40.7480, -73.8924), (40.7509, -73.8703), (40.7443, -73.8508)],   # Queens
    "M60+": [(40.7829, -73.9490), (40.7744, -73.9280), (40.7686, -73.9116)],  # Manhattan/Queens
    "BX9": [(40.8277, -73.9211), (40.8476, -73.9020), (40.8717, -73.8813)],   # Bronx
    "B54": [(40.7003, -73.9388), (40.7099, -73.9555), (40.7155, -73.9766)],   # Brooklyn
    "B44": [(40.6214, -73.9312), (40.6506, -73.9407), (40.6801, -73.9533)]    # Brooklyn
}

# Save bus route coordinates to a separate file for easy loading
import json
with open('bus_route_coordinates.json', 'w') as f:
    json.dump(bus_routes, f)

print("Bus route coordinates saved to bus_route_coordinates.json") 