import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv", nrows=1000)

regions = df.groupby('Detection Region')['CRZ Entries'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
regions.plot(kind='bar')
plt.title('Traffic by Entry Region (Sample Data)')
plt.ylabel('Number of Entries')
plt.xlabel('Region')
plt.tight_layout()
plt.savefig('region_entries.png')

# Path to the CSV file
csv_file_path = 'MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv'

# Initialize an empty set to store unique detection groups
unique_detection_groups = set()

# Read the CSV file in chunks
chunk_size = 10000  # Adjust the chunk size if needed
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    # Add unique entries from the 'Detection Group' column to the set
    unique_detection_groups.update(chunk['Detection Group'].dropna().unique())

# Output the number of distinct entries
print(unique_detection_groups)