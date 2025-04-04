import pandas as pd
import matplotlib.pyplot as plt

# Read the first 1000 rows to test
print("Reading CSV file...")
df = pd.read_csv("MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20250404.csv", nrows=1000)

# Display basic info about the dataset
print("\nDataset Sample Info:")
print(f"Columns: {df.columns.tolist()}")
print(f"Total rows read: {len(df)}")
print(f"\nSample data types:")
print(df.dtypes)

# Print unique values for categorical columns
print("\nUnique Detection Regions:")
print(df['Detection Region'].unique())

print("\nUnique Vehicle Classes:")
print(df['Vehicle Class'].unique())

# Create a simple plot of entries by Detection Region
print("\nCreating plot...")
regions = df.groupby('Detection Region')['CRZ Entries'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
regions.plot(kind='bar')
plt.title('Traffic by Entry Region (Sample Data)')
plt.ylabel('Number of Entries')
plt.xlabel('Region')
plt.tight_layout()
plt.savefig('region_entries.png')
print("Plot saved as region_entries.png")

print("\nDone!") 