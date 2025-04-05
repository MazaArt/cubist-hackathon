import pandas as pd
import json
import numpy as np
import os

print("Generating population data for NYC regions...")

# Define mapping between entry points and NYC boroughs/neighborhoods
# This is an approximation based on the location of each entry point
entry_point_regions = {
    "Brooklyn Bridge": {"borough": "Brooklyn/Manhattan", "neighborhoods": ["DUMBO", "Brooklyn Heights", "Financial District"]},
    "West Side Highway at 60th St": {"borough": "Manhattan", "neighborhoods": ["Upper West Side", "Lincoln Square", "Midtown West"]},
    "West 60th St": {"borough": "Manhattan", "neighborhoods": ["Upper West Side", "Midtown West", "Columbus Circle"]},
    "Queensboro Bridge": {"borough": "Queens/Manhattan", "neighborhoods": ["Long Island City", "Sutton Place", "Upper East Side"]},
    "Queens Midtown Tunnel": {"borough": "Queens/Manhattan", "neighborhoods": ["Long Island City", "Murray Hill", "Midtown East"]},
    "Lincoln Tunnel": {"borough": "Manhattan/New Jersey", "neighborhoods": ["Hell's Kitchen", "Midtown West"]},
    "Holland Tunnel": {"borough": "Manhattan/New Jersey", "neighborhoods": ["Tribeca", "SoHo", "West Village"]},
    "FDR Drive at 60th St": {"borough": "Manhattan", "neighborhoods": ["Upper East Side", "Sutton Place", "Midtown East"]},
    "East 60th St": {"borough": "Manhattan", "neighborhoods": ["Upper East Side", "Midtown East"]},
    "Williamsburg Bridge": {"borough": "Brooklyn/Manhattan", "neighborhoods": ["Williamsburg", "Lower East Side"]},
    "Manhattan Bridge": {"borough": "Brooklyn/Manhattan", "neighborhoods": ["Chinatown", "DUMBO", "Downtown Brooklyn"]},
    "Hugh L. Carey Tunnel": {"borough": "Brooklyn/Manhattan", "neighborhoods": ["Red Hook", "Financial District", "Battery Park"]}
}

# Approximate population data for NYC neighborhoods (based on census data)
# These are approximations and would need to be replaced with actual census data in a real application
neighborhood_populations = {
    "Financial District": 61000,
    "Battery Park": 13000,
    "Chinatown": 116000,
    "SoHo": 11000,
    "Tribeca": 18000,
    "Lower East Side": 72000,
    "West Village": 34000,
    "DUMBO": 12000,
    "Brooklyn Heights": 23000,
    "Downtown Brooklyn": 28000,
    "Williamsburg": 152000,
    "Red Hook": 11000,
    "Midtown East": 45000,
    "Midtown West": 61000,
    "Murray Hill": 29000,
    "Hell's Kitchen": 45000,
    "Upper East Side": 217000,
    "Upper West Side": 209000,
    "Sutton Place": 21000,
    "Lincoln Square": 32000,
    "Columbus Circle": 17000,
    "Long Island City": 85000
}

# Calculate the approximate population for each entry point region
entry_point_populations = {}

for entry_point, region_info in entry_point_regions.items():
    neighborhoods = region_info["neighborhoods"]
    
    # Calculate total population for all associated neighborhoods
    total_population = sum(neighborhood_populations.get(neighborhood, 0) for neighborhood in neighborhoods)
    
    # Calculate average population density (simplistic approach for demonstration)
    average_population = total_population / len(neighborhoods) if neighborhoods else 0
    
    entry_point_populations[entry_point] = {
        "total_population": int(total_population),
        "neighborhoods": neighborhoods,
        "borough": region_info["borough"],
        "avg_population_density": int(average_population)
    }

# Add some random variation to make the data more realistic
# In reality, this would be replaced with actual census data
for entry_point in entry_point_populations:
    # Add +/- 10% random variation to make the data look more natural
    variation = np.random.uniform(0.9, 1.1)
    entry_point_populations[entry_point]["total_population"] = int(
        entry_point_populations[entry_point]["total_population"] * variation
    )

# Save the data to a JSON file
with open("entry_point_populations.json", "w") as f:
    json.dump(entry_point_populations, f, indent=2)

print("Population data generated and saved to entry_point_populations.json")

# Print a summary for review
print("\nPopulation Summary by Entry Point:")
for entry_point, data in sorted(entry_point_populations.items(), key=lambda x: x[1]["total_population"], reverse=True):
    print(f"{entry_point}: {data['total_population']:,} people ({data['borough']})")

print("\nNote: This is approximate data for demonstration purposes only.")
print("In a real application, this would use actual census data from official sources.") 