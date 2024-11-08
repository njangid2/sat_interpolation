import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(file_path):
    """Load CSV file using optimized parameters."""
    dtype_dict = {
        'Position X (km)': np.float32,
        'Position Y (km)': np.float32,
        'Position Z (km)': np.float32,
        'Velocity X (km/s)': np.float32,
        'Velocity Y (km/s)': np.float32,
        'Velocity Z (km/s)': np.float32,
        'Satellite Number': np.int32
    }
    return pd.read_csv(file_path, dtype=dtype_dict, usecols=dtype_dict.keys())
df1_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_5000.csv")
df2_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_10000.csv")
df3_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_15000.csv")
df4_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_20000.csv")
df5_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_25000.csv")
df6_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_30000.csv")
df7_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_parallel_35000.csv")


dg_results = load_and_prepare_data("/Volumes/New Volume/Campuscluster/all_satellites_positions_velocities_sequential.csv")
de_results = pd.read_csv("/Volumes/New Volume/Campuscluster/TLEs.csv")

# Extract specific fields into lists
sat_epoch = de_results['epoch'].tolist()
sat_number = de_results['sat_id'].tolist()

   
# Create a dictionary to map satellite numbers to epoch times
sat_epoch_dict = dict(zip(sat_number, sat_epoch))


tle1 = de_results['tle_line1'].tolist()
tle2 = de_results['tle_line2'].tolist()
sat_tle = {sat_num: (tle1_line, tle2_line) for sat_num, tle1_line, tle2_line in zip(sat_number, tle1, tle2)}



from skyfield.api import load,EarthSatellite,Time
from datetime import datetime

def create_cubic_spline(jd_points, positions, index):
    """Creates a CubicSpline for the specified position index."""
    valid_indices = np.isfinite(positions[:, index])
    if not np.any(valid_indices):
        return None
    return CubicSpline(jd_points[valid_indices], positions[valid_indices, index])

def process_satellite(sat_number, df_results, dg_results, jdstart, jdend, n, n1, line1, line2):
    """Processes the satellite data to calculate maximum differences in positions."""
    sat_data = df_results[df_results['Satellite Number'] == sat_number]
    sat_data_dg = dg_results[dg_results['Satellite Number'] == sat_number]
    if sat_number == 12945913:
        return None
   
    ts = load.timescale()
    t = ts.utc(2024, 10, 13, 0, 0, 0)
    satellite = EarthSatellite(line1, line2,'', ts)
    days = abs(t - satellite.epoch)
    epoch_time=satellite.epoch
   
    if days>=20:
#        print(f"outdated by {days} for satellite {sat_number}")
        return "outdated"
   
    if sat_data_dg.empty:
#        print(f"Epoch {epoch_time}: No data for satellite {sat_number}.")
        return None  # Return None if no data found


    #uncomment to get histogram for position or velocity 
    
    #for position
    positions = sat_data[['Position X (km)', 'Position Y (km)', 'Position Z (km)']].values
    positions11 = sat_data_dg[['Position X (km)', 'Position Y (km)', 'Position Z (km)']].values
    
    #for velocity
    #positions = sat_data[['Velocity X (km/s)', 'Velocity X (km/s)', 'Velocity X (km/s)']].values
    #positions11 = sat_data_dg[['Velocity X (km/s)', 'Velocity X (km/s)', 'Velocity X (km/s)']].values
    
    
    cheb_nodes = np.cos((2 * np.arange(n,dtype=np.float64) - 1) * np.pi / (2 * n))
    t_cheb = (cheb_nodes + 1) * (jdend - jdstart) / 2 + jdstart
    jd_points = np.linspace(jdstart, jdend, n1)
   
    cubics = [create_cubic_spline(jd_points, positions11, i) for i in range(3)]
    if any(cubic is None for cubic in cubics):
        return None

    interpolated = np.array([cubic(t_cheb) for cubic in cubics])
    diff = np.abs(positions[:, :3].T - interpolated) * 1e6  # Convert to m
    max_diffs = np.max(diff[:, 500:4500], axis=1)

    # Check if any max_diff is greater than 10^10 mm
    if np.any(max_diffs > 1e7):
#        print(f"Epoch {epoch_time}: Satellite {sat_number} excluded due to difference > 10^10 mm: {max_diffs}")
        return None

    for i, max_diff in enumerate(max_diffs):
        if max_diff > 20:
            print(f"Epoch {epoch_time}: Max diff_{['x', 'y', 'z'][i]} for satellite {sat_number} is greater than 10000: {max_diff:.2f} m")

    return tuple(max_diffs)




def process_satellite_combined(datasets, dg_results, sat_tle):
    logging.info("Processing combined satellite data")
   
    jdstart, jdend = np.float64(2460596.5), np.float64(2460597.5)
    n, n1 = 5000, 1000
   
    all_results = []
    total_excluded = 0
    total_outdated = 0
   
    # Process each dataset
    for df_results in datasets:
        unique_sat_numbers = df_results['Satellite Number'].unique()
       
        for sat in tqdm(unique_sat_numbers, desc='Processing Satellites'):
            tle1, tle2 = sat_tle.get(sat, ("Unknown", "Unknown"))
            result = process_satellite(sat, df_results, dg_results, jdstart, jdend, n, n1, tle1, tle2)
           
            if result is None:
                total_excluded += 1
            elif result == "outdated":
                total_outdated += 1
            else:
                all_results.append(result)
   
    logging.info(f"Total satellites excluded due to large differences: {total_excluded}")
    logging.info(f"Total satellites included in final results: {len(all_results)}")
    logging.info(f"Total satellites excluded due to outdated epoch: {total_outdated}")
   
    # Create combined histogram
    if all_results:
        max_diff_x, max_diff_y, max_diff_z = zip(*all_results)
       
        plt.figure(figsize=(15, 5))
       
        # Plot three subplots side by side
        titles = ['VX', 'VY', 'VZ']
        colors = ['blue', 'orange', 'green']
        data = [max_diff_x, max_diff_y, max_diff_z]
       
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.hist(data[i], bins=50, color=colors[i], alpha=0.7)
            plt.title(f'Max Velocity Difference {titles[i]}')
            plt.xlabel(f'Max diff {titles[i]} (mm/s)')
            plt.ylabel('Frequency')
            plt.grid(True)
           
       
        plt.tight_layout()
        plt.savefig('combined_histogram_max_diff_Vxyz.png', dpi=300, bbox_inches='tight')
        plt.close()
       
    
    
# Usage
datasets = [df1_results, df2_results, df3_results, df4_results, df5_results, df6_results, df7_results]
process_satellite_combined(datasets, dg_results, sat_tle)