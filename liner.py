import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from skyfield.api import load,EarthSatellite,Time
from datetime import datetime
import pickle

# Read the CSV data into a pandas DataFrame
df = pd.read_csv('TLEs.csv')

# Extract specific fields into lists
sat_names = df['id'].tolist()
sat_number = df['sat_id'].tolist()
tle_lines1 = df['tle_line1'].tolist()
tle_lines2 = df['tle_line2'].tolist()

def propagateSatellite(sat_data):
    """Use Skyfield to propagate satellite and observer states.

    Parameters
    ----------
    sat_data : tuple
        Tuple containing TLE line 1, TLE line 2, Julian start time, Julian end time, time step, and satellite number.
    """
    tleLine1, tleLine2, times, satnum = sat_data

    secperday = np.float64(86400)
    dtsec = np.float64(1)
    ts = load.timescale()
    
    try:
        satellite = EarthSatellite(tleLine1, tleLine2, ts=ts)
    except ValueError as e:
        print(f"Error creating satellite ({satnum}) from TLE data: {e}")
        return []
    
    #check if TLE is older than 20 days
    t = ts.utc(2024, 10, 13, 0, 0, 0) #change this to todays date
    days = abs(t - satellite.epoch)   
    if days>=20:
        print(f"outdated by {days} for satellite {sat_number}")
        return "outdated"
   
    
    dtday = dtsec / secperday
    dtx2 = 2 * dtsec
    
    # Initialize lists to store positions, velocities, and Julian dates
    positions = []
    velocities = []
    j_d = []

    for jd in times:
        try:
            t = ts.ut1_jd(jd)
            tplusdt = ts.ut1_jd(jd + dtday)
            tminusdt = ts.ut1_jd(jd - dtday)

            sat_pos = satellite.at(t).position.km
            satpdt_pos = satellite.at(tplusdt).position.km
            satmdt_pos = satellite.at(tminusdt).position.km

            velocity = (satpdt_pos - satmdt_pos) / dtx2

            positions.append((sat_pos[0], sat_pos[1], sat_pos[2]))
            velocities.append((velocity[0], velocity[1], velocity[2]))
            j_d.append(jd)
        except Exception as e:
            print(f"Error during propagation at JD {jd} for satellite {satnum}: {e}")
            continue  # Skip this time step and continue with the next
    
    return positions,velocities


def create_cubic_spline(jd_points, positions, index, sat_number):
    """Creates a CubicSpline for the specified position index."""
    
    # Check for finite values in the specified position
    valid_positions = [pos[index] for pos in positions if np.isfinite(pos[index])]
    valid_jd_points = [jd for jd, pos in zip(jd_points, positions) if np.isfinite(pos[index])]

    if not valid_positions:
        print(f"No finite values found in positions for satellite {sat_number} at index {index}.")
        return None  # Return None if no valid positions

    # Create the CubicSpline with valid data
    return CubicSpline(valid_jd_points, valid_positions)


def process_satellite(position,velocity, jdstart, jdend, n1, sat_number):
    """Processes the satellite data to calculate maximum differences in positions."""
    
    #time 
    jd_points = np.linspace(jdstart, jdend, n1)
   
    #creating equation
    cubics_pos = [create_cubic_spline(jd_points, position, i, sat_number) for i in range(3)]
    cubics_vel = [create_cubic_spline(jd_points, velocity, i, sat_number) for i in range(3)]

    if any(cubic is None for cubic in cubics_pos):
        print("Interpolation Failed (position)")
        return None
    
    if any(cubic is None for cubic in cubics_vel):
        print("Interpolation Failed (velocity)")
        return None

    return cubics_pos,cubics_vel


def process_satellites_sequentially():
    jdstart = np.float64(2460596.5000000)
    jdend = np.float64(2460597.5000000)
    print(f"Start Julian Date: {jdstart}")

    # Create an array of 1000 evenly spaced points between jdstart and jdend
    num_points = 1000
    jd_points = np.linspace(jdstart, jdend, num_points)
   
    print(f"Processing {len(sat_names)} satellites sequentially")
    all_cubics_pos = {}
    all_cubics_vel = {}

    for i in range(len(sat_number)):
        sat_data = (tle_lines1[i], tle_lines2[i], jd_points, sat_number[i])
        positions,velocities = propagateSatellite(sat_data)
        #print(positions,velocities)
        cubics_pos,cubics_vel=process_satellite(positions,velocities, jdstart, jdend, num_points,sat_number[i])
        
        if cubics_pos is not None and cubics_vel is not None:
            all_cubics_pos[sat_number[i]] = cubics_pos
            all_cubics_vel[sat_number[i]] = cubics_vel
        else:
            print(f"Skipping satellite {sat_number[i]} due to interpolation failure.")
            

    # Save all cubics_pos and cubics_vel to a single file
    with open("all_satellite_cubics.pkl", "wb") as f:
        pickle.dump((all_cubics_pos, all_cubics_vel), f)

    print("All satellite cubics saved to file.")
        

if __name__ == '__main__':
    process_satellites_sequentially()

