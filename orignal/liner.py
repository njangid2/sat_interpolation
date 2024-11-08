import pandas as pd
from skyfield.api import load, EarthSatellite
import numpy as np

# Read the CSV data into a pandas DataFrame
df = pd.read_csv('/home/njangid2/Sat_checker/TLEs.csv')

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
    
    all_data = []
    for i, (pos, vel, t) in enumerate(zip(positions, velocities, j_d)):
        all_data.append({
            'Satellite Number': satnum,
            'Julian Date': t,
            'Position X (km)': pos[0],
            'Position Y (km)': pos[1],
            'Position Z (km)': pos[2],
            'Velocity X (km/s)': vel[0],
            'Velocity Y (km/s)': vel[1],
            'Velocity Z (km/s)': vel[2]
        })
    
    return all_data

def process_satellites_sequentially():
    jdstart = np.float64(2460596.5000000)
    jdend = np.float64(2460597.5000000)
    print(f"Start Julian Date: {jdstart}")

    # Create an array of 2000 evenly spaced points between jdstart and jdend
    num_points = 1000
    t_cheb = np.linspace(jdstart, jdend, num_points)
   
    print(f"Processing {len(sat_names)} satellites sequentially")

    all_data = []
    for i in range(len(sat_names)):
        sat_data = (tle_lines1[i], tle_lines2[i], t_cheb, sat_number[i])
        result = propagateSatellite(sat_data)
        all_data.extend(result)
        print(f"Processed satellite {i+1}/{len(sat_names)}")

    # Create a DataFrame from the collected data
    df_all = pd.DataFrame(all_data)

    # Save to a single CSV file with specified precision
    try:
        df_all.to_csv('all_satellites_positions_velocities_sequential.csv', index=False, float_format='%.15f')
        print("Saved all data to all_satellites_positions_velocities_sequential.csv")
    except IOError as e:
        print(f"Error saving the file: {e}")

if __name__ == '__main__':
    process_satellites_sequentially()