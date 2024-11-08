# sat_interpolation
#In folder Orignal
#liner.py: This script can generate positions for all satellites.
#compare.py: Use this script to compare actual positions with interpolated positions. You will need the source data for the satellite, including both the data used to derive equations and additional data for comparison.



#to use extract data from file to get positions
#Example
with open("all_satellite_cubics.pkl", "rb") as f:
    all_cubics_pos, all_cubics_vel = pickle.load(f)

#Access the cubic spline equations for a specific satellite
sat_number = 2
cubics_pos = all_cubics_pos[sat_number]
cubics_vel = all_cubics_vel[sat_number]

t_cheb=2460596.5000000
interpolated = np.array([cubic(t_cheb) for cubic in cubics_pos])
print(interpolated)




