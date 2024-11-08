# sat_interpolation

#to use extract data from file to get postions:
#Example

with open("all_satellite_cubics.pkl", "rb") as f:
    all_cubics_pos, all_cubics_vel = pickle.load(f)

# Access the cubic spline equations for a specific satellite
sat_number = 2
cubics_pos = all_cubics_pos[sat_number]
cubics_vel = all_cubics_vel[sat_number]

t_cheb=2460596.5000000
interpolated = np.array([cubic(t_cheb) for cubic in cubics_pos])
print(interpolated)
