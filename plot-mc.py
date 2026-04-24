import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('multi-cam-leo.csv')

# Calculate FOV and check for errors
df['sensor_width'] = df['cam_x_resolution'] * df['cam_x_pixel_pitch']
df['fov_rad'] = 2 * np.arctan(df['sensor_width'] / (2 * df['cam_focal_length']))
df['fov_deg'] = np.degrees(df['fov_rad'])

# Round FOV to avoid floating point grouping issues
df['fov_deg_rounded'] = df['fov_deg'].round(2)

# Check position_distance_error_m
print("Nulls in position_distance_error_m:", df['position_distance_error_m'].isnull().sum())

# If position_distance_error_m is null, calculate it
if df['position_distance_error_m'].isnull().all():
    df['calc_error'] = np.sqrt(
        (df['true_pos_x'] - df['out_pos_x'])**2 +
        (df['true_pos_y'] - df['out_pos_y'])**2 +
        (df['true_pos_z'] - df['out_pos_z'])**2
    )
    metric = 'calc_error'
else:
    metric = 'position_distance_error_m'

# Grouping and calculating stats
# Group and calculate Mean, Std, and 95th Percentile
stats = df.groupby('fov_deg_rounded')[metric].agg(['mean', 'std', lambda x: x.quantile(0.99)]).reset_index()
stats.columns = ['FOV (deg)', 'Mean Error', 'Std Dev', '95th Percentile']

print(stats)

