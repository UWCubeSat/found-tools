import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

# Presentation — bump these to enlarge fig, type, and strokes
AXIS_SIDE_IN = 7.5
PLOT_LINEWIDTH = 2.5
PLOT_MARKERSIZE = 7
LEGEND_LINEWIDTH = 3.5
FS_TITLE = 18
FS_LABEL = 16
FS_TICK = 14
FS_LEGEND = 14
FS_LEGEND_TITLE = 15
M_TO_KM = 1e-3

# Load data
df = pd.read_csv('multi-cam-26-centroid.csv')

# Preprocessing
# Calculate FOV and true distance magnitude
df['fov_x_deg'] = np.round(2 * np.degrees(np.arctan((df['cam_x_resolution'] * df['cam_x_pixel_pitch']) / (2 * df['cam_focal_length']))), 1)
df['true_pos_mag'] = np.sqrt(df['true_pos_x']**2 + df['true_pos_y']**2 + df['true_pos_z']**2)
# ‖out_pos − true_pos‖ (m); column may be NaN where positions are missing
df['abs_d_error'] = np.abs(df['delta_distance'])

# Define simulation grid distance nominals
nominals = (
    6.8e6, 6.9e6, 7.1e6, 7.3e6, 7.7e6, 7.9e6, 8.6e6, 9.4e6, 11e6, 12.6e6,
    16e6, 20e6, 25e6, 30e6, 35e6, 50e6, 100e6, 150e6, 200e6
)


def find_nearest(val):
    return min(nominals, key=lambda x: abs(x - val))


df['dist_nominal'] = df['true_pos_mag'].apply(find_nearest)

# Threshold (m ‖out − true‖); title/x-labels use km where applicable
threshold = 1000

# Aggregation: Group by FOV, Resolution, and Distance Nominal
grouped = df.groupby(['fov_x_deg', 'cam_x_resolution', 'dist_nominal'])
accuracy_df = grouped.apply(lambda x: (x['abs_d_error'] < threshold).mean() * 100).reset_index(name='pct_below_thresh')

# Mapping for colors and styles
fov_list = sorted(accuracy_df['fov_x_deg'].unique())
res_list = sorted(accuracy_df['cam_x_resolution'].unique())

_fov_colors = {5: 'tab:blue', 20: 'tab:orange', 50: 'tab:red', 85: 'tab:green'}
color_map = {f: _fov_colors[min(_fov_colors, key=lambda k: abs(k - f))] for f in fov_list}
_base_styles = ['-', '--', ':']
style_map = {r: _base_styles[min(i, len(_base_styles) - 1)] for i, r in enumerate(res_list)}
if 1024 in res_list and 2048 in res_list:
    style_map[1024] = '--'
    style_map[2048] = '-.'
elif 2048 in res_list:
    style_map[2048] = '--'
    for r in res_list:
        if r != 2048 and style_map[r] == '--':
            style_map[r] = '-.'

fig, ax = plt.subplots(
    figsize=(AXIS_SIDE_IN * 1.38, AXIS_SIDE_IN * 1.18),
    layout='constrained',
)

for fov in fov_list:
    for res in res_list:
        subset = accuracy_df[(accuracy_df['fov_x_deg'] == fov) & (accuracy_df['cam_x_resolution'] == res)]
        # Sort by distance nominal for proper line connection
        subset = subset.dropna(subset=['pct_below_thresh']).sort_values('dist_nominal')

        if not subset.empty:
            ax.plot(subset['dist_nominal'] * M_TO_KM, subset['pct_below_thresh'],
                     color=color_map[fov],
                     linestyle=style_map[res],
                     marker='o', markersize=PLOT_MARKERSIZE, linewidth=PLOT_LINEWIDTH,
                     markeredgewidth=0.8, alpha=0.8)

# Create Separate Legends
fov_legend_elements = [Line2D([0], [0], color=color_map[f], lw=LEGEND_LINEWIDTH, label=f'FOV {f}°') for f in fov_list]
res_legend_elements = [Line2D([0], [0], color='black', linestyle=style_map[r], lw=LEGEND_LINEWIDTH, label=f'{r}x{r}') for r in res_list]

legend_kw = dict(fontsize=FS_LEGEND, title_fontsize=FS_LEGEND_TITLE, handlelength=3.2)
first_legend = ax.legend(handles=fov_legend_elements, loc='upper left', title="FOV", bbox_to_anchor=(1.02, 1), **legend_kw)
ax.add_artist(first_legend)
ax.legend(handles=res_legend_elements, loc='lower left', title="Resolution", bbox_to_anchor=(1.02, 0.38), **legend_kw)

ax.set_title(f'Availability vs Range (km) (Δ distance < {threshold * M_TO_KM:g} km)', fontsize=FS_TITLE)
ax.set_xlabel('Range (km)', fontsize=FS_LABEL)
ax.set_ylabel('Availability (%)', fontsize=FS_LABEL)
ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=1.0)
ax.set_xlim(0, 20e7 * M_TO_KM)
_x_fmt = ScalarFormatter(useMathText=True)
_x_fmt.set_powerlimits((-2, 2))
ax.xaxis.set_major_formatter(_x_fmt)
ax.set_ylim(0, 100)
ax.set_box_aspect(1)

plt.savefig('multi_cam_connected_lines_d.png', bbox_inches='tight', pad_inches=0.12)
plt.close(fig)
