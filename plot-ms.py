import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Presentation (match plot-r / plot-c / plot-d)
AXIS_SIDE_IN = 7.5
PLOT_LINEWIDTH = 2.5
PLOT_MARKERSIZE = 7
CAPSIZE = 6
FS_TITLE = 18
FS_LABEL = 16
FS_TICK = 14
FS_LEGEND = 14
FS_LEGEND_TITLE = 15
X_RIGHT_PAD = 1.04  # extend x-axis past last range bin (multiplicative)
M_TO_KM = 1e-3

# Load data
df = pd.read_csv('big-night.csv')

# True range magnitude (same as other multi-cam plots)
df['true_pos_mag'] = np.sqrt(df['true_pos_x']**2 + df['true_pos_y']**2 + df['true_pos_z']**2)

nominals = (
    6.8e6, 6.9e6, 7.1e6, 7.3e6, 7.7e6, 7.9e6, 8.6e6, 9.4e6, 11e6, 12.6e6,
    16e6, 20e6, 25e6, 30e6, 35e6, 50e6, 100e6, 150e6, 200e6
)


def find_nearest(val):
    return min(nominals, key=lambda x: abs(x - val))


df['dist_nominal'] = df['true_pos_mag'].apply(find_nearest)

# (column, output png, title, y-axis label)
METRIC_SPECS = (
    ('delta_centroid', 'multi_cam_mean_std_c.png', 'Δ Centroid vs Range (km)', 'Δc (px)'),
    ('delta_r_apparent', 'multi_cam_mean_std_r.png', 'Δ Apparent Radius vs Range (km)', 'Δr (px)'),
    ('delta_distance', 'multi_cam_mean_std_d.png', 'Δ Distance vs Range (km)', 'Δp (km)'),
)


def remove_outliers(series):
    q1 = series.quantile(0.05)
    q3 = series.quantile(0.95)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]


stats_list = []
for metric, _, _, _ in METRIC_SPECS:
    metric_stats = []
    for dist in sorted(df['dist_nominal'].unique()):
        subset = df[df['dist_nominal'] == dist][metric].dropna()
        if not subset.empty:
            clean_subset = remove_outliers(subset)
            if not clean_subset.empty:
                metric_stats.append({
                    'dist': dist,
                    'mean': clean_subset.mean(),
                    'std': clean_subset.std(),
                })
                print(f'{metric} for dist {dist}: mean {clean_subset.mean()} std {clean_subset.std()}')
    stats_list.append(pd.DataFrame(metric_stats))

legend_kw = dict(fontsize=FS_LEGEND, title_fontsize=FS_LEGEND_TITLE, handlelength=3.2)

for (metric_col, out_path, title, y_label), stats_df in zip(METRIC_SPECS, stats_list):
    if stats_df.empty:
        continue

    x_km = stats_df['dist'].to_numpy() * M_TO_KM
    if metric_col == 'delta_distance':
        y_plot = stats_df['mean'].to_numpy() * M_TO_KM
        y_err = stats_df['std'].to_numpy() * M_TO_KM
    else:
        y_plot = stats_df['mean'].to_numpy()
        y_err = stats_df['std'].to_numpy()

    fig, ax = plt.subplots(
        figsize=(AXIS_SIDE_IN * 1.38, AXIS_SIDE_IN * 1.18),
        layout='constrained',
    )

    ax.errorbar(
        x_km,
        y_plot,
        yerr=y_err,
        fmt='-o',
        capsize=CAPSIZE,
        capthick=PLOT_LINEWIDTH * 0.85,
        elinewidth=PLOT_LINEWIDTH * 0.85,
        ecolor='black',
        color='tab:blue',
        markersize=PLOT_MARKERSIZE,
        linewidth=PLOT_LINEWIDTH,
        markeredgewidth=0.8,
        label='Mean ± σ',
    )

    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel('Range (km)', fontsize=FS_LABEL)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    if metric_col == 'delta_distance':
        ax.yaxis.set_major_locator(MaxNLocator(nbins=14, min_n_ticks=10))
    ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=1.0)
    dmax_km = float(stats_df['dist'].max()) * M_TO_KM * X_RIGHT_PAD
    ax.set_xlim(0, dmax_km)
    _x_fmt = ScalarFormatter(useMathText=True)
    _x_fmt.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(_x_fmt)
    ax.legend(loc='best', **legend_kw)
    ax.set_box_aspect(1)

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)

saved = [METRIC_SPECS[i][1] for i, s in enumerate(stats_list) if not s.empty]
print('Plots saved:', ', '.join(saved) if saved else '(no data)')
