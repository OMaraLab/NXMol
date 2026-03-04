import pickle
import pandas as pd
import matplotlib.pyplot as plt
from evaluate.dfb import calculate_band_distance

name = 'original_hessian'
print(f'Loading data from {name}_CH_CO_OH.pickle')
data = pickle.load(open(f'{name}_CH_CO_OH.pickle', 'rb'))

ch_df = pd.Series(data['CH'])
co_df = pd.Series(data['CO'])
oh_df = pd.Series(data['OH'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
series_list = [ch_df, co_df, oh_df]
titles = ['aromatic C-H', 'carboxylic C-O', 'alcohol/phenol O-H']
y_values = [2980, 3080, 1710, 1780, 3200, 3550]
for i, series in enumerate(series_list):
        # Determine the band for the current series
        band_lower = min(y_values[i * 2], y_values[i * 2 + 1])
        band_upper = max(y_values[i * 2], y_values[i * 2 + 1])

        # Calculate distances for all values in the series
        distances = series.apply(lambda x: calculate_band_distance(x, band_lower, band_upper))

        # Calculate the average distance
        average_distance = distances.mean()

        print(f"mean of series {titles[i]}: {series.mean():.2f}")
        print(f"std of series {titles[i]}: {series.std():.2f}")
        print(f"{titles[i]}: Average distance from band [{band_lower:.2f}, {band_upper:.2f}] = {average_distance:.4f}")


x_ranges = [[1400, 3400], [700, 1900], [400, 3800]]
for i, ax in enumerate(axes):
    current_series = series_list[i]
    current_title = titles[i]

    # Plot the histogram
    ax.hist(current_series, bins=150, alpha=0.7)
    ax.set_title(current_title)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_ranges[i][0], x_ranges[i][1])

    # Add the two dotted vertical lines for the current histogram
    # y_values are grouped: y_values[0], y_values[1] for the first histogram, etc.
    line1_y = y_values[i * 2]
    line2_y = y_values[i * 2 + 1]

    ax.axvline(line1_y, color='red', linestyle=':', linewidth=2, label=f'Line 1 at {line1_y:.2f}')
    ax.axvline(line2_y, color='red', linestyle=':', linewidth=2, label=f'Line 2 at {line2_y:.2f}')
    ax.axvline(current_series.mean(), color='blue', linestyle=':', linewidth=2, label='Mean')
    ax.legend()

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
plt.savefig(f'{name}_histograms.svg')
