import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import matplotlib.font_manager

# Configuration
txt_file = "G:/Edgar Dobra/GPC Samples/Fall 2024/11.15.2024_GB_GRAFT_PS-b-2PLA.txt"
rt_lim = [7, 19]  # Retention time limits
y_lim = [-0.02, 1]
number_of_peaks = 4
peaks = []  # [11.74, 12.23, 13.02, 17.53] for manual entry
peak_names = ["PS-b-2PLA-b-PS", "PS-b-2PLA", "PS-b", "PS"]
peak_wideness_range = [100, 400]  # set to [100, 800] for default
baseline_method = 'quadratic'
baseline_ranges = [[9.0, 10], [16, 16.3], [18.3, 19]]

# Calibration and MW conversion
RI_calibration = "G:/Edgar Dobra/GPC Samples/Calibration Curves/RI Calibration Curve 2024 September.txt"
mw_x_lim = [1e3, 1e8]

# Font settings
matplotlib.rcParams['font.family'] = 'Avenir Next LT Pro'
matplotlib.rcParams['font.size'] = 18

# Data Loading
data_array = np.loadtxt(txt_file, delimiter='\t', skiprows=2)

# Load calibration data
data_array_RI = np.loadtxt(RI_calibration, delimiter='\t', skiprows=2)
retention_time_calib = data_array_RI[:, 0].astype(float)
log_mw_calib = data_array_RI[:, 1].astype(float)
f_log_mw = interp1d(retention_time_calib, log_mw_calib, kind='linear', fill_value='extrapolate')

# Utility Functions
def max_of_y_within_range(x_array, y_array, x_min, x_max):
    mask = (x_array > x_min) & (x_array < x_max)
    return np.max(y_array[mask]) if np.any(mask) else 1.0

def extract_data(data_index=0):
    x = data_array[:, data_index * 2]
    y = data_array[:, data_index * 2 + 1]
    x = x.astype(float)
    y = y.astype(float)
    max_y = max_of_y_within_range(x, y, rt_lim[0], rt_lim[1])
    return x, y / max_y

def format_data(x, y):
    mask = (x >= rt_lim[0]) & (x <= rt_lim[1])
    return x[mask], y[mask]

# Baseline Correction
def baseline_correction(x, y, method='quadratic'):
    ref_points = []
    required_ranges = {'flat': 1, 'linear': 2, 'quadratic': 3}.get(method, 1)
    if len(baseline_ranges) != required_ranges:
        raise ValueError(f"{method} method requires {required_ranges} baseline ranges")
    for bl_range in baseline_ranges:
        mask = (x >= bl_range[0]) & (x <= bl_range[1])
        if np.sum(mask) == 0:
            raise ValueError(f"No data points in baseline range {bl_range}")
        x_ref, y_ref = np.mean(x[mask]), np.mean(y[mask])
        ref_points.append((x_ref, y_ref))
    x_vals = [p[0] for p in ref_points]
    y_vals = [p[1] for p in ref_points]
    if method == 'flat':
        baseline = np.full_like(y, np.mean(y_vals))
    elif method == 'linear':
        coeffs = np.polyfit(x_vals, y_vals, 1)
        baseline = np.polyval(coeffs, x)
    elif method == 'quadratic':
        coeffs = np.polyfit(x_vals, y_vals, 2)
        baseline = np.polyval(coeffs, x)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    return y - baseline, baseline

# Main Processing
x_raw, y_raw = extract_data(0)
x_rt, y_formatted = format_data(x_raw, y_raw)
y_corrected, baseline = baseline_correction(x_rt, y_formatted, method=baseline_method)

# Convert retention time to molecular weight for plotting
x_mw = 10 ** f_log_mw(x_rt)

# Peak Detection
if len(peaks) == 0:
    indices, _ = find_peaks(y_corrected, distance=200, width=50)
    x_peaks_rt = x_rt[indices]
    y_peaks = y_corrected[indices]
    if len(y_peaks) >= number_of_peaks:
        top_indices = np.argsort(y_peaks)[-number_of_peaks:][::-1]
        x_peaks_rt = x_peaks_rt[top_indices]
        y_peaks = y_peaks[top_indices]
    else:
        print(f"Found only {len(y_peaks)} peaks. Adjust parameters.")
        exit()
else:
    x_peaks_rt, y_peaks = [], []
    for peak in peaks:
        idx = np.argmin(np.abs(x_rt - peak))
        x_peaks_rt.append(x_rt[idx])
        y_peaks.append(y_corrected[idx])

# Gaussian Fitting
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


best_fit = None
best_residual = np.inf
best_width = peak_wideness_range[0]
best_fit_params = []  # Added to store Gaussian parameters

for width in range(peak_wideness_range[0], peak_wideness_range[1]):
    y_current = y_corrected.copy()
    gaussians = []
    params_list = []  # Store parameters for each Gaussian
    try:
        for i in range(number_of_peaks):
            mu = x_peaks_rt[i]
            idx = np.argmin(np.abs(x_rt - mu))
            start, end = max(0, idx - width), min(len(x_rt), idx + width)
            initial_guess = [y_peaks[i], mu, 0.1]
            params, _ = curve_fit(gaussian, x_rt[start:end], y_current[start:end], p0=initial_guess)
            y_fit = gaussian(x_rt, *params)
            gaussians.append(y_fit)
            params_list.append(params)  # Track parameters
            y_current -= y_fit
        residual = np.sum(np.abs(y_current))
        if residual < best_residual:
            best_residual = residual
            best_fit = np.array(gaussians)
            best_fit_params = params_list  # Save best parameters
            best_width = width
    except (RuntimeError, IndexError):
        continue

# Sort peaks by molecular weight (x-axis) from highest to lowest
if best_fit is not None and len(best_fit_params) == number_of_peaks:
    # Extract mu parameters and calculate molecular weights
    mus = [params[1] for params in best_fit_params]
    mn_values = 10 ** f_log_mw(mus)  # Convert to actual MW

    # Get sorting order for descending molecular weight
    sorted_indices = np.argsort(mn_values)[::-1]

    # Reorder all components based on sorted indices
    best_fit = best_fit[sorted_indices]
    best_fit_params = [best_fit_params[i] for i in sorted_indices]
    integrals = [trapezoid(g, x_rt) for g in best_fit]
    percentages = [(area / sum(integrals)) * 100 for area in integrals]

# Visualization
plt.figure()
plt.subplots_adjust(bottom=0.19, left=0.19)
plt.plot(x_mw, y_corrected, label='Original Data', linewidth=2, color='#ef476f')

if best_fit is not None:
    colors = ['#ffd166', '#06d6a0', '#118ab2', '#073b4c']
    for i, (fit, pct) in enumerate(zip(best_fit, percentages)):
        plt.plot(x_mw, fit, color=colors[i % 4], label=f'{peak_names[i]}: {pct:.1f}%')

plt.xscale('log')
plt.xlim(mw_x_lim)
plt.ylim(y_lim)
plt.xlabel("Molecular weight (g/mol)", fontstyle='italic', fontweight='demi')
plt.ylabel("Normalized Response", fontstyle='italic', fontweight='demi')
plt.legend()
plt.grid(False)
plt.tight_layout()
# plt.savefig("Figure1.png", dpi=1200)
plt.show()