import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


txt_file = "G:/Edgar Dobra/GPC Samples/02.19.2024_OT2_Test2_Air.txt"
RI_calibration = "./RI Calibration curve.txt"
graph_number = 7  # a number in which location the respective graph is, from 1

# PS values
mass_of_PS = 100  # in mg
PS_percent_of_functionalization = 95.5  # give it in x%

# G3 values
G3_used = 46  # in microliters
concentration_of_G3 = 2  # in mg/mL

# Molar masses
molar_mass_of_G3 = 884.54
PS_Mn = 4900

# Graph setup
x_lim = [8, 19]
y_lim = [0, 1.2]
number_of_peaks = 2
peaks = []  # only enter peaks if you want to specify them manually, otherwise leave list empty
peak_wideness_range = [100, 300]  # set default to [100, 800], only change if peaks are really wide or narrow

# Initial variables defined, do not change!!
integrals = []
graphs = []
Mn = []
initial_guess_for_peak_fit = [1, 0.1]  # initial guess for peak amplitude & standard deviation, set default to [1, 0.1]

# Reading in data and initial formatting
file = open(txt_file, "r")
content = file.readlines()
file.close()
list_content = []
for row in content:
    list_content.append(row.strip("\n").split("\t"))
data_array = np.array(list_content)


file_RI = open(RI_calibration, "r")
content_RI = file_RI.readlines()
file_RI.close()
list_content_RI = []
for row in content_RI:
    list_content_RI.append(row.strip("\n").split("\t"))
data_array_RI = np.array(list_content_RI)


# Get maximum between bounds
def max_of_y_within_range(x_array, y_array):
    mask = np.logical_and(x_array > x_lim[0], x_array < x_lim[1])
    return np.max(y_array[mask])


def min_of_y_within_range(x_array, y_array):
    mask = np.logical_and(x_array > 8, x_array < 18)
    return np.min(y_array[mask])


# Normalize and get data
def get_data(data_index=0):
    get_x_data = np.delete(data_array[:, data_index * 2], [0, 1], 0)
    get_x_data = get_x_data.astype(float)
    get_y_data = np.delete(data_array[:, data_index * 2 + 1], [0, 1], 0)
    get_y_data = get_y_data.astype(float)
    min_y = min_of_y_within_range(get_x_data, get_y_data)
    for value in range(len(get_y_data)):
        get_y_data[value] = get_y_data[value] - min_y
    max_y = max_of_y_within_range(get_x_data, get_y_data)
    for value in range(len(get_y_data)):
        get_y_data[value] = get_y_data[value] / max_y

    return get_x_data, get_y_data


# Get data between new bounds
def get_new_data(x_array, y_array):
    mask = np.logical_and(x_array > x_lim[0], x_array < x_lim[1])
    return x_array[mask], y_array[mask]


# Gaussian distribution
def gaussian(x_val, amplitude, mean, stddev):
    return amplitude * np.exp(-(x_val - mean) ** 2 / (2 * stddev ** 2))


# Find nth largest value
def n_largest(arr, n):
    if n < 1 or n > len(arr):
        return None
    sorted_arr = np.sort(arr)[::-1]
    return np.where(arr == sorted_arr[n - 1])[0][0]


# Reorder Gaussian peaks based on where they are on the x-axis
def sort_gaussian_peaks(arr):
    arr = arr.tolist()
    largest_indices = [max(enumerate(subarray), key=lambda x_val: x_val[1])[0] for subarray in arr]
    sorted_arr = [subarray for _, subarray in sorted(zip(largest_indices, arr))]
    return np.array(sorted_arr)


x_a = np.delete(data_array_RI[:, 1], [0, 1], 0)
x_a = x_a.astype(float)
x_IR = np.delete(data_array_RI[:, 0], [0, 1], 0)
x_IR = x_IR.astype(float)
x_a = 10 ** x_a
_, x_a = get_new_data(x_IR, x_a)

for graph in range(int(len(data_array[0, :])/2)):
    #print(graph)
    #y_Gauss_New = np.empty((0,0))
    fit_check_0 = 1e10
    Gauss_Peaks_New = []
    y_Gauss_New = []

    # Get x and y values for original graph and create original graph
    x, y = get_new_data(get_data(graph)[0], get_data(graph)[1])
    plt.plot(x, y, label="Original Graph " + str(graph+1))


    # Manual or automatic peak detection
    x_indices = []
    y_indices = []
    if len(peaks) == 0:
        indices, _ = find_peaks(y, distance=200, width=50)
        x_indices = [x[j] for j in indices]
        y_indices = [y[j] for j in indices]
    else:
        x_indices = []
        y_indices = []
        for i in range(len(peaks)):
            index = np.where(peaks[i] == x)
            x_indices.append(float(x[index]))
            y_indices.append(float(y[index]))
    #plt.plot(x_indices, y_indices, "go")
    #Gauss_peaks = []
    #Gauss_peaks_array = np.array(Gauss_peaks)

    #y_Gauss = np.empty((0, 0))
    #y_new = np.sbtract(y_new, y_fit)
    # Generate Gaussian fits, find best fit
    for val in range(peak_wideness_range[0], peak_wideness_range[1]):
        y_new = y
        y_Gauss = np.array([0 for j in y_new])
        Gauss_peaks = []
        fit_check = 0
        for i in range(number_of_peaks):
            peak_x = int(np.where(x == x_indices[n_largest(y_indices, i + 1)])[0])
            y_p = [0] * len(y_new)

            for j in range(peak_x - val, peak_x + val):
                y_p[j] = y_new[j]

            initial_guess = [initial_guess_for_peak_fit[0], x_indices[n_largest(y_indices, i + 1)],
                             initial_guess_for_peak_fit[1]]  # initial values for amplitude, mean, and standard deviation
            params, covariance = curve_fit(gaussian, x, y_p, p0=initial_guess)

            amplitude_fit, mean_fit, stddev_fit = params

            y_fit = gaussian(x, amplitude_fit, mean_fit, stddev_fit)

            Gauss_peaks.append(y_fit)
            Gauss_peaks_array = np.array(Gauss_peaks)

            y_Gauss = np.add(y_Gauss, y_fit)
            y_new = np.subtract(y_new, y_fit)

        fit_check = np.sum(np.abs(y_new))
        if fit_check < fit_check_0:
            fit_check_0 = fit_check
            y_Gauss_New = y_Gauss
            Gauss_Peaks_New = Gauss_peaks_array

    # Sort peaks based on x range
    Gauss_Peak_1 = sort_gaussian_peaks(Gauss_Peaks_New)[0]
    graphs.append(Gauss_Peak_1.tolist())
    #print(x_a)
    #print("GP " + str(graph_number) + ": " + str(Gauss_Peak_1))
    Mn.append(round(np.sum(np.multiply(Gauss_Peak_1, x_a))/np.sum(Gauss_Peak_1)))

    # Plot Gaussian fit
    plt.plot(x, y_Gauss_New, "--", label='Gaussian Fit ' + str(graph+1))
    ax = plt.gca()
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.xlabel("Time")
    plt.ylabel("Normalization")
    plt.legend()

    # Final print



graphs = np.array(graphs)

print("Mn: " + str(Mn))
plt.show()

# Calculations
G3_moles_0 = (concentration_of_G3 * G3_used * 1e-3) / molar_mass_of_G3
PS_moles = mass_of_PS * PS_percent_of_functionalization / 100 / PS_Mn




'''
# Set up plotting
ax = plt.gca()
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.xlabel("Time")
plt.ylabel("Normalization")
plt.legend()


# Final print

plt.show()
'''