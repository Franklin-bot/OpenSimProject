# Augment a given motion through a Gaussian distribution
#   Generate gaussian noise for each dimension of the motion data
#   Add noise for each dimension at each time step


import pandas as pd
import numpy as np
from itertools import islice
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate as interpolate
from scipy import signal
from tqdm import tqdm
import pywt
from scipy import fft
from PyEMD import EMD
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter1d

# read IMU data file and parse it into a header and dataframe
def parse_IMU_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 4)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=4, engine="python", dtype="str")
    # display(dataframe)
    display(dataframe)
    
# read experimental .mot file and parse it into a header and dataframe
def parse_motion_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 10)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=8, engine="python", dtype=np.float64)
    # display(dataframe)

    time = dataframe['time']
    dataframe = dataframe.drop(labels='time', axis=1)

    return header, dataframe, time

# create a .mot file using a header and modified dataframe
def write_motion_file(header, dataframe, time, new_file_name):

    # insert header
    new_file_path = "/Users/FranklinZhao/OpenSimProject/Dataset/Motion/Augmented/" + new_file_name + ".mot"
    new_file = open(new_file_path, "a")
    new_file.write(header)
    new_file.close()

    # insert dataframe
    dataframe.insert(loc=0, column = 'time', value = time)
    dataframe.to_csv(path_or_buf=new_file_path, sep='\t', header=True, mode="a", index=False)

def movingAverageVelocities(data):

    window_size = 401
    window_half = int(window_size/2)
    velocities = abs(np.gradient(data))
    average_velocities = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        index = np.clip([i-window_half, i+window_half], 0, data.shape[0]-1)
        average_velocities[i] = np.mean((velocities[index[0] : index[1]]))
    
    # normalize the average velocities, (currently, normalized between 2% and 10% to be multiplied by range)
    average_velocities = minmax_scale(average_velocities, feature_range=(0.02,0.1))
    return average_velocities

# plot each feature of multivariate timeseries
def plotMulvariateCurves(filename, dataset, original_data, time, feature_headers):
    num_features = dataset.shape[2]
    num_generated = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in tqdm(range(num_features)):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(num_generated):
                ax.plot(time[:2000], (dataset[j][:, i])[:2000])
                ax.plot(time[:2000], (original_data[:, i])[:2000], color="black")
            ax.set_xlabel("time")
            ax.set_ylabel("Kinematics values")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()

# lowpass filter
def lowpass_filter(data, fs, fc, order, axis):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=axis)
    return output

# graph the distribution of synthetic signals at time point
def plot_random_timepoint_distribution(dataset, num):
    fig, axs = plt.subplots(nrows=num)
    for i in range(num):
        rand_timepoint = np.random.randint(0, dataset.shape[1])
        rand_feature = np.random.randint(0, dataset.shape[2])
        array = dataset[:, rand_timepoint, rand_feature]
        axs[i].hist(array, density=True, bins=30)
        axs[i].set_title(f'Timepoint {rand_timepoint}, Feature {rand_feature}')
    plt.tight_layout()
    plt.show()

def plot_timepoint_distribution(dataset, original_data, feature_num):

    rand_timepoint = np.random.randint(0, dataset.shape[1])
    data = dataset[:, rand_timepoint, feature_num]
    print(data.shape)
    plt.hist(data, bins=50, density=True, edgecolor='black')
    plt.axvline(x=original_data[rand_timepoint, feature_num], color='red', linestyle='dashed', linewidth=2)
    plt.show()

# translate signal vertically
def MagnitudeOffset(data, std_dev):
    n_features = data.shape[1]
    offsets = np.random.normal(loc=0, scale=std_dev, size=n_features)
    return data + offsets

def MagnitudeWarp(data, time, n_points):
    step = int(len(time) / n_points)
    time_indices = np.arange(0, len(time), step, dtype=int)
    time_points = time[time_indices]

    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        distortion = np.random.normal(loc=1, scale=0.2, size=len(time_indices))

        cs = interpolate.CubicSpline(time_points, distortion)
        new_data[:, i] = data[:, i]*cs(time)

    return new_data

# Warp time domain
def TimeWarp(data, time, n_points):

    step = int(len(time)/(n_points))
    time_indices = (np.arange(0, len(time), step, dtype=int))
    time_points = time[time_indices]
    distortion = np.random.normal(loc=1, scale=0.05, size=len(time_indices))


    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        cs = interpolate.CubicSpline(time_points, distortion)
        distorted_time = ((np.cumsum(cs(time)))/len(time))*np.max(time)
        new_data[:, i] = np.interp(time, distorted_time, data[:, i])
    return new_data

# conduct wavelet decomposition and perturb
def WaveletDecomp(data):

    wavelet = 'db4'
    levels = 10

    ranges = np.ptp(data, axis=0)
    new_data = np.zeros(data.shape)
    for i in range (data.shape[1]):
        signal = data[:, i]
        coeffs = pywt.wavedec(signal, wavelet, level = levels, mode='symmetric')
        coeffs[0] += np.random.normal(loc=0, scale=0.5*ranges[i], size=[len(coeffs[0])])
        coeffs[1] += np.random.normal(loc=0, scale=0.25*ranges[i], size=[len(coeffs[1])])
        coeffs[2] += np.random.normal(loc=0, scale=0.1*ranges[i], size=[len(coeffs[2])])
        perturbed_signal = pywt.waverec(coeffs, wavelet)
        perturbed_signal = perturbed_signal[:len(signal)]
        new_data[:, i] = perturbed_signal
    return new_data

# conduct fourier decomposition and perturb
def FftDecomp(data):
    ranges = np.ptp(data, axis=0)

    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        signal = data[:, i]
        y = fft.fft(signal)
        y *= np.random.normal(loc=1, scale=0.1, size = len(y))
        # y += np.random.normal(loc=0, scale=ranges[i]*10, size=len(y))
        perturbed_signal = fft.ifft(y).real
        new_data[:, i] = perturbed_signal
    return new_data

# conduct EMD decomposition
def EMDDecomp(data):

    imfs_list = []
    imfs_ranges_list = []
    mavs = np.zeros((data.shape))
    for j in tqdm(range(data.shape[1])):
        signal = data[:, j]
        emd = EMD.EMD()
        imfs = emd(signal)
        imfs_list.append(imfs)
        imfs_ranges_list.append(np.ptp(imfs, axis = 1))
        mavs[:, j] = movingAverageVelocities(data[:, j]) 
    
    print(imfs_ranges_list[1])
    return imfs_list, imfs_ranges_list, mavs

# def MagnitudeWarpRow(data, time, n_points, m):
#     step = int(len(time) / n_points)
#     time_indices = np.arange(0, len(time), step, dtype=int)
#     time_points = time[time_indices]

#     new_data = np.zeros(len(data))
#     distortion = np.random.normal(loc=1, scale=0.2*m, size=len(time_indices))

#     cs = interpolate.CubicSpline(time_points, distortion)
#     new_data = data*cs(time)

#     return new_data

def MagnitudeWarpRow(data, time, n_points, m, mav):

    time_indices = np.linspace(start=0, stop=len(time)-1, num=n_points, dtype=int, endpoint=True)
    time_points = time[time_indices]

    distortion = np.random.normal(loc=0, scale=m, size=len(time_indices))
    distortion *= mav[time_indices]

    new_data = np.zeros(len(data))
    cs = interpolate.CubicSpline(time_points, distortion)
    spline = cs(time)
    new_data = data + spline

    return new_data

# perturb EMD
def EMDAugment(data, imfs_list, imf_ranges_list, mavs):

    num_knots = 8

    new_data = np.zeros(data.shape)
    for i in range(len(imfs_list)):
        imfs_new = imfs_list[i]
        imfs_new[0, :] = MagnitudeWarpRow(imfs_new[0,:], time, num_knots, (imf_ranges_list[i])[0], mavs[:, i])
        new_data[:, i] = sum(imfs_new)
    return new_data

def generateNewDataset(data, time, n_generated):

    ranges = np.ptp(data, axis=0)
    imfs_list, imf_ranges_list, mavs = EMDDecomp(data)
        

    new_dataset = np.zeros((n_generated, *data.shape))
    for i in tqdm(range(n_generated)):
        # new_dataset[i, :, :] = WaveletDecomp(TimeWarp(data, time, 2))
        new_dataset[i, :, :] = (EMDAugment(data, imfs_list, imf_ranges_list, mavs))
    return new_dataset



h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
original_data = original_data[:, :6]
time = t.to_numpy()
feature_headers = list(df.columns)

# smooth_data = lowpass_filter(original_data, 200, 10, 5, 1)
bruh = generateNewDataset(original_data, time, 20)
plotMulvariateCurves("EmdTimeWarp.pdf", bruh, original_data, time, feature_headers)