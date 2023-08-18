from vmdpy import VMD
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
from tqdm import tqdm
import scipy.interpolate as interpolate
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import minmax_scale
from scipy import signal

# apply lowpass filter
# fs = sampling rate
# fc = cutoff rate
# order = order
# axis = axis to apply filter - 0 for rows, 1 for columns
def lowpass_filter(data, fs, fc, order, axis):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=axis)
    return output

# plot perturbed curves and original curve for each feature
# exports as pdf to filename path
def plotMulvariateCurves(filename, dataset, original_data, time, feature_headers):
    num_features = dataset.shape[2]
    num_generated = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in tqdm(range(num_features)):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(num_generated):
                ax.plot(time, (dataset[j][:, i]))
                ax.plot(time, (original_data[:, i]), color="black")
            ax.set_xlabel("time")
            ax.set_ylabel("Kinematics values")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()

# plot the splines of each perturbed curve for each feature
# will be exported as pdf to filename path
def plotMulvariateSplines(filename, dataset, time, time_points, distortion, feature_headers):
    num_features = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in tqdm(range(num_features)):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(dataset.shape[1]):
                ax.plot(time, (dataset[i][j, :]))
                ax.scatter(time_points[i][j, :], distortion[i][j, :], color='red')
            ax.set_xlabel("time")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()

# convert a signal to the 360 degree convention
def convertTo360Degrees(signal):
    signal = [math.radians(deg) for deg in signal]
    signal = [math.cos(rad) for rad in signal]
    signal = [math.acos(rad) for rad in signal]
    signal = [math.degrees(rad) for rad in signal]
    return signal

# parse the OpenSim .mot file
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

# calculate the average velocities using sliding window
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

# apply magnitude warp to 1d time series data array
# data = kinematics values
# time = time values
# n_points = number of knots
# m = multiplier to apply when scaling, usually the range of the current feature

def MagnitudeWarpRow(data, time, n_points, m):
    step = int(len(time) / n_points)
    time_indices = np.arange(0, len(time), step, dtype=int)
    time_points = time[time_indices]

    mav = movingAverageVelocities(data)

    distortion = np.zeros(len(time_points))
    for i in range(len(distortion)):
        distortion[i] = np.random.normal(loc=0, scale= m*mav[(time_indices[i])])

    new_data = np.zeros(len(data))
    cs = interpolate.CubicSpline(time_points, distortion)
    spline = cs(time)
    new_data = data + spline

    return new_data, spline, time_points, distortion

# Warp time domain
def TimeWarpRow(data, time, n_points):

    step = int(len(time)/(n_points))
    time_indices = (np.arange(0, len(time), step, dtype=int))
    time_points = time[time_indices]
    distortion = np.random.normal(loc=1, scale=0.05, size=len(time_indices))

    new_data = np.zeros(len(data))
    cs = interpolate.CubicSpline(time_points, distortion)
    distorted_time = ((np.cumsum(cs(time)))/len(time))*np.max(time)
    new_data = np.interp(time, distorted_time, data)
    return new_data

# data = 2d array of original data
# num_generated = number of curves to be generated
def VMDAugment(data, num_generated):

    alpha = 2000       # moderate bandwidth constraint  
    tau = 0            # noise-tolerance (no strict fidelity enforcement)  
    K = 5              # modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7
    num_knots = 8

    # create matrices to store spline, distortion, and knot time point data for graphing
    splines = np.zeros((data.shape[1], num_generated, data.shape[0]))
    distortions = np.zeros((data.shape[1], num_generated, num_knots + math.ceil((7510%num_knots)/num_knots)))
    time_points = np.zeros((data.shape[1], num_generated, num_knots + math.ceil((7510%num_knots)/num_knots)))
    
    # create matrices to store imfs and the imf ranges
    imfs = np.zeros((data.shape[1], K, data.shape[0]))
    imf_ranges = np.zeros((data.shape[1], K))

    # for each feature, conduct VMD and store the imfs and imf ranges into the matrices
    # this is so that VMD only has to be conducted once, saving time
    for i in tqdm(range(data.shape[1])):
        u, u_hat, omega = VMD(data[:, i], alpha, tau, K, DC, init, tol)
        imfs[i, :, :] = u
        imf_ranges[i, :] = np.ptp(u, axis=1)

    new_dataset = np.zeros((num_generated, *data.shape))
    # create new datasets
    for i in tqdm(range(num_generated)):
        new_data = np.zeros((data.shape[0], data.shape[1]))
        for j in range(data.shape[1]):
            u_new = imfs[j].copy()
            # conduct magnitude warping on the residual
            u_new[0, :], s, t, d = MagnitudeWarpRow(u_new[0,:], time, num_knots, imf_ranges[j, 0])
            # store splines, time_points, and distortions for graphing
            splines[j, i, :] = s
            time_points[j, i, :] = t
            distortions[j, i, :] = d
            new_signal = np.sum(u_new,axis=0)
            new_data[:, j] = new_signal
        new_dataset[i, :, :] = new_data

    return new_dataset, splines, time_points, distortions

# parse the .mot file
h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
# original_data is a 2d array: the rows represent the time, each column is a different feature
original_data = df.to_numpy()
# only examine the first 6 features
original_data = original_data[:, :6]
# time is an array to hold time values
time = t.to_numpy()
# list of feature names to be used when plotting
feature_headers = list(df.columns)

# apply 10 hz lowpass filter and degree conversion
smooth_data = lowpass_filter(original_data, 200, 10, 5, 0)
smooth_data[:, 2] = convertTo360Degrees(smooth_data[:, 2])

# conduct VMD augmentation, currently generating 20 curves
bruh, splines, time_points, distortions = VMDAugment(smooth_data, 20)
# plot
plotMulvariateCurves("VMD.pdf", bruh, smooth_data, time, feature_headers)
plotMulvariateSplines("Splines.pdf", splines, time, time_points, distortions, feature_headers)
