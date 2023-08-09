import numpy as np
import pandas as pd
from GenerateDataset import parse_motion_file, plotMulvariateCurves, lowpass_filter
import scipy.interpolate as interpolate


def MagnitudeOffset(data, std_dev):
    n_features = data.shape[1]
    offsets = np.random.normal(loc=0, scale=std_dev, size=n_features)
    return data + offsets

def MagnitudeWarp(data, time, n_points):

    step = int(len(time)/(n_points))
    time_indices = (np.arange(0, len(time), step, dtype=int))
    time_points = time[time_indices]
    distortion = np.random.normal(loc=1, scale=0.3, size=len(time_indices))

    cs = interpolate.CubicSpline(time_points, distortion)

    distortion_matrix = np.zeros(data.shape)
    for i in range(data.shape[1]):
        distortion_matrix[:, i] = cs(time)

    return np.multiply(data, distortion_matrix)
    # return new_data

def TimeWarp(data, time, n_points):

    step = int(len(time)/(n_points))
    time_indices = (np.arange(0, len(time), step, dtype=int))
    time_points = time[time_indices]
    distortion = np.random.normal(loc=1, scale=0.05, size=len(time_indices))

    cs = interpolate.CubicSpline(time_points, distortion)
    distorted_time = ((np.cumsum(cs(time)))/len(time))*np.max(time)

    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(time, distorted_time, data[:, i])
    return new_data

def generateNewDataset(data, time, n_generated):
    new_dataset = np.zeros((n_generated, *data.shape))
    for i in range(n_generated):
        new_dataset[i, :, :] = MagnitudeWarp((TimeWarp(data, time, 2)), time, 2)
    return new_dataset



h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

bruh = generateNewDataset(original_data, time, 10)
plotMulvariateCurves("bruh.pdf", bruh, original_data, time, feature_headers)

