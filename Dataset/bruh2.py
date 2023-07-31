import numpy as np
import matplotlib.pyplot as plt
from GenerateDataset import parse_motion_file, write_motion_file
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import scipy.interpolate as interpolate
from scipy import signal

def TimePointGaussianAugmentation(dataset, time, dataset_freq, sampling_freq, variance_mod, num_generated):
    
    sampling_interval = dataset_freq/sampling_freq

    # array holding standard deviations for each kinmatic feature
    std_dev = np.ptp(dataset, 0) * variance_mod
    # indices of rows to be sampled
    sample_rows = np.arange(start=0, stop=dataset.shape[0], step=sampling_interval, dtype=int)
    # list of time values of samples
    time_points = time[sample_rows]
    # array of values sampled
    points = dataset[sample_rows, :]

    new_dataset = np.zeros((num_generated, *dataset.shape))
    for k in range(num_generated):
        gaussian_points = np.zeros(points.shape)
        new_motion = np.zeros(dataset.shape)
        # for each column
        for i in range (points.shape[1]):
            # for each row
            gaussian_points[:, i] = np.random.normal(loc=0, scale=std_dev[i], size=len(time_points)) + points[:, i]
            # interpolate for each column (feature)
            cs = interpolate.CubicSpline(time_points, gaussian_points[:, i])
            # sample values for original frequency
            new_motion[:, i] = cs(time)

        # add new motion to dataset
        new_dataset[k, :, :] = new_motion
    
    return new_dataset

def plotMulvariateCurves(filename, dataset, original_data):
    num_features = dataset.shape[2]
    num_generated = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in range(num_features):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(num_generated):
                ax.plot(time, dataset[j][:, i])
                ax.plot(time, original_data[:, i], color="black")
            ax.set_xlabel("time")
            ax.set_ylabel("Kinematics values")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()


h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

def lowpass_filter(data, fs, fc, order):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=0)
    return output

smooth_data = lowpass_filter(original_data, 60, 10, 5)

bruh = TimePointGaussianAugmentation(smooth_data, time, dataset_freq=60, sampling_freq=15, variance_mod=0.05, num_generated=100)

print(bruh.shape)
smooth_bruh = np.zeros(bruh.shape)
for i in range(bruh.shape[0]):
    smooth_bruh[i] = lowpass_filter(bruh[i], 60, 10, 5)

plotMulvariateCurves("bruh.pdf", smooth_bruh, smooth_data)

