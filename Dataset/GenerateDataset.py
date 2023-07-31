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

# augment motion data by sampling gaussian random points and interpolating
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

# plot each feature of multivariate timeseries
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


# lowpass filter
def lowpass_filter(data, fs, fc, order, axis):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=axis)
    return output


# extract original data
h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

# smooth original data
smooth_data = lowpass_filter(original_data, 60, 10, 5, 0)
# create augmented motions
bruh = TimePointGaussianAugmentation(smooth_data, time, dataset_freq=60, sampling_freq=15, variance_mod=0.05, num_generated=100)
# smooth augmented motions
smooth_bruh = lowpass_filter(bruh, 60, 10, 5, 1)
# plot
plotMulvariateCurves("bruh.pdf", smooth_bruh, smooth_data)
