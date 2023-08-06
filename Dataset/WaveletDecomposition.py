import pywt
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
from scipy import signal
from scipy import fft
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing as p

# lowpass filter
def lowpass_filter(data, fs, fc, order, axis):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=axis)
    return output

def plotMulvariateCurves(filename, dataset, original_data):
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



def WaveletAugmentation(data, num_generated):

    wavelet = 'db4'
    levels = 10
    variance_mod = 1

    ranges = np.ptp(data, axis=0)

    perturbation_factor = ranges*variance_mod
    new_dataset = np.zeros((num_generated, *data.shape))
    for k in tqdm(range(num_generated)):
        new_data = np.zeros(data.shape)
        for i in range (data.shape[1]):
            signal = data[:, i]
            coeffs = pywt.wavedec(signal, wavelet, level = levels, mode='symmetric')
            coeffs[0] += np.random.normal(loc = 0, scale= perturbation_factor[i]*1, size=len(coeffs[0]))
            coeffs[1] += np.random.normal(loc = 0, scale= perturbation_factor[i]*0.25, size=len(coeffs[1]))
            coeffs[2] += np.random.normal(loc = 0, scale= perturbation_factor[i]*0.1, size=len(coeffs[2]))
            coeffs[3] += np.random.normal(loc = 0, scale= perturbation_factor[i]*0.05, size=len(coeffs[3]))
            # # coeffs[4] += np.random.normal(loc = 0, scale= perturbation_factor[i]*0.01, size=len(coeffs[4]))
            # coeffs[8] += np.random.normal(loc = 0, scale= perturbation_factor[i]*2, size=len(coeffs[8]))
            perturbed_signal = pywt.waverec(coeffs, wavelet)
            perturbed_signal = perturbed_signal[:len(signal)]
            new_data[:, i] = perturbed_signal
        # add new motion to dataset
        new_dataset[k, :, :] = new_data
    return new_dataset


h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

# smooth original data
smooth_data = lowpass_filter(original_data, 200, 10, 5, 0)
# create augmented motions
bruh = WaveletAugmentation(smooth_data, 20)
# smooth augmented motions
smooth_bruh = lowpass_filter(bruh, 200, 10, 5, 1)
# plot
plotMulvariateCurves("dataset.pdf", smooth_bruh, smooth_data)

print((bruh[1, :, 0])[:20]-(bruh[2, :, 0])[:20])