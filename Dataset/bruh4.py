from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
from scipy import signal
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import fft
import pywt
from PyEMD import EMD
from scipy.ndimage import gaussian_filter1d

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



def FFTAugmentation(data, num_generated):

    new_dataset = np.zeros((num_generated, *data.shape))
    for k in tqdm(range(num_generated)):
        new_data = np.zeros(data.shape)
        for i in range (data.shape[1]):
            signal = data[:, i]
            y = fft.fft(signal)
            y *= np.random.normal(loc=1, scale=0.25, size=len(y))
            new_signal = fft.ifft(y)
            new_signal = RowEmd(np.abs(new_signal))
            new_data[:, i] = new_signal
        # add new motion to dataset
        new_dataset[k, :, :] = new_data
    return new_dataset


def WaveletAugmentation(data, num_generated, perturbation_level, variance_mod):

    wavelet = 'db4'
    levels = 10

    ranges = np.ptp(data, axis=0)

    perturbation_factor = ranges*variance_mod
    new_dataset = np.zeros((num_generated, *data.shape))
    for k in range(num_generated):
        new_data = np.zeros(data.shape)
        for i in range (data.shape[1]):
            signal = data[:, i]
            coeffs = pywt.wavedec(signal, wavelet, level = levels, mode='symmetric')
            coeffs[perturbation_level] += np.random.normal(loc = 0, scale= perturbation_factor[i], size=len(coeffs[perturbation_level]))
            coeffs[perturbation_level] = RowFFt(coeffs[perturbation_level])
            perturbed_signal = pywt.waverec(coeffs, wavelet)
            perturbed_signal = perturbed_signal[:len(signal)]
            new_data[:, i] = perturbed_signal
        # add new motion to dataset
        new_dataset[k, :, :] = new_data
    return new_dataset

def EMDAugmentation(data, num_generated):

    new_dataset = np.zeros((num_generated, *data.shape))
    for k in tqdm(range(num_generated)):
        new_data = np.zeros(data.shape)
        for i in range (data.shape[1]):
            signal = data[:, i]
            emd = EMD.EMD()
            imfs = emd(signal)
            perturbedImfs = np.array(imfs)*np.random.normal(loc=1, scale=0.25, size=imfs.shape)
            new_signal = sum(perturbedImfs)
            new_data[:, i] = new_signal
        # add new motion to dataset
        new_dataset[k, :, :] = new_data
    return new_dataset

def RowFFt(row):
    range = np.ptp(row)
    y = fft.fft(row)
    y *= np.random.normal(loc=1, scale=0.25, size = len(y))
    y += np.random.normal(loc=0, scale=range*10, size=len(y))
    new_signal = fft.ifft(y).real
    return new_signal

def RowEmd(row):
    range = np.ptp(row)
    emd = EMD.EMD()
    imfs = emd(row)
    perturbedImfs = np.array(imfs[1:])*np.random.uniform(0, 2, size=(imfs.shape[0]-1, imfs.shape[1]))
    perturbedImfs = np.array(imfs)+ np.random.normal(loc=0, scale=0.05*range, size=imfs.shape)

    new_signal = sum(perturbedImfs) + imfs[1,:]

    return new_signal

def RowWavelet(row):
    range = np.ptp(row)
    wavelet = 'db4'
    levels = 10
    coeffs = pywt.wavedec(row, wavelet, level = levels, mode='symmetric', axis=0)
    # coeffs[0] *= np.random.normal(loc=1, scale=0.15, size=[len(coeffs[0])])
    coeffs[0] += np.random.normal(loc=0, scale=0.5*range, size=[len(coeffs[0])])
    coeffs[1] += np.random.normal(loc=0, scale=0.25*range, size=[len(coeffs[1])])
    coeffs[2] += np.random.normal(loc=0, scale=0.1*range, size=[len(coeffs[2])])
    new_signal = pywt.waverec(coeffs, wavelet)
    return new_signal


def CombinedAugmentation(data, num_generated):
    new_dataset = np.zeros((num_generated, *data.shape))
    for k in tqdm(range(num_generated)):
        new_data = np.zeros(data.shape)
        for i in range (data.shape[1]):
            signal = data[:, i]
            new_signal = (gaussian_filter1d((RowEmd(signal)), 50))
            new_data[:, i] = new_signal
        # add new motion to dataset
        new_dataset[k, :, :] = new_data
    return new_dataset






h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

# smooth original data
# smooth_data = lowpass_filter(original_data, 200, 10, 5, 0)
smooth_data = gaussian_filter1d(original_data, 50, axis=0)
# # create augmented motions

# bruh = CombinedAugmentation(smooth_data, 10)

# # smooth_bruh = lowpass_filter(bruh, 200, 10, 5, 1)
# plotMulvariateCurves("bruh.pdf", bruh, smooth_data)

# def combination(data, num_generated):
#     new_dataset = np.zeros((num_generated, *data.shape))
#     for j in tqdm(range(num_generated)):
#         new_data = np.zeros(data.shape)
#         for i in range(data.shape[1]):
#             signal = data[:, i]
#             print(len(signal))
#             new_signal = (gaussian_filter1d(RowWavelet(RowEmd(signal)), 50))
#             new_data[:, i] = new_signal
#         # add new motion to dataset
#         new_dataset[j, :, :] = new_data
#     return new_dataset

# bruh = combination(smooth_data, 10)
# plotMulvariateCurves("bruh.pdf", bruh, smooth_data)




num_generated = 10
new_data = np.zeros((num_generated, original_data.shape[0]))
for i in range(num_generated):
    smooth_data = gaussian_filter1d(original_data[:, 2], 50)
    plt.plot(time, (gaussian_filter1d(RowEmd(original_data[:,2]), 50)))
    plt.plot(time, smooth_data, color="black")
plt.show()



