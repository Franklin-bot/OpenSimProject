import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
from scipy import signal
from tqdm import tqdm

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

h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

signal1 = original_data[:, 0]


t, f, sxx = signal.stft(signal1, fs = 200, nfft = 512, noverlap=256, nperseg=512)
n_generated = 10

plt.figure()
for i in range(n_generated):
    perturbation = np.random.normal(loc=1, scale=0.3, size=(sxx.shape[0], sxx.shape[1]))
    # part_sxx = np.multiply(sxx[2:], perturbation)
    # sxxi = np.vstack((sxx[:2], part_sxx))
    sxxi = sxx*perturbation
    new_signal = signal.istft(sxxi, fs=200, nfft=512, noverlap=256, nperseg=512)
    
    plt.plot(time, (new_signal[1])[:len(time)])
plt.plot(time, signal1, color="black")
plt.show()

