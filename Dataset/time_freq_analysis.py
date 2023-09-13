import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pywt
from PyEMD import EMD
from scipy import signal
from Processing import parse_motion_file, lowpass_filter
from vmdpy import VMD
import emd as emd2

def stft_decomp(data):
    sampling_freq = 200
    window_size = 1024
    overlap = 1023

    s, f, t, im = plt.specgram(data, Fs=sampling_freq, NFFT=window_size, noverlap=overlap, cmap='rainbow')
    plt.ylim(0, 20)

def wavelet_decomp(data):
    fs = 200
    wavelet = 'cmor'
    frequencies = np.array(np.arange(start=0.01, stop=20.01, step=0.05)) / fs
    scales = pywt.frequency2scale(wavelet, frequencies)
    print(len(scales))

    coef, freqs = pywt.cwt(data, scales, wavelet)
    
    # graph
    plt.imshow(abs(coef), extent=[0, 7510, 20, 0] , aspect='auto', cmap='rainbow')
    plt.gca().invert_yaxis()
    plt.xticks(ticks=np.arange(0, coef.shape[1], step=1000), labels=np.arange(0, coef.shape[1], step=1000)/fs)

def emd_decomp(data):
    imf = emd2.sift.sift(data)
    
    # instantaneous frequency, phase and amplitude
    IP, IF, IA = emd2.spectra.frequency_transform(imf, 200, 'hilbert')

    return imf

def graphEMDModeSpectrums(data):
    imf = emd_decomp(data)

    for i in range(imf.shape[1]):
        fft_result = np.fft.fft(imf[:, i])
        frequencies = np.fft.fftfreq(len(fft_result), 1/200)
        plt.figure()
        plt.title(f'IMF {i+1} Spectrum')
        plt.plot(frequencies, np.abs(fft_result))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim(0, frequencies.max())
        plt.show()



    



def vmd_decomp(data):
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0            # noise-tolerance (no strict fidelity enforcement)  
    K = 8              # modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly
    tol = 1e-7

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u, u_hat, omega, K
    
def graphVMDModeSpectrums(data):
    u, u_hat, omega, K = vmd_decomp(data)

    for i in range(K):
        magnitude_spectrum = np.abs(u_hat[:, i])
        
        freqs = np.fft.fftshift(np.fft.fftfreq(len(magnitude_spectrum)))

        # Create a new figure for each mode
        plt.figure()
        
        # Plot the magnitude spectrum
        plt.plot(freqs, magnitude_spectrum)
        
        # Set labels and title
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude Spectrum')
        plt.title(f'Mode {i + 1} Spectrum')
        plt.xlim(0, freqs.max())
        
        # Show the plot
        plt.show()












h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
feature = original_data[:, 0]
t = np.array(t)

graphEMDModeSpectrums(feature)
#graphVMDModeSpectrums(feature)
