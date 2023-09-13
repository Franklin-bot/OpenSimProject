import numpy as np
import matplotlib.pyplot as plt


Fs = 1000  # Sampling frequency (Hz)
T = 1     # Duration of the signal (seconds)
t = np.linspace(0, T, int(Fs * T), endpoint=False)  # Time array

# Create a sine wave signal
frequency = 5  # Frequency of the sine wave (Hz)
signal = np.sin(2 * np.pi * frequency * t)

# Compute the Fourier Transform
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(fft_result), 1/Fs)

# Create a logarithmic plot
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum (Log Scale)')
plt.grid(True)
plt.show()
