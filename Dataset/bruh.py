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

# def MagnitudeWarpRow(data, time, n_points, m, mav):

#     time_indices = np.linspace(start=0, stop=len(time)-1, num=n_points, dtype=int, endpoint=True)
#     time_points = time[time_indices]

#     distortion = np.random.normal(loc=0, scale=m, size=len(time_indices))
#     distortion *= mav[time_indices]

#     new_data = np.zeros(len(data))
#     cs = interpolate.CubicSpline(time_points, distortion)
#     spline = cs(time)
#     new_data = data + spline

#     return new_data, spline, time_points, distortion

start_time = 0
end_time = 3

time = np.linspace(start=start_time, stop=end_time, num=200)

time_points = np.linspace(start=start_time, stop=end_time, num=20)
distortion = np.random.normal(loc=0, scale=0.2, size=len(time_points))

cs = interpolate.CubicSpline(time_points, distortion)
spline = cs(time)

w = np.fft.fft(spline)
freqs = np.fft.fftfreq(len(spline))

for coef,freq in zip(w,freqs):
    if coef:
        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,f=freq))


