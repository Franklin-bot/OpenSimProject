import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd
from itertools import islice
from IPython.display import display
import numpy as np
from PyEMD import EMD2d
from MEMD_all import memd
from GenerateDataset import parse_motion_file, write_motion_file
from matplotlib.backends.backend_pdf import PdfPages


h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

imf = memd(original_data)
imf_1 = imf[0, :, :]  # IMFs corresponding to the 1st component
imf_2 = imf[1, :, :]  # IMFs corresponding to the 2nd component
imf_3 = imf[2, :, :]  # IMFs corresponding to the 3rd component
imf_4 = imf[3, :, :]  # IMFs corresponding to the 3rd component
imf_5 = imf[4, :, :]  # IMFs corresponding to the 3rd component
imf_6 = imf[5, :, :]  # IMFs corresponding to the 3rd component

num_generated = 100
new_dataset_size = original_data.shape
dataset_of_datasets = np.zeros((num_generated, new_dataset_size[0], new_dataset_size[1]))
num_features = dataset_of_datasets.shape[2]

for i in range(num_generated):
    # imf_1_prime = imf_1 * abs(np.random.normal(loc=1, scale=0.33))
    # imf_2_prime = imf_2 * abs(np.random.normal(loc=1, scale=0.33))
    # imf_3_prime = imf_3 * abs(np.random.normal(loc=1, scale=0.33))
    # imf_4_prime = imf_4 * abs(np.random.normal(loc=1, scale=0.33))
    # imf_5_prime = imf_5 * abs(np.random.normal(loc=1, scale=0.33))

    imf_1_prime = imf_1 * np.random.uniform(0, 2)
    imf_2_prime = imf_2 * np.random.uniform(0, 2)
    imf_3_prime = imf_3 * np.random.uniform(0, 2)
    imf_4_prime = imf_4 * np.random.uniform(0, 2)
    imf_5_prime = imf_5 * np.random.uniform(0, 2)

    new_dataset = (imf_1_prime + imf_2_prime + imf_3_prime + imf_4_prime + imf_5_prime + imf_6).T
    dataset_of_datasets[i] = new_dataset

with PdfPages("No_Time_Augmentation_Uniform_Perturbation_1.pdf") as pdf:
    for i in range(num_features):
        fig, ax = plt.subplots(figsize=(8, 6))
        for j in range(dataset_of_datasets.shape[0]):
            ax.plot(time, dataset_of_datasets[j][:, i])
            ax.plot(time, original_data[:, i], color="black")
        ax.set_xlabel("time")
        ax.set_ylabel("Kinematics values")
        ax.set_title(feature_headers[i])
        pdf.savefig(fig)
        plt.close(fig)
        
    plt.tight_layout()



