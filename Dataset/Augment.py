# Augment a given motion through a Gaussian distribution
#   Generate gaussian noise for each dimension of the motion data
#   Add noise for each dimension at each time step

import pandas as pd
import numpy as np
from IPython.display import display

# def generate_motion_set(original_file_path, gaussian_parameters, output_size):
    


original_file_path = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot"
dataframe = pd.read_csv(filepath_or_buffer=original_file_path, skiprows=11, skipinitialspace=True, sep='\t', header=None, engine="python", dtype=np.float64)
display(dataframe)


