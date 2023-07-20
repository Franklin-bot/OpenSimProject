# Augment a given motion through a Gaussian distribution
#   Generate gaussian noise for each dimension of the motion data
#   Add noise for each dimension at each time step

import pandas as pd
import numpy as np
from itertools import islice
from IPython.display import display

# def generate_motion_set(original_file_path, gaussian_parameters, output_size):

def parse_motion_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 10)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=8, engine="python", dtype=np.float64)
    # display(dataframe)

    return header, dataframe



def write_motion_file(header, dataframe, new_file_name):
    # insert header
    new_file_path = "/Users/FranklinZhao/OpenSimProject/Dataset/Augmented/" + new_file_name + ".mot"
    new_file = open(new_file_path, "a")
    new_file.write(header)
    new_file.close()

    # insert dataframe
    dataframe.to_csv(path_or_buf=new_file_path, sep='\t', header=True, mode="a", index=False)



delete_file = open("/Users/FranklinZhao/OpenSimProject/Dataset/Augmented/bruh.mot", "w")
delete_file.close()
h, df = parse_motion_file(input_file_path="/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
write_motion_file(h, df, "bruh")

