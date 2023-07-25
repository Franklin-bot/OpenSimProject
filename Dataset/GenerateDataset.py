# Augment a given motion through a Gaussian distribution
#   Generate gaussian noise for each dimension of the motion data
#   Add noise for each dimension at each time step


import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import islice
from IPython.display import display



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



def create_augmented_motion_dataset(input_file_path): #, output_size
    
    h, df, t = parse_motion_file(input_file_path)
    num_features = len(df.columns)
    num_timesteps = len(df.index)

    # create covariance matrix (numpy array)
    cov = df.cov()

    # mean of zero
    noise_mean = np.zeros(num_features)

    # generate noise
    noise = np.random.multivariate_normal(noise_mean, cov, size=num_timesteps)
    noise = pd.DataFrame(noise, columns=list(df.columns), dtype=np.float64)
    display(noise)

    df = df.add(noise)
    # display(df)
    
    write_motion_file(h, df, t, "bruh")

    return

def parse_IMU_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 4)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=4, engine="python", dtype="str")
    # display(dataframe)
    display(dataframe)

open('/Users/FranklinZhao/OpenSimProject/Dataset/Motion/Augmented/bruh.mot', 'w').close()
create_augmented_motion_dataset("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")