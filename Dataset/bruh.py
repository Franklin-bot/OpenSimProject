import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import pandas as pd
from itertools import islice
from IPython.display import display

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



h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
original_data = df.to_numpy()
time = t.to_numpy()
time_range = (time[0], time[1])
n_samples = len(time)

# Step 2: Define the mean function and the squared exponential (RBF) kernel for multiple dimensions
def mean_function(x):
    return np.zeros_like(x)  # Zero mean function (GP centered at 0)

length_scale = 1.0
variance = 1.0

kernel = ConstantKernel(constant_value=variance, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))


# Step 3: Generate the GP model based on the original dataset
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_model.fit(time.reshape(-1, 1), original_data)

# Step 4: Generate new datasets using the trained GP model
n_generated_datasets = 1  # Change this value to generate more datasets
generated_datasets = []

for _ in range(n_generated_datasets):
    # Generate new X values to predict
    
    print(x_generated)

    # Predict the new datasets using the GP model
    generated_data = gp_model.predict(x_generated.reshape(-1, 1))
    generated_datasets.append(generated_data)
print(len(generated_datasets))

df2 = pd.DataFrame(generated_datasets[0], columns=list(df.columns))
display(df2)



def write_motion_file(header, dataframe, time, new_file_name):

    # insert header
    new_file_path = "/Users/FranklinZhao/OpenSimProject/Dataset/Motion/Augmented/" + new_file_name + ".mot"
    new_file = open(new_file_path, "a")
    new_file.write(header)
    new_file.close()

    # insert dataframe
    dataframe.insert(loc=0, column = 'time', value = time)
    dataframe.to_csv(path_or_buf=new_file_path, sep='\t', header=True, mode="a", index=False)

open('/Users/FranklinZhao/OpenSimProject/Dataset/Motion/Augmented/bruh.mot', 'w').close()
write_motion_file(h, df2, t, "bruh")