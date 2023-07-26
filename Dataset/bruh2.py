import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from IPython.display import display
import numpy as np
from MEMD_all import memd
from GenerateDataset import parse_motion_file, write_motion_file
from matplotlib.backends.backend_pdf import PdfPages


def EMDAugmentation(original_data, num_generated):
    # conduct MEMD
    imf = memd(original_data)
    imf_1 = imf[0, :, :]  # IMFs corresponding to the 1st component
    imf_2 = imf[1, :, :]  # IMFs corresponding to the 2nd component
    imf_3 = imf[2, :, :]  # IMFs corresponding to the 3rd component
    imf_4 = imf[3, :, :]  # IMFs corresponding to the 3rd component
    imf_5 = imf[4, :, :]  # IMFs corresponding to the 3rd component
    imf_6 = imf[5, :, :]  # IMFs corresponding to the 3rd component

    motion_data_size = original_data.shape
    dataset = np.zeros((num_generated, motion_data_size[0], motion_data_size[1]))

    # Create datasets by perturbing IMFs
    for i in range(num_generated):
        # normal distribution perturbation
        # imf_1_prime = imf_1 * abs(np.random.normal(loc=1, scale=0.33))
        # imf_2_prime = imf_2 * abs(np.random.normal(loc=1, scale=0.33))
        # imf_3_prime = imf_3 * abs(np.random.normal(loc=1, scale=0.33))
        # imf_4_prime = imf_4 * abs(np.random.normal(loc=1, scale=0.33))
        # imf_5_prime = imf_5 * abs(np.random.normal(loc=1, scale=0.33))

        # uniform distribution perturbation
        imf_1_prime = imf_1 * np.random.uniform(0, 2)
        imf_2_prime = imf_2 * np.random.uniform(0, 2)
        imf_3_prime = imf_3 * np.random.uniform(0, 2)
        imf_4_prime = imf_4 * np.random.uniform(0, 2)
        imf_5_prime = imf_5 * np.random.uniform(0, 2)

        new_dataset = (imf_1_prime + imf_2_prime + imf_3_prime + imf_4_prime + imf_5_prime + imf_6).T
        dataset[i] = new_dataset
    
    return dataset


def MixupAugmentation(original_dataset, alpha, num_generated):
    # Create dataset through mix-up augmentation
    new_dataset = np.zeros((original_dataset.shape[0],original_dataset.shape[1], original_dataset.shape[2] ))
    for i in range(num_generated):
        # extract two random datasets
        r1 = np.random.randint(0, original_dataset.shape[0]-1)
        r2 = np.random.randint(0, original_dataset.shape[0]-1)
        si = original_dataset[r1]
        sj = original_dataset[r2]

        # slice datasets
        d = np.random.randint(0, original_dataset.shape[1])
        hi = si[:d, :]
        fi = si[d:, :]
        hj = sj[:d, :]
        fj = sj[d:, :]

        lambda_ = np.random.beta(alpha, alpha)

        mixed_past = lambda_ * hi + (1 - lambda_) * hj
        mixed_future = lambda_ * fi + (1 - lambda_) * fj
        mixed_sequence = np.concatenate([mixed_past, mixed_future], axis=0)
        new_dataset[i] = mixed_sequence
    
    return new_dataset


def plotMulvariateCurves(filename, dataset):
    num_features = dataset.shape[2]
    num_generated = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in range(num_features):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(num_generated):
                ax.plot(time, dataset[j][:, i])
                ax.plot(time, original_data[:, i], color="black")
            ax.set_xlabel("time")
            ax.set_ylabel("Kinematics values")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()


h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/gait2354/inverse_kinematics_data/subject01_walk1_ik.mot")
original_data = df.to_numpy()
time = t.to_numpy()
feature_headers = list(df.columns)

EMD_dataset = EMDAugmentation(original_data, 100)
plotMulvariateCurves("bruh.pdf", EMD_dataset)

EMD_Mixup_dataset = MixupAugmentation(EMD_dataset, 0.2, 100)
plotMulvariateCurves("bruh2.pdf", EMD_Mixup_dataset)

