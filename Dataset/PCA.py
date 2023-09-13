import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Processing import parse_motion_file

def conduct_pca(data, n_components):
    pca = PCA(n_components)
    new_data = pca.fit_transform(data)
    return pca, new_data

h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()

nums = np.arange(39)
var_ratio = []
for i in nums:
    pca, new_data = conduct_pca(original_data, i)
    var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

plt.show()


