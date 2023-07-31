import numpy as np

# Create a 2D array (matrix)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Calculate the sum along the columns (axis=1)
sum_along_columns = np.sum(data, axis=1)
print(sum_along_columns)  # Output: [ 6 15 24]
