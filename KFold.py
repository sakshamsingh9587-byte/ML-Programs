# Import libraries
from sklearn.model_selection import KFold
import numpy as np

# Sample dataset
X = np.array([1,2,3,4,5,6,7,8,9,10])

# Create KFold object (5 folds)
kf = KFold(n_splits=5)

# Apply KFold
for train_index, test_index in kf.split(X):
    print("Train Data Index:", train_index)
    print("Test Data Index:", test_index)
    print()