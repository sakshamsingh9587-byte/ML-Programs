# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
# X = [Study hours, Sleep hours]
X = np.array([
    [2, 7],
    [3, 8],
    [4, 6],
    [5, 7],
    [6, 8]
])

# y = Marks
y = np.array([50, 55, 60, 65, 70])

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict marks for new data
# Example: Study = 5 hours, Sleep = 6 hours
prediction = model.predict([[5, 6]])

print("Predicted Marks:", prediction)