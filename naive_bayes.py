import json
import numpy as np

# Load the training data
with open('data/nba_training_data.json', 'r') as f:
    data = json.load(f)

# Initialize a list to store the training data
training_data = []
labels = []

# Iterate over the data
for item in data:
    # The inputs are every numerical value except for 'point_dif'
    inputs = []
    for k, v in item.items():
        if k != 'point_dif' and isinstance(v, (int, float)):
            inputs.append(v)
    # The label is 'WL' or 'point_dif'
    label = item['WL']
    # Add the inputs and label to arrays
    training_data.append(inputs)
    labels.append(label)

# Convert the list of training data and labels to a NumPy array
training_data = np.array(training_data)
labels = np.array(labels)

