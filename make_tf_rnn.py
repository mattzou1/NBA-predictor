import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input

def load_data_and_labels():
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
        # The label is 'point_dif'
        label = item['point_dif']
        # Add the inputs and label to arrays
        training_data.append(inputs)
        labels.append(label)

    # Convert the list of training data and labels to a NumPy array
    training_data = np.array(training_data)
    labels = np.array(labels)
    return training_data, labels

def build_model():
    # Initialize the model
    model = Sequential()
    # Add an Input layer
    model.add(Input(shape=(None, 1)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    # Add a Dense layer for output
    model.add(Dense(1))
    
    return model

def train_model(model, training_data, labels):
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(training_data, labels, epochs=200, batch_size=32)

# Load the data and labels
training_data, labels = load_data_and_labels()

# Reshape the training data to be suitable for RNN
training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], 1))

# Build the model
model = build_model()

# Train the model
print("Starting Training!")
train_model(model, training_data, labels)

file_name = 'nba_rnn.h5'
# Save the trained model
model.save("models/"+file_name)
print(f"{file_name} saved!")