import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class nbaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(nbaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def load_data_and_labels():
    # Load the training data
    with open('data/nba_training_data2.json', 'r') as f:
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

def train(model, training_data, labels, num_epochs=20, learning_rate=0.0005, batch_size=32):
    torch.autograd.set_detect_anomaly(True)
    # Convert the training data and labels to PyTorch tensors
    training_data = torch.tensor(training_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Split data into training and validation sets
    val_split = 0.1  # 10% for validation
    val_size = int(val_split * len(training_data))
    X_train, X_val = training_data[:-val_size], training_data[-val_size:]
    Y_train, Y_val = labels[:-val_size], labels[-val_size:]

    print("Starting Training!")
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            # Zero the gradients
            optimizer.zero_grad()

            # Get the current batch
            batch_X_train = X_train[i:i+batch_size]
            batch_Y_train = Y_train[i:i+batch_size]

            # Forward pass
            output, _ = model(batch_X_train, model.init_hidden())
            loss = criterion(output, batch_Y_train.view(-1, 1))  # Reshape target to match input

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss for this epoch
            epoch_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / len(X_train)}")

        # Validation: Evaluate on validation set
        with torch.no_grad():
            val_output, _ = model(X_val, model.init_hidden())  # Corrected line
            val_loss = criterion(val_output, Y_val.view(-1, 1))  # Reshape target to match input
            print(f"Validation Loss: {val_loss.item()}")

    return model


training_data, labels = load_data_and_labels()
n_features = len(training_data[0])  # Number of features in your data
n_hidden = 256  # Number of hidden units
n_classes = 1  # Regression task (predicting 'point_dif')

rnn = nbaRNN(n_features, n_hidden, n_classes)
rnn = train(rnn, training_data, labels)


# Save the trained model
torch.save(rnn.state_dict(), 'nba_rnn2.pth')
print("Model saved!")
