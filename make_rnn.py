import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, 1)

    def forward(self, input, hidden):
        # Ensure hidden state has the same batch size as input
        hidden = hidden.expand(input.size(0), -1)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden


    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


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


training_data, labels = load_data_and_labels()
n_features = len(training_data[0])  # Number of features in your data
n_hidden = 64  # Number of hidden units
n_classes = 1  # Regression task (predicting 'point_dif')

rnn = SimpleRNN(n_features, n_hidden, n_classes)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

# Split data into training and validation sets
val_split = 0.1  # 10% for validation
val_size = int(val_split * len(training_data))
X_train, X_val = training_data[:-val_size], training_data[-val_size:]
Y_train, Y_val = labels[:-val_size], labels[-val_size:]



# Training loop
num_epochs = 20
log_interval = 100
batch_size = 32

for epoch in range(num_epochs):
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = batch_start + batch_size
        batch_inputs = torch.tensor(X_train[batch_start:batch_end], dtype=torch.float32)
        batch_labels = torch.tensor(Y_train[batch_start:batch_end], dtype=torch.float32)

        optimizer.zero_grad()
        batch_output, _ = rnn(batch_inputs, rnn.init_hidden())
        batch_loss = criterion(batch_output, batch_labels)
        batch_loss.backward()
        optimizer.step()

        if batch_start % log_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_start//batch_size+1}/{len(X_train)//batch_size}], Loss: {batch_loss.item()}")

    # Validation: Evaluate on validation set
    with torch.no_grad():
        val_inputs = torch.tensor(X_val, dtype=torch.float32)
        val_labels = torch.tensor(Y_val, dtype=torch.float32)
        val_output, _ = rnn(val_inputs, rnn.init_hidden())
        val_loss = criterion(val_output, val_labels)
        print(f"Validation Loss: {val_loss.item()}")
    

# Save the trained model
torch.save(rnn.state_dict(), 'nba_rnn1.pth')
print("Model saved!")
