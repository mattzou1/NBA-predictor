"""
Takes a .pth model and predicts the point differential and outcome of a given NBA game

Authors: David Lybeck, Matthew Zou
5/2/2024
"""

import torch
import torch.nn as nn
import json
import sys
from nba_api.stats.static import teams
import os.path

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

SEASON = "2023-24"

def create_data(team1, team2):
    """
    Creates a custom data set using the given teams to use as input for the neural network

    Returns: input array
    """
    # Load the averages from the averages JSON file
    with open('data/nba_averages.json', 'r') as f:
        averages = json.load(f)
        
    with open('data/nba_training_data.json', 'r') as f:
        stats = json.load(f)

    # Initialize a list to store the training data
    input_data = {}
    
    team_averages = {f'team_average_{k}': v for k, v in averages.get(team1, {}).get(SEASON, {}).items()}
    opponent_averages = {f'opponent_average_{k}': v for k, v in averages.get(team2, {}).get(SEASON, {}).items()}
    
    game = {}
    for game in stats:
        if(game['team'] == team1): break
    
    previous = {}
    for k, v in game.items():
        if('previous' in k):
            previous[k] = v
    
    input_data = {**team_averages, **opponent_averages, **previous, 'home_game': 1}
    
    # Initialize a list to store the training data
    input_arr = []


    for k, v in input_data.items():
        input_arr.append(v)

    return input_arr
    
def run(team1, team2, fileName):
    input_data = create_data(team1, team2)
        
    # Load the trained model
    n_features = len(input_data)  # Assuming you have access to training_data
    n_hidden = 256  # Same values as used during training
    n_classes = 1
    model = nbaRNN(n_features, n_hidden, n_classes)
    model.load_state_dict(torch.load("models/"+fileName))
    model.eval()

    # Convert the input to a PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    # Run the input through the model
    output, _ = model(input_tensor, model.init_hidden())

    # Print the predicted point difference
    if(round(output.item(), 0) > 0): return f"I predict {team1} will beat {team2} by {int (round(output.item(), 0))} points!"
    if(round(output.item(), 0) < 0): return f"I predict {team2} will beat {team1} by {int (round(abs(output.item()), 0))} points!"
    if(round(output.item(), 0) == 0):
        if(output.item() > 0): return f"I predict {team1} will beat {team2} in a VERY close match"
        if(output.item() < 0): return f"I predict {team2} will beat {team1} in a VERY close match"
        if(output.item() == 0): return f"I predict a TIE"