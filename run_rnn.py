import torch
import torch.nn as nn
import json
import sys
from nba_api.stats.static import teams

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

def process_Input():
    def inputError():
        print(team_str)
        print("Usage: python naive_bayes.py <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>")
        sys.exit()
    
    #Get a dict of all teams
    teams_dict = teams.get_teams()

    #dictionary mapping team names to team abbreviations
    team_name_to_abbr = {team['full_name']: team['abbreviation'] for team in teams_dict}
    
    all_teams = set()
    team_str = "Abbreviation | Full Name\n"
    
    #create a string to show abbeviations for each team and create a set of team abbreviations
    for team_name, team_abbr in sorted(team_name_to_abbr.items()):
        team_str += f"        {team_abbr}:   {team_name}\n"
        all_teams.add(team_abbr)
        
        
    if(len(sys.argv) != 4): inputError()
    location = sys.argv[1]
    team1 = sys.argv[2]
    team2 = sys.argv[3]
    if(team1 not in all_teams or team2 not in all_teams or team1 == team2 or (location != "Home" and location != "Away")): inputError()
    
    if(location == "Home"): location = 1
    elif(location == "Away"): location = 0
        
    return team1, team2, location

def create_data(team1, team2, location):
    # Load the averages from the averages JSON file
    with open('data/nba_averages.json', 'r') as f:
        averages = json.load(f)
        
    with open('data/nba_training_data2.json', 'r') as f:
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
    
    input_data = {**team_averages, **opponent_averages, **previous, 'home_game': location}
    
    # Initialize a list to store the training data
    input_arr = []


    for k, v in input_data.items():
        input_arr.append(v)

    return input_arr



team1, team2, location = process_Input()
input_data = create_data(team1, team2, location)
        
# Load the trained model
n_features = len(input_data)  # Assuming you have access to training_data
n_hidden = 256  # Same values as used during training
n_classes = 1
model = nbaRNN(n_features, n_hidden, n_classes)
model.load_state_dict(torch.load('nba_rnn1.pth'))
model.eval()

# Convert the input to a PyTorch tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

# Run the input through the model
output, _ = model(input_tensor, model.init_hidden())

# Print the predicted point difference
if(round(output.item(), 0) > 0): print(f"I predict {team1} will beat {team2} by {int (round(output.item(), 0))} points!")
if(round(output.item(), 0) < 0): print(f"I predict {team2} will beat {team1} by {int (round(abs(output.item()), 0))} points!")
if(round(output.item(), 0) == 0): print(f"I predict a TIE!")
