import json
import sys
import numpy as np
from nba_api.stats.static import teams
import os.path
import tensorflow as tf


SEASON = "2023-24"

def process_Input():
    """
    Processes the input when the file is run to make sure everything is good to go

    Returns: team1 name, team2 name, whether the game is Home or away for team 1, and the desired file name
    """
    def inputError():
        """
        Error statement
        """
        print(team_str)
        print("Usage: python run_rnn.py <trained_file>.pth <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>")
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
        
        
    if(len(sys.argv) != 5): inputError()
    file_name = sys.argv[1]
    location = sys.argv[2]
    team1 = sys.argv[3]
    team2 = sys.argv[4]
    if(team1 not in all_teams or team2 not in all_teams or team1 == team2 or (location != "Home" and location != "Away")): inputError()
    if(not os.path.isfile("models/"+file_name)):
        print(f"'{file_name}' does not exist in models")
        sys.exit()
    
    if(location == "Home"): location = 1
    elif(location == "Away"): location = 0
        
    return team1, team2, location, file_name

def create_data(team1, team2, location):
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
    
    input_data = {**team_averages, **opponent_averages, **previous, 'home_game': location}
    
    # Initialize a list to store the training data
    input_arr = []


    for k, v in input_data.items():
        input_arr.append(v)

    return input_arr

team1, team2, location, fileName = process_Input()
input_data = create_data(team1, team2, location)

# Load the trained model
model = tf.keras.models.load_model("models/"+fileName)

# Convert the input to a NumPy array and reshape it
input_array = np.array(input_data).reshape(1, -1, 1)

# Run the input through the model
output = model.predict(input_array)

# Print the predicted point difference
if(round(output[0][0], 0) > 0): print(f"I predict {team1} will beat {team2} by {int (round(output[0][0], 0))} points!")
if(round(output[0][0], 0) < 0): print(f"I predict {team2} will beat {team1} by {int (round(abs(output[0][0]), 0))} points!")
if(round(output[0][0], 0) == 0):
    if(output[0][0] > 0): print(f"I predict {team1} will beat {team2} in a VERY close match")
    if(output[0][0] < 0): print(f"I predict {team2} will beat {team1} in a VERY close match")
    if(output[0][0] == 0): print(f"I predict a TIE")