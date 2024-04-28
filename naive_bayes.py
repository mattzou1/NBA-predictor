import json
import sys
from nba_api.stats.static import teams
import numpy as np

def getInput():
    def inputError():
        print(team_str)
        print("ERROR Invalid input")
        print("Usage: python naive_bayes.py <team1 abbreviation> <team2 abbreviation>")
        print("(input in all caps)")
        sys.exit()
    
    #Get a dict of all teams
    teams_dict = teams.get_teams()

    #dictionary mapping team names to team abbreviations
    team_id_to_name = {team['full_name']: team['abbreviation'] for team in teams_dict}
    
    all_teams = set()
    team_str = "Abbreviation | Full Name\n"
    #create a string to show abbeviations for each team and create a set of team abbreviations
    for team_name, team_abbr in sorted(team_id_to_name.items()):
        team_str += f"        {team_abbr}:   {team_name}\n"
        all_teams.add(team_abbr)
        
        
    if(len(sys.argv) != 3): inputError()
    team1 = sys.argv[1]
    team2 = sys.argv[2]
    if(team1 not in all_teams or team2 not in all_teams or team1 == team2): inputError()
        
    
        
        

def prepareInput():
    # Load the training data
    with open('data/data/nba_training_data.json', 'r') as f:
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
                inputs.append(round(v, 1))
        # The label is 'WL' or 'point_dif'
        label = item['WL']
        # Add the inputs and label to arrays
        training_data.append(inputs)
        labels.append(label)

    # Convert the list of training data and labels to a NumPy array
    training_data = np.array(training_data)
    labels = np.array(labels)

    print(labels)
    print(training_data)

getInput()