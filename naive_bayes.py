import json
import sys
from nba_api.stats.static import teams
import numpy as np

SEASON = "2023-24"

def process_Input():
    def inputError():
        print(team_str)
        print("ERROR Invalid input")
        print("Usage: python naive_bayes.py <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>")
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
        
        
    if(len(sys.argv) != 4): inputError()
    location = sys.argv[1]
    team1 = sys.argv[2]
    team2 = sys.argv[3]
    if(team1 not in all_teams or team2 not in all_teams or team1 == team2 or (location != "Home" and location != "Away")): inputError()
    
    if(location == "Home"): location = 1
    elif(location == "Away"): location = 0
        
    return team1, team2, location
    
        
def get_Input(team1, team2, location):
    with open('data/nba_averages.json', 'r') as f:
        averages = json.load(f)
        
    input = []    
 
        
    team1_Averages = averages.get(team1, {}).get(SEASON)
    for k, v in team1_Averages.items():
        if k != 'point_dif' and isinstance(v, (int, float)):
            input.append(round(v, 1))

    team2_Averages = averages.get(team2, {}).get(SEASON)
    for k, v in team2_Averages.items():
        if k != 'point_dif' and isinstance(v, (int, float)):
            input.append(round(v, 1))
    input.append(location)
    return np.array(input)


def prepareInput():
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
                inputs.append(round(v, 1))
        # The label is 'WL' or 'point_dif'
        label = item['WL']
        # Add the inputs and label to arrays
        training_data.append(inputs)
        labels.append(label)

    # Convert the list of training data and labels to a NumPy array
    training_data = np.array(training_data)
    labels = np.array(labels)
    
    print("Example general stats:")
    print(training_data[0])
    print(f"({len(training_data[0])}) long")

team1, team2, location = process_Input()
input = get_Input(team1, team2, location)

print(f"input stats:\n {input}")
print(f"({len(input)}) long\n")

prepareInput()