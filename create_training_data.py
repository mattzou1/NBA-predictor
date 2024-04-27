import json
import math
from collections import defaultdict

# Load the data from the original JSON file
with open('data/NBA_Data_Apr_25.json', 'r') as f:
    data = json.load(f)

# Load the averages from the averages JSON file
with open('data/nba_averages.json', 'r') as f:
    averages = json.load(f)

# Initialize a list to store the training data
training_data = []

# Iterate over the data
for season_data in data:
    for game_data in season_data:
        # Get the team, opponent, season, and win/loss
        team = game_data['TEAM_ABBREVIATION']
        opponent = game_data['MATCHUP'].split(' ')[2]
        season = game_data['SEASON']
        winlose = game_data['WL']
        point_dif = game_data['PLUS_MINUS']

        # Only consider the data from the 1985-86 season and later
        if int(season.split('-')[0]) < 1985:
            continue

        # Get the averages for the team and opponent
        team_averages = {f'team_average_{k}': v for k, v in averages.get(team, {}).get(season, {}).items()}
        opponent_averages = {f'opponent_average_{k}': v for k, v in averages.get(opponent, {}).get(season, {}).items()}

        # Determine if the game was a home game or an away game
        home_game = 1 if 'vs.' in game_data['MATCHUP'] else 0

        # Create a data point with the averages, the home game indicator, and the label
        data_point = {'team': team, 'opponent': opponent, 'season': season, **team_averages, **opponent_averages, 
            'home_game': home_game, 'WL': winlose, 'point_dif': point_dif}

        # Add the data point to the training data
        training_data.append(data_point)

# Write the training data to a new JSON file
with open('data/nba_training_data.json', 'w') as f:
    json.dump(training_data, f, indent=4)