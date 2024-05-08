"""
Creates the training data for neural networks and naive bayes and writes it to nba_training_data.json. Training data includes 
season averages for both teams, previous 3 game stats, home or away game, point differential, and win/lose. 

Authors: David Lybeck, Matthew Zou
5/2/2024
"""

import json
import math
from collections import defaultdict

def get_previous_game(num, index):
    """
    Returns a dictionary that includes stats from num previous games ago
    """
    # Fields to exclude
    exclude_fields = {'TEAM_ID', 'AVAILABLE_FLAG', 'VIDEO_AVAILABLE', 'BLKA', 'PFD', 'WL'}
    team_previous_game = {}
    previous_game = season_data[index + num].items()
    for stat, value in previous_game:
            # Exclude fields that end with '_RANK'
            if stat.endswith('_RANK') or stat in exclude_fields:
                continue
            if isinstance(value, (int, float)) and not math.isnan(value):
                team_previous_game[f'team_{num}previous_game_{stat}'] = value
    return team_previous_game

# Load the data from the original JSON file
with open('data/nba_data.json', 'r') as f:
    data = json.load(f)

# Load the averages from the averages JSON file
with open('data/nba_averages.json', 'r') as f:
    averages = json.load(f)

# Initialize a list to store the training data
training_data = []

correct_length = -1

oldSeason = ''
# Iterate over the data
for season_data in data:
    for i in range(len(season_data)):
        game_data = season_data[i]
        # Get the team, opponent, season, and win/loss
        team = game_data['TEAM_ABBREVIATION']
        opponent = game_data['MATCHUP'].split(' ')[2]
        season = game_data['SEASON']
        winlose = game_data['WL']
        point_dif = game_data['PLUS_MINUS']
        
        # Print status
        if(season != oldSeason):
            oldSeason = season
            print(f"    Parsing data for the {season} season")

        # Only consider the data from the 1985-86 season and later
        if int(season.split('-')[0]) < 1985:
            continue

        num_previous_games = 3
        # Don't include last few games of a season
        if i + num_previous_games >= len(season_data):
            continue
        
        # Get the averages for the team and opponent
        team_averages = {f'team_average_{k}': v for k, v in averages.get(team, {}).get(season, {}).items()}
        opponent_averages = {f'opponent_average_{k}': v for k, v in averages.get(opponent, {}).get(season, {}).items()}
        
        previous_games_data = {}
        for j in range(1, num_previous_games + 1):
            previous_games_data.update(get_previous_game(j, i))

        # Determine if the game was a home game or an away game
        home_game = 1 if 'vs.' in game_data['MATCHUP'] else 0

        # Create a data point with the averages, the home game indicator, and the label
        data_point = {'team': team, 'opponent': opponent, 'season': season, **team_averages, **opponent_averages, **previous_games_data,
            'home_game': home_game, 'WL': winlose, 'point_dif': point_dif}
        # Set the correct length as the length of the first data point
        if correct_length == -1:
            correct_length = len(data_point)
    
        # Add the data point to the training data if it is the correct length
        if len(data_point) == correct_length:
            training_data.append(data_point)

# Write the training data to a new JSON file
print(" Saving...")
with open('data/nba_training_data.json', 'w') as f:
    json.dump(training_data, f, indent=4)