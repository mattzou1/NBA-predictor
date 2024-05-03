import json
import math
from collections import defaultdict

# Load the data from the JSON file
with open('data/nba_data.json', 'r') as f:
    data = json.load(f)

# Initialize a dictionary to store the sum of stats for each team for each season
team_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# Initialize a dictionary to store the number of games and wins for each team for each season
team_games = defaultdict(lambda: defaultdict(int))
team_wins = defaultdict(lambda: defaultdict(int))

# Initialize a dictionary to store the count of non-NaN stats for each team for each season
team_stat_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Fields to exclude
exclude_fields = {'TEAM_ID', 'AVAILABLE_FLAG', 'VIDEO_AVAILABLE', 'BLKA', 'PFD'}

# A set to store all stat names
all_stat_names = set()

# Iterate over the data
for season_data in data:
    for game_data in season_data:
        # Get the team and season
        team = game_data['TEAM_ABBREVIATION']
        season = game_data['SEASON']

        # Only consider the data from the 1985-86 season and later
        if int(season.split('-')[0]) < 1985:
            continue

        # Update the sum of stats and the number of games
        for stat, value in game_data.items():
            # Exclude fields that end with '_RANK'
            if stat.endswith('_RANK') or stat in exclude_fields:
                continue
            if isinstance(value, (int, float)) and not math.isnan(value):
                team_stats[team][season][stat] += value
                team_stat_counts[team][season][stat] += 1
                all_stat_names.add(stat)
        team_games[team][season] += 1

        # Count the number of wins
        if game_data['WL'] == 'W':
            team_wins[team][season] += 1

# Calculate the averages and win percentages
team_averages = defaultdict(lambda: defaultdict(dict))
for team, seasons in team_stats.items():
    for season, stats in seasons.items():
        for stat in all_stat_names:
            total = stats.get(stat, 0)
            count = team_stat_counts[team][season].get(stat, 0)
            team_averages[team][season][stat] = total / count if count > 0 else None
        team_averages[team][season]['WIN_PCT'] = team_wins[team][season] / team_games[team][season]

# Write the averages to a new JSON file
with open('data/nba_averages.json', 'w') as f:
    json.dump(team_averages, f, indent=4)
