from nba_api.stats.endpoints import playergamelog
import pandas as pd
from nba_api.stats.library.parameters import SeasonAll
import json

# Call the API endpoints passing in LeBron's ID & which season
gamelog_bron = playergamelog.PlayerGameLog(player_id='2544', season='2023')

# Convert gamelog object into a pandas DataFrame
df_bron_games_2023 = pd.concat(gamelog_bron.get_data_frames())

# Get data for all seasons
game_log_bron_all = playergamelog.PlayerGameLog(player_id='2544', season=SeasonAll.all)
df_bron_games_all = pd.concat(game_log_bron_all.get_data_frames())

# Convert the DataFrames to JSON
json_data_2023 = df_bron_games_2023.to_json(orient='records')
json_data_all = df_bron_games_all.to_json(orient='records')

# Write the JSON data to files
with open('lebron_2023.json', 'w') as f:
    json.dump(json_data_2023, f, indent=4)

with open('lebron_all.json', 'w') as f:
    json.dump(json_data_all, f, indent=4)

print("JSON files created successfully!")