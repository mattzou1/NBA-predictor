import json
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.library.parameters import SeasonAll

#Get a divt og all teams
teams_dict = teams.get_teams()

#Create a dictionary mapping team IDs to team names
team_id_to_name = {team['id']: team['nickname'] for team in teams_dict}

#Print the dictionary
print("\nTeam ID to Name Mapping:")

team_ids = []

#print out all teams (for reference late) and create a list of all team IDs
for team_id, team_name in team_id_to_name.items():
    print(f"    {team_id}: {team_name}")
    team_ids.append(team_id)


#Set up empty DataFrames
df_teams_2023 = pd.DataFrame()
df_teams_all = pd.DataFrame()

# oop through each team and get data
for team_id in team_ids:
    gamelog_teams = teamgamelog.TeamGameLog(team_id=team_id, season='2023')
    df_teams_2023 = pd.concat([df_teams_2023, pd.concat(gamelog_teams.get_data_frames())])

    game_log_teams_all = teamgamelog.TeamGameLog(team_id=team_id, season=SeasonAll.all)
    df_teams_all = pd.concat([df_teams_all, pd.concat(game_log_teams_all.get_data_frames())])

# Convert the DataFrames to JSON
json_data_2023 = df_teams_2023.to_json(orient='records')
json_data_all = df_teams_all.to_json(orient='records')

# Write the JSON data to files
with open('teams_2023.json', 'w') as f:
    json.dump(json_data_2023, f, indent=4)

with open('teams_all.json', 'w') as f:
    json.dump(json_data_all, f, indent=4)

print("JSON files created successfully!")