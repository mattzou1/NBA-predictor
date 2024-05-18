"""
Using the nba_api create a large data set of all NBA stats by season and team

Authors: David Lybeck, Matthew Zou
5/2/2024
"""

from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams
from pandas import DataFrame
import json
import os
import datetime

# Get the current year
current_year = datetime.datetime.now().year 

# Initialize an empty list to store the seasons
seasons = []

# Generate the list of seasons starting from the 1947-48 season until the current season
start_year = 1946
end_year = current_year
while start_year < current_year+1:
    season = f"{start_year}-{(start_year+1)%100:02d}"
    seasons.append(season)
    start_year += 1
    end_year += 1
seasons.reverse()

#Get a dict of all teams
teams_dict = teams.get_teams()

#dictionary mapping team IDs to team names
team_id_to_name = {team['id']: team['nickname'] for team in teams_dict}

team_ids = []

#create a list of all team IDs
for team_id, team_name in team_id_to_name.items():
    team_ids.append(team_id)

#create empty list to store the data
data = []

#Iterate over each season for team
for season in seasons:
    print(f"    Fetching data for the {season} season", end='\r', flush=True)
    for team_id in team_ids:
        try:
            team_gamelogs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
            team_gamelogs_data = team_gamelogs.get_data_frames()[0]
            team_gamelogs_data['SEASON'] = season
            team_gamelogs_data['TEAM_ID'] = team_id
            data.append(team_gamelogs_data)
        except Exception as e:
            print(f"    Error fetching data for the {team_id_to_name.get(team_id)} in season {season}: {e}")

#Save the data as a JSON file
print(" Saving...")
with open('data/nba_data.json', 'w') as file:
    json_data = [row.to_dict('records') for row in data]
    json.dump(json_data, file, indent=4)