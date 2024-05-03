"""
Using the nba_api create a large data set of all NBA stats by season and team

Authors: David Lybeck, Matthew Zou
5/2/2024
"""

from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams
import json
import os

# List of all seasons to fetch data for
seasons = [
    '2023-24','2022-23','2021-22','2020-21','2019-20','2018-19',
    '2017-18','2016-17','2015-16','2014-15','2013-14','2012-13',
    '2011-12','2010-11','2009-10','2008-09','2007-08','2006-07',
    '2005-06','2004-05','2003-04','2002-03','2001-02','2000-01',
    '1999-00','1998-99','1997-98','1996-97','1995-96','1994-95',
    '1993-94','1992-93','1991-92','1990-91','1989-90','1988-89',
    '1987-88','1986-87','1985-86','1984-85','1983-84','1982-83',
    '1981-82','1980-81','1979-80','1978-79','1977-78','1976-77',
    '1975-76','1974-75','1973-74','1972-73','1971-72','1970-71',
    '1969-70','1968-69','1967-68','1966-67','1965-66','1964-65',
    '1963-64','1962-63','1961-62','1960-61','1959-60','1958-59',
    '1957-58','1956-57','1955-56','1954-55','1953-54','1952-53',
    '1951-52','1950-51','1949-50','1948-49','1947-48','1946-47'
]

#Get a dict of all teams
teams_dict = teams.get_teams()

#dictionary mapping team IDs to team names
team_id_to_name = {team['id']: team['nickname'] for team in teams_dict}

#Print the dictionary
print("\nTeam ID to Name Mapping:")

team_ids = []

#print out all teams (for reference later) and create a list of all team IDs
for team_id, team_name in team_id_to_name.items():
    print(f"    {team_id}: {team_name}")
    team_ids.append(team_id)

#create empty list to store the data
data = []

#Iterate over each season for team
for season in seasons:
    print(season)
    for team_id in team_ids:
        try:
            team_gamelogs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
            team_gamelogs_data = team_gamelogs.get_data_frames()[0]
            team_gamelogs_data['SEASON'] = season
            team_gamelogs_data['TEAM_ID'] = team_id
            data.append(team_gamelogs_data)
            print(f"    Fetched data for the {team_id_to_name.get(team_id)}")
        except Exception as e:
            print(f"    Error fetching data for the {team_id_to_name.get(team_id)} in season {season}: {e}")

#Save the data as a JSON file
with open('data/nba_data.json', 'w') as file:
    json_data = [row.to_dict('records') for row in data]
    json.dump(json_data, file, indent=4)