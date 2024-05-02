import json
import sys
from nba_api.stats.static import teams

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
    team_name_to_abbr = {team['full_name']: team['abbreviation'] for team in teams_dict}
    
    all_teams = set()
    team_str = "Abbreviation | Full Name\n"
    
    #create a string to show abbeviations for each team and create a set of team abbreviations
    for team_name, team_abbr in sorted(team_name_to_abbr.items()):
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

def create_data(team1, team2, location):
    # Load the averages from the averages JSON file
    with open('data/nba_averages.json', 'r') as f:
        averages = json.load(f)

    # Initialize a list to store the training data
    input_data = {}
    
    team_averages = {f'team_average_{k}': v for k, v in averages.get(team1, {}).get(SEASON, {}).items()}
    opponent_averages = {f'opponent_average_{k}': v for k, v in averages.get(team2, {}).get(SEASON, {}).items()}
    
    input_data = {'team': team1, 'opponent': team2, 'season': SEASON, **team_averages, **opponent_averages, 
            'home_game': location}

    # Write the training data to a new JSON file
    with open(f'data/{team1}v{team2}.json', 'w') as f:
        json.dump(input_data, f, indent=4)
        
team1, team2, location = process_Input()

create_data(team1, team2, location)