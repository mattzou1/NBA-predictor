import sys
from nba_api.stats.static import teams
import json

def process_Input():
    """
    Processes the input when the file is run to make sure everything is good to go

    Returns: team1 name, team2 name, whether the game is Home or away for team 1, and the desired file name
    """
    def inputError():
        """
        Error statement
        """
        print(team_str)
        print("Usage: python run_rnn.py <trained_file>.pth <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>")
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
        
        
    if(len(sys.argv) != 3): inputError()
    team1 = sys.argv[1]
    team2 = sys.argv[2]
    if(team1 not in all_teams or team2 not in all_teams or team1 == team2): inputError()

        
    return team1, team2

def predict(team1, team2):
    # Load the data from the JSON file
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    return results[team1][team2]

team1, team2 = process_Input()
outcome = predict(team1, team2)

print(outcome)