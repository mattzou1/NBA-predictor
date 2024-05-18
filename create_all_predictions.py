import copy
from nba_api.stats.static import teams
from run_rnn import run
import json

MODEL = 'nba_rnn.pth'

def create_team_list():
    # Get a dict of all teams
    teams_dict = teams.get_teams()

    # Dictionary mapping team names to team abbreviations
    team_abbrs = sorted([team['abbreviation'] for team in teams_dict])

    return team_abbrs

def all_combinations(list1):
    list2 = copy.deepcopy(list1)
    matches = []

    while(list1[-1] != list2[0]):
        # Move first elem of list2 to the end
        first = list2.pop(0)
        list2.append(first)

        combos = list(zip(list1, list2))
        for match in combos:
            matches.append(match)

    return matches

def findOutcomes(matches):
    outcomes = {}
    
    length = len(matches)
    
    count = 1
    for team1, team2 in matches:
        print(f"    ({round(count/length*100, 2)}%) Predicting {team1} vs {team2}", end='\r', flush=True)
        outcome = run(team1, team2, MODEL)
        if team1 not in outcomes:
            outcomes[team1] = {}
            
        outcomes[team1][team2] = outcome
        count += 1

    return outcomes

teams = create_team_list()
print(" Setting up all matches")
matches = all_combinations(teams)
print(" Predicitng all matches")
outcomes = findOutcomes(matches)

print("Saving...")
with open('results.json', 'w') as f:
    json.dump(outcomes, f, indent=4)
