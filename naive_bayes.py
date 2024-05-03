"""
This program uses naive bayes using gaussian probability density to predict
the outcome of NBA games

Authors: David Lybeck, Matthew Zou
5/2/2024
"""

import json
import sys
from nba_api.stats.static import teams
import numpy as np

SEASON = "2023-24"

def process_Input():
    """
    Processes the input when the file is run to make sure everything is good to go

    Returns: team1 name, team2 name and whether the game is Home or away for team 1
    """
    def inputError():
        """
        Error statement
        """
        print(team_str)
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
    
def get_Input(team1, team2, location):
    """
    Gets the data for the given teams and creates an array of input data

    Returns: Numpy array of input data
    """
    with open('data/nba_averages.json', 'r') as f:
        averages = json.load(f)
        
    input = []    
        
    team1_Averages = averages.get(team1, {}).get(SEASON)
    for k, v in team1_Averages.items():
        if k != 'point_dif' and isinstance(v, (int, float)):
            input.append(round(v, 1))

    team2_Averages = averages.get(team2, {}).get(SEASON)
    for k, v in team2_Averages.items():
        if k != 'point_dif' and isinstance(v, (int, float)):
            input.append(round(v, 1))
    input.append(location)
    return np.array(input)

def prepareTrainingData():
    """
    Creates custom training data from the large data set
    Returns: Numpy array of training data
    """
    # Load the training data
    with open('data/nba_training_data.json', 'r') as f:
        data = json.load(f)

    # Create a list of tuples with inputs and labels
    training_data = []
    normal_length = len(data[0])
    for item in data:
        if len(item) != normal_length:
            print(f"{item['team']} {item['opponent']} {item['season']} {len(item)}")
        inputs = []
        for k, v in item.items():
            if k != 'point_dif' and "previous" not in k and isinstance(v, (int, float)):
                inputs.append(round(v, 1))
        label = item['WL']
        training_data.append((inputs, label))

    # Convert the list of tuples to a NumPy structured array
    training_data = np.array(training_data, dtype=[('inputs', float, len(training_data[0][0])), ('label', 'U2')])
    return training_data

def separate_by_class(training_data):
    """
    Creates a Dict to separate training data by Wins and Losses 
    Returns: Dict for wins and losses
    """
    separated = {'W': [], 'L': []}
    for instance in training_data:
        if instance['label'] == 'W':
            separated['W'].append(instance['inputs'])
        else:
            separated['L'].append(instance['inputs'])
    return separated

def calculate_statistics(data):
    """
    Finds the mean and standard deviation of all the data in each dataset
    Returns a numpy array of the means and a numpy array of the stdevs
    """
    means = np.mean(data, axis=0)
    stdevs = np.std(data, axis=0)
    return means, stdevs

def gaussian_probability_density(x, mean, std_dev):
    """
    Gaussian probability density fucntion
    Returns: the Gaussian Probability Density
    """
    exponent = -((x - mean) ** 2 / (2 * std_dev ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)

def find_W_Percent(input, training):
    """
    Find the percent change of a win when given the input data and training data
    Returns: The probability of a win
    """
    #Separate the training data by class
    separated = separate_by_class(training)

    #Calculate statistics for each class
    statistics = {label: calculate_statistics(separated[label]) for label in separated}

    #Calculate likelihoods for each class
    likelihoods = {}
    for label in ['W', 'L']:
        means, stdevs = statistics[label]
        likelihoods[label] = []

        for i in range(len(input)):
            probability = gaussian_probability_density(input[i], means[i], stdevs[i])
            likelihoods[label].append(probability)


    #Calculate probabilities
    prob_win = .5
    prob_loss = .5
    for likelihood_win, likelihood_loss in zip(likelihoods['W'], likelihoods['L']):
        prob_win *= likelihood_win
        prob_loss *= likelihood_loss

    #Prob loss and prob win are very very small numbers right here

    #Normalize the probabilies
    total = prob_win + prob_loss
    return prob_win / total
            

team1, team2, location = process_Input()
#get the input data
input = get_Input(team1, team2, location)

#get the training data
training_data = prepareTrainingData()

#Find the win probability
win_prob = find_W_Percent(input, training_data)

#Get a dict of all teams
teams_dict = teams.get_teams()

#dictionary mapping team abbreviations to team names
team_abbr_to_name = {team['abbreviation']: team['full_name'] for team in teams_dict}

#print result
if(location == 1): location = "at a home game"
else: location = "at an away game"
print(f"I predict there is a {win_prob*100:.2f}% chance that the {team_abbr_to_name[team1]} beat the {team_abbr_to_name[team2]} {location}")