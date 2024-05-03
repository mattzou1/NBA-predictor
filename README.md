
# NBA Game Predictor

This is an implementation of both a recurrent neural network to predict the outcome and point differential for NBA games, and naive bayes to predict the probability of a win for a given match

2 May 2024


## Authors

- [David Lybeck](https://github.com/Dlybeck)
- [Matthew Zou](https://github.com/mattzou1)


## Set up

Clone the project

```bash
  git clone https://github.com/mattzou1/NBA-predictor.git
```

Go to the project directory

Unzip the data folder into the directory


```bash
  pip install numpy
```

[Install pytorch](https://pytorch.org/get-started/locally/)
#### If you want to update data or train a new RNN:

```bash
  pip install nba_api
```

```bash
  pip install tensorflow
```

## Run Naive Bayes
```bash
python naive_bayes.py <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>"
```

## Run Using Pytorch
```bash
python run_run.py <trained_file>.pth <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>
```

## Run Using Tensorflow
```bash
python run_tf_run.py <trained_file>.h5 <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>
```

### List of team abbreviations to team names for reference
##### ATL:   Atlanta Hawks
##### BOS:   Boston Celtics
##### BKN:   Brooklyn Nets
##### CHA:   Charlotte Hornets
##### CHI:   Chicago Bulls
##### CLE:   Cleveland Cavaliers
##### DAL:   Dallas Mavericks
##### DEN:   Denver Nuggets
##### DET:   Detroit Pistons
##### GSW:   Golden State Warriors
##### HOU:   Houston Rockets
##### IND:   Indiana Pacers
##### LAC:   Los Angeles Clippers
##### LAL:   Los Angeles Lakers
##### MEM:   Memphis Grizzlies
##### MIA:   Miami Heat
##### MIL:   Milwaukee Bucks
##### MIN:   Minnesota Timberwolves
##### NOP:   New Orleans Pelicans
##### NYK:   New York Knicks
##### OKC:   Oklahoma City Thunder
##### ORL:   Orlando Magic
##### PHI:   Philadelphia 76ers
##### PHX:   Phoenix Suns
##### POR:   Portland Trail Blazers
##### SAC:   Sacramento Kings
##### SAS:   San Antonio Spurs
##### TOR:   Toronto Raptors
##### UTA:   Utah Jazz
##### WAS:   Washington Wizards

## Create data (optional)
This will create all the data files currently inside data.zip

### Create complete data Set

```bash
python create_data.py
```

### Create data of averages for Naive Bayes and RNN

```bash
python create_averages.py
```

### Create data of averages for Naive Bayes and RNN

```bash
python create_training_data.py
```

## Create RNN (optional)

### Pytorch
```bash
python make_rnn.py
```

### Tensorflow
```bash
python make_tf_rnn.py
```