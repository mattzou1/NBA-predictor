
# NBA Game Predictor

This is an implementation of both a recurrent neural network to predict the outcome and point differential for NBA games, and naive bayes to predict the probability of a win for a given match.  
We could not get our RNNs to heavily consider previous sequential output. Our RNNs act almost like feed foward networks.   
Our RNNs and our naive bayes programs seem to predict current NBA games' point differential and outcome to a good degree of accuracy     
models/torch_DEMO: RNN created and trained using Pytorch using simple RNN cells.  
models/tf_DEMO: RNN created and trained using tensorflow using LSTM Cells.  

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

Unzip the data folder into the directory. After this, you should have a data folder in the directory.


[Install pytorch](https://pytorch.org/get-started/locally/)
#### If you want to update data or train a new RNN:

```bash
  pip install nba_api
```

```bash
  pip install tensorflow
```

```bash
  pip install numpy
```


## Run Pytorch Neural Network (Most User Friendly)
```bash
python predict.py <Home Team Abbreviation> <Away Team Abbreviation>
```

#### Example
```bash
python predict.py DEN LAL
```


## Run Naive Bayes
```bash
python naive_bayes.py <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>"
```
#### Example
```bash
python naive_bayes.py Home DEN LAL
```


## Run Tensorflow Neural Network
```bash
python run_tf_run.py <trained_file>.h5 <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>
```
#### Example
```bash
python run_tf_rnn.py tf_DEMO.h5 Home DEN LAL
```


## List of team abbreviations to team names for reference:
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

## Update RNN (pytorch version only)
```bash
python update_data.py
```

This command will update all the data to be as current as possible, retrain the pytorch RNN, and create a file 'results.json' that contains all of the possible matches and predicted outcomes for the model.
