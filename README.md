
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




```bash
  pip install numpy
```

[Install pytorch](https://pytorch.org/get-started/locally/)
#### If you want to update data or train new RNN:

```bash
  pip install nba_api
```

## Run Naive Bayes
```bash
python naive_bayes.py <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>"
```

## Run Using Pytorch
```bash
python run_run.py <trained_file>.pth <team1 'Home' or 'Away'> <team1 abbreviation> <team2 abbreviation>
```

## Create data (optional)

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