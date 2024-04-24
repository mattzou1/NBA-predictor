#First we import the endpoints
#We Will be using Pandas dataframes to manipulates the data
from nba_api.stats.endpoints import playergamelog
import pandas as pd

#Call the API endpoints passing in lebron's ID & which season
gamelog_bron =playergamelog.PlayerGameLog(player_id='2544' , season='2018')

#Convert gamelog object into a pandas
#Can Also Convert to JSON to dictionary
df_bron_games_2018 = gamelog_bron.get_data_frames()

#if want all seasons, you must imort SeasonAll parameters
from nba_api.stats.library.parameters import SeasonAll 

game_log_bron_all = playergamelog.PlayerGameLog(player_id='2544' ,season=SeasonAll.all)

df__bron_games_all = game_log_bron_all.get_data_frames()