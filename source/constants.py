"""
    league-draft-analyzer project wide constants
"""

### league-draft-analyzer
LDA_EMPTY_VALUE = "lda_empty"
LDA_NOT_USED_VALUE = "lda_not_used"
LDA_WEIGHTS_PATH = "data/lda_net.pth"
LDA_BAD_VALUE = "lda_bad_value"
DATA_FOLDER = './data'
CHAMP_TO_INT_DATABASE = f'{DATA_FOLDER}/champ_mapping.yml'
#GAME_DATABASE = f'{DATA_FOLDER}/all_games.json'
GAME_DATABASE = f'{DATA_FOLDER}/all_games_seperated_all_synergy.json'

# pytorch values
DEVICE_CUDA = "cpu"

# driver args
DRIVER_ACTION = "action"
DRIVER_TRAIN = "train"
DRIVER_PREDICT = "predict"

### Game data dict
PICK_DATA = "pick"
BAN_DATA  = "ban"
BLUE_SIDE = "blue"
RED_SIDE  = "red"
PATCH_DATA = "patch"
TEAMS_DATA = "teams"
TOURNAMENT_DATA = "tournament"
GAMEDATE_DATA = "game-date"
GAMETIME_DATA = "game-time"
GAMERESULT_DATA = "result"
SYNERGY_DATA = "synergy"
COUNTER_DATA = "counter"

### Results
BLUE_WIN = 0
RED_WIN = 1

### Roles
TOP_ROLE = "1"
JGL_ROLE = "2"
MID_ROLE = "3"
ADC_ROLE = "4"
SUP_ROLE = "5"

### gol.gg constants
GOL_GG_WIN = "WIN"
