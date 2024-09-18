"""
    league-draft-analyzer project wide constants
"""

### league-draft-analyzer
LDA_EMPTY_VALUE = "lda_empty"
LDA_NOT_USED_VALUE = "lda_not_used"
LDA_BAD_VALUE = "lda_bad_value"
DATA_FOLDER = './data'
CHAMP_TO_INT_DATABASE = f'{DATA_FOLDER}/champ_mapping.yml'
GAME_DATABASE = f'{DATA_FOLDER}/all_games.json'

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

### Results
BLUE_WIN = 0
RED_WIN = 1

### gol.gg constants
GOL_GG_WIN = "WIN"
