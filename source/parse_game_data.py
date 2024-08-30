"""
    Parses game data from `gol.gg` into a league-draft-analyzer readable format

    Json format:
    {
        "pick":{
            "blue": dict[int:Champ], # Where int describes champ's lane assignment
            "red":  dict[int:Champ]  # Where int describes champ's lane assignment 
        },

        "ban":{
            "blue": list[Champs],
            "red":  list[Champs]
        },

        "tournament": str,
        "game-time":  int,
        "teams":      dict[str1:str2], # Where str1 is blue/red and str2 is team name
        "game-date":  int,
        "result":     int # 0 for blue; 1 for red
    }

    How to run:
    $ python3 source/parse_game_data.py <output.json>

    Requirements:
    requests - for getting page html
    beautifulsoup4 - for scrapping html elements

    TODO: Make json format better for searches
    or make utility functions i can call in command line
    like some sort of library we can import in py shell
    and filter like that
"""

import sys
import requests
import argparse

from typing import Any

### Constants
OUTPUT_FILE_ARG = "output_file"
NUMBER_PLACEHOLDER:str = "&0&1"
GOL_GG_URL:str = f"https://gol.gg/game/stats/{NUMBER_PLACEHOLDER}/page-game"

def main() -> int:
    """Entry point of script
        1. Get info from arguments
        2. 
    """
    # 1. Get info from arguments
    parsed_args:dict = parse_args()
    output_file:str = parsed_args[OUTPUT_FILE_ARG]

    # 2. 

    return 0

def parse_args() -> dict[str,Any]:
  """Parse required args for script"""
  parser = argparse.ArgumentParser("game_data_parser_py")
  parser.add_argument(OUTPUT_FILE_ARG, help="Where game data output will be written into(json).", type=str)
  args = parser.parse_args()

  return {
    OUTPUT_FILE_ARG: getattr(args, OUTPUT_FILE_ARG)
  }

def parse_gol_gg_game(url:str) -> dict:
    """Parses a game from a `gol.gg` page-game

    Main div class='row rowbreak fond-main-cadre'

    Game-Time div class='col-6 text-center'
        Value is in the only <h1> that exists in that div

    Patch div class='col-3 text-right'
        Value is in the div itself

    Blue and Red side team names and match result:
        Team name div class='col-12 <blue|red>-line-header'
            Team name value is in the only <a title='<team-name> stats'>
            Result value (WIN|LOSS) comes after the team name in normal text
  """

if __name__ == "__main__":
    ret:int = main()
    assert isinstance(ret, int)
    sys.exit(ret)
