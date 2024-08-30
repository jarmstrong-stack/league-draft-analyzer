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
    "game-date":  int
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

### Constants
OUTPUT_FILE_ARG = "output_file"
NUMBER_PLACEHOLDER:str = "&0&1"
GOL_GG_URL:str = f"https://gol.gg/game/stats/{NUMBER_PLACEHOLDER}/page-summary/"

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

def parse_args() -> dict[str:]:
  """Parse required args for script"""
  parser = argparse.ArgumentParser("game_data_parser_py")
  parser.add_argument(OUTPUT_FILE_ARG, help="Where game data output will be written into(json).", type=str)
  args = parser.parse_args()

  return {
    OUTPUT_FILE_ARG: getattr(args, OUTPUT_FILE_ARG)
  }

def parse_gol_gg_summary(url:str) -> dict:
  """Parses all games from a `gol.gg` page-summary

    Main div class='col-cadre pb-4'

    Team div class='row pb-3'
      Team name div classes='col-4 col-sm-5 text-center'

    Game div class='row pb-1'
      Pick/ban teams div class='col-4 col-sm-5 text-center' (x2!)
        Match win or lose status h1 class='text_<victory|defeat>'
        Match pick/bans separation span class='text-uppercase'
          Champion pick/bans <a title='* stats'/>

      Middle part div class='col-4 col-sm-2 text-center align-middle <left|right>-side-win'
        Time is presented in the single <h1>x</h1>
  """

if __name__ == "__main__":
  sys.exit(main())
