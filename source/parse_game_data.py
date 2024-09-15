"""
    Parses game data from `gol.gg` into a league-draft-analyzer readable format

    Json format:
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
        "patch":      int,
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
from bs4 import BeautifulSoup
from typing import Any

import constants as CONST

### Constants
OUTPUT_FILE_ARG = "output_file"
START_NUMBER_ARG = "start_number"
STOP_NUMBER_ARG = "stop_number"
NUMBER_PLACEHOLDER:str = "&0&1"
GOL_GG_URL:str = f"https://gol.gg/game/stats/{NUMBER_PLACEHOLDER}/page-game"

def parse_args() -> dict[str,Any]:
    """Parse required args for script"""
    parser = argparse.ArgumentParser("game_data_parser_py")
    parser.add_argument(OUTPUT_FILE_ARG, help="Where game data output will be written into(json).", type=str)
    parser.add_argument(START_NUMBER_ARG, help="Which game number to start.", type=int)
    parser.add_argument(STOP_NUMBER_ARG, help="Which game number to stop.", type=int)
    args = parser.parse_args()

    return {
        OUTPUT_FILE_ARG: getattr(args, OUTPUT_FILE_ARG),
        START_NUMBER_ARG: getattr(args, START_NUMBER_ARG),
        STOP_NUMBER_ARG: getattr(args, STOP_NUMBER_ARG)
    }

def main() -> int:
    """Entry point of script
        1. Get info from arguments
        2. 
    """
    # 1. Get info from arguments
    parsed_args:dict = parse_args()

    # 2. 

    return 0

def parse_gol_gg_game(url:str) -> dict|None:
    """Takes in a `gol.gg` page-game url and scrapes the content into `result_dict`"""
    # Initialize the result dictionary
    result_dict = {
        CONST.PICK_DATA: {CONST.BLUE_SIDE: {}, CONST.RED_SIDE: {}},
        CONST.BAN_DATA: {CONST.BLUE_SIDE: [], CONST.RED_SIDE: []},
        CONST.TOURNAMENT_DATA: CONST.LDA_EMPTY_VALUE,
        CONST.GAMETIME_DATA: -1,
        CONST.PATCH_DATA: -1,
        CONST.TEAMS_DATA: {CONST.BLUE_SIDE: CONST.LDA_EMPTY_VALUE, CONST.RED_SIDE: CONST.LDA_EMPTY_VALUE},
        CONST.GAMEDATE_DATA: -1,
        CONST.GAMERESULT_DATA: -1
    }

    # Simulate a real browser with headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': url
    }

    # Send the request with headers
    response = requests.get(url, headers=headers)

    # Ensure the request was successful
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return None
    
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract game time
    game_time_element = soup.find('div', class_='col-6 text-center')
    result_dict[CONST.GAMETIME_DATA] = game_time_element.find('h1').text.strip()

    # Extract patch
    patch_element = soup.find('div', class_='col-3 text-right')
    result_dict[CONST.PATCH_DATA] = patch_element.text

    # Extract game date
    game_date_element = soup.find('div', class_='col-12 col-sm-5 text-right')
    result_dict[CONST.GAMEDATE_DATA] = game_date_element.text

    # Extract tournament
    tournament_element = soup.find('div', class_='col-12 col-sm-7')
    result_dict[CONST.TOURNAMENT_DATA] = tournament_element.find('a').text.split(' ')[0] # Get the first word of tournament

    # Extract team elements
    blue_team_element = soup.find('div', class_='blue-line-header')
    red_team_element = soup.find('div', class_='red-line-header')

    # Extract team names and game result
    result_dict[CONST.TEAMS_DATA][CONST.BLUE_SIDE] = blue_team_element.text.strip().split(' - ')[0]
    if CONST.GOL_GG_WIN in blue_team_element.text:
        result_dict[CONST.GAMERESULT_DATA] = CONST.BLUE_WIN

    result_dict[CONST.TEAMS_DATA][CONST.RED_SIDE] = red_team_element.text.strip().split(' - ')[0]
    if CONST.GOL_GG_WIN in red_team_element.text:
        result_dict[CONST.GAMERESULT_DATA] = CONST.RED_WIN

    # Extract champion bans (blue and red side)
    bans_blue_element = soup.find_all('div', class_='col-sm-6')[0]
    blue_bans = bans_blue_element.find_all('a', class_='black_link')
    bans_red_element = soup.find_all('div', class_='col-sm-6')[1]
    red_bans = bans_red_element.find_all('a', class_='black_link')

    # Parse bans
    result_dict[CONST.BAN_DATA][CONST.BLUE_SIDE] = [champ.find('img')['alt'] for champ in blue_bans]
    result_dict[CONST.BAN_DATA][CONST.RED_SIDE] = [champ.find('img')['alt'] for champ in red_bans]

    # Extract champion picks
    picks_blue_element = soup.find_all('div', class_='col-sm-6')[0]
    picks_blue_element = picks_blue_element.find_all('div', class_='col-10')[1]
    blue_picks = picks_blue_element.find_all('a', title=True)
    picks_red_element = soup.find_all('div', class_='col-sm-6')[1]
    picks_red_element = picks_red_element.find_all('div', class_='col-10')[1]
    red_picks = picks_red_element.find_all('a', title=True)

    # Parse picks 
    result_dict[CONST.PICK_DATA][CONST.BLUE_SIDE] = {i+1: champ.find('img')['alt'] for i, champ in enumerate(blue_picks)}
    result_dict[CONST.PICK_DATA][CONST.RED_SIDE] = {i+1: champ.find('img')['alt'] for i, champ in enumerate(red_picks)}

    return result_dict

def normalize_data(result_dict:dict) -> dict:
    """Normalize data from `parse_gol_gg_game` into model friendly data
    
    - game-time: str('28:10') > int(1681) # seconds
    - patch: str(' v14.18') > int(1418)
    - game-date: str('2024-08-30 (xyz)') > int(20240830)
    - champs: str('Ahri') > int(22)
    """

    # game-time
    game_time_minutes, game_time_seconds = map(int, result_dict[CONST.GAMETIME_DATA].split(":"))
    result_dict[CONST.GAMETIME_DATA] = game_time_minutes * 60 + game_time_seconds # Store time as seconds

    # patch
    normalized_patch = result_dict[CONST.PATCH_DATA].replace(' v', '').replace('.', '') # v14.18 into 1418
    result_dict[CONST.PATCH_DATA] = int(normalized_patch)

    # game-date
    game_date = result_dict[CONST.GAMEDATE_DATA].split(' ')[0].replace('-', '') # Get YYYYMMDD date from game date
    result_dict[CONST.GAMEDATE_DATA] = int(game_date)

if __name__ == "__main__":
    ret:int = main()
    assert isinstance(ret, int)
    a = parse_gol_gg_game("https://gol.gg/game/stats/62439/page-game/")
    normalize_data(a)
    print(a)
    sys.exit(ret)
