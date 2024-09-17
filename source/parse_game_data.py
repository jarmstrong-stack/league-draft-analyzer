"""
    Parses game data from `gol.gg` into a league-draft-analyzer readable format

    Json format: # Note that Champs are normalized into int's
    {
        "pick":{
            "blue": dict[int:Champ], # Where int describes champ's lane assignment
            "red":  dict[int:Champ]  # Where int describes champ's lane assignment
        },

        "ban":{
            "blue": list[Champ], 
            "red":  list[Champ]
        },

        "tournament": str,
        "game-time":  int,
        "patch":      int,
        "teams":      dict[str1:str2], # Where str1 is blue/red and str2 is team name
        "game-date":  int,
        "result":     int # 0 for blue; 1 for red
    }

    How to run:
    $ python3 source/parse_game_data.py <output.json> <start_game_number> <stop_game_number>

    5k games:
        - eta ~30 min
        - 2.6Mb size 

    Requirements:
    requests - for getting page html
    beautifulsoup4 - for scrapping html elements
"""

import sys
import time
import json
import yaml
import requests
import argparse
from bs4 import BeautifulSoup
from typing import Any

import constants as CONST

### Constants
OUTPUT_FILE_ARG = "output_file"
START_NUMBER_ARG = "start_number"
STOP_NUMBER_ARG = "stop_number"

NUMBER_PLACEHOLDER = "&0&1"
GOL_GG_URL = f"https://gol.gg/game/stats/{NUMBER_PLACEHOLDER}/page-game/"

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
        2. Initialize
        3. Go through `START_NUMBER_ARG` until `STOP_NUMBER_ARG` and compute urls
        4. Save dataset into `OUTPUT_FILE_ARG`
        5. Print finish analytics
    """

    # 0. Get info from arguments
    parsed_args:dict = parse_args()

    # 2. Initialize
    time_start = time.time()
    scraped_data = list()
    champ_int_mapping = load_champ_to_int_dict(CONST.CHAMP_TO_INT_DATABASE)

    # 3. Start loop 
    for current_game_number in range(parsed_args[START_NUMBER_ARG], parsed_args[STOP_NUMBER_ARG] + 1):
        print(f">>> Parsing game number {current_game_number}")
        try:
            game_url = GOL_GG_URL.replace(NUMBER_PLACEHOLDER, str(current_game_number))
            scraped_game = parse_gol_gg_game(game_url)
            normalize_data(scraped_game, champ_int_mapping)
            scraped_data.append(scraped_game)
        except Exception as e:
            print(f">>> ERROR in {current_game_number} : {str(e)}")

    # 4. Save data
    with open(parsed_args[OUTPUT_FILE_ARG], 'w', encoding='utf-8') as output_file:
        json.dump(scraped_data, output_file, indent=2)

    # 5. Finish analytics
    time_taken = round(time.time() - time_start, 3)
    total_games = len(scraped_data)
    games_per_second = round(total_games / time_taken, 3)

    print(f"#" * 38)
    print(f"# Total games parsed: {total_games}")
    print(f"# Time taken: {time_taken}s")
    print(f"# Games per second: {games_per_second}g/s")
    print(f"# Output to: {parsed_args[OUTPUT_FILE_ARG]}")
    print(f"#" * 38)
    return 0

def parse_gol_gg_game(url:str) -> dict:
    """Takes in a `gol.gg` page-game url and scrapes the content into `result_dict`

    - This function should be handled in a try catch statement because it has no safety checks
    This is because we WANT it to fail if it doesn't find something in the web page
    """
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
        raise Exception(f"Request failed with status code {response.status_code}")
    
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

def normalize_data(result_dict:dict, champ_mapping:dict) -> None:
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
    # Hack single digit patch numbers (.1, .2, .3, etc..) to (.01, .02, .03)
    if len(result_dict[CONST.PATCH_DATA].replace(' v', '').split('.')[1]) == 1:
        normalized_patch = normalized_patch.replace(normalized_patch, normalized_patch[:2] + '0' + normalized_patch[2])
    result_dict[CONST.PATCH_DATA] = int(normalized_patch)

    # game-date
    game_date = result_dict[CONST.GAMEDATE_DATA].split(' ')[0].replace('-', '') # Get YYYYMMDD date from game date
    result_dict[CONST.GAMEDATE_DATA] = int(game_date)

    # Champs
    def parse_champs_helper(entry):
        """Handles parsing picked and banned champs from result_dict"""
        if isinstance(entry, dict):
            for role, champ in entry.items():
                entry[role] = get_champ_int_by_name(champ, champ_mapping, CONST.CHAMP_TO_INT_DATABASE)
        elif isinstance(entry, list):
            for i in range(len(entry)):
                entry[i] = get_champ_int_by_name(entry[i], champ_mapping, CONST.CHAMP_TO_INT_DATABASE)
    parse_champs_helper(result_dict[CONST.PICK_DATA][CONST.BLUE_SIDE])
    parse_champs_helper(result_dict[CONST.PICK_DATA][CONST.RED_SIDE])
    parse_champs_helper(result_dict[CONST.BAN_DATA][CONST.BLUE_SIDE])
    parse_champs_helper(result_dict[CONST.BAN_DATA][CONST.RED_SIDE])

def load_champ_to_int_dict(yml_path: str) -> dict:
    """Loads a yml style dict of champ mappings to integer

    Example of yml:
    ```yml
    Yone: 1
    Thresh: 2
    Nidalee: 3```
    """
    parsed_dict = dict()
    with open(yml_path, 'r', encoding='utf-8') as input_file:
        yml_content = input_file.read()
        parsed_dict = yaml.safe_load(yml_content)

    # If we cant parse anything means the file does not exist, create it
    if parsed_dict == None:
        with open(yml_path, 'w'):
            parsed_dict = dict()
    return parsed_dict

def write_to_champ_to_int_dict(yml_path: str, new_dict: dict) -> None:
    """Override champ to dict mapping file in `yml_path` with `new_dict`"""
    print("### Updating champ to int mapping...")
    with open(yml_path, 'w', encoding='utf-8') as input_file:
        input_file.write(yaml.dump(new_dict, sort_keys=False))

def get_champ_int_by_name(champ_name: str, champ_mapping: dict, yml_file: str) -> int:
    """Uses `champ_mapping` to get a champion's id/integer value by their name
    Will add the champ if the name is not found in the database yet
    """
    if champ_name in champ_mapping:
        return champ_mapping[champ_name]
    
    # Calculate new champ's integer value and add it
    if len(champ_mapping) == 0:
        new_int_value = 1
    else:
        highest_champ:str = max(champ_mapping, key=champ_mapping.get)
        new_int_value:int = champ_mapping[highest_champ] + 1
    champ_mapping[champ_name] = new_int_value 
    write_to_champ_to_int_dict(yml_file, champ_mapping)
    return new_int_value 

if __name__ == "__main__":
    ret:int = main()
    assert isinstance(ret, int)
    sys.exit(ret)
