"""
    Helper code to join large ammounts of `league-draft-analyzer` data
    into a single file
"""

import json
import glob
import constants as CONST

def remove_bad_data(data:list):
    """Removes bad data from `data`"""
    new_data = list()

    for game in data:
        if not is_bad_game(game):
            new_data.append(game)

    return new_data

def is_bad_game(game:dict) -> bool:
    """Checks if `data` contains any badly formatted, badly parsed data or missing content"""
    try:
        # Valid Pick/Ban
        assert len(game[CONST.PICK_DATA][CONST.BLUE_SIDE]) == 5
        assert len(game[CONST.PICK_DATA][CONST.RED_SIDE]) == 5
        assert len(game[CONST.BAN_DATA][CONST.BLUE_SIDE]) != 0
        assert len(game[CONST.BAN_DATA][CONST.RED_SIDE]) != 0

        # Valid result
        assert game[CONST.GAMERESULT_DATA] in [CONST.BLUE_WIN, CONST.RED_WIN]

        # Valid patch
        assert isinstance(game[CONST.PATCH_DATA], int)
    except (AssertionError, ValueError):
        return True 
    return False 

def merge_json_files(input_folder, output_file, file_mask):
    """Function to load all json files and combine the entries"""
    all_data = []

    # Get all JSON files in the input folder
    json_files = glob.glob(f"{input_folder}/{file_mask}")

    # Loop through each file and load its data
    for file in json_files:
        print(f">>> Loading {file}...")
        with open(file, 'r') as f:
            data = json.load(f)
            # Check if the file contains a list of entries
            if isinstance(data, list):
                print(f"> Checking for bad data...")
                good_data = remove_bad_data(data)
                print(f"> Bad games: {len(data) - len(good_data)}")
                all_data.extend(good_data)

    # Write combined data to a new JSON file
    print(f"### Finished... Exporting to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Successfully merged {len(json_files)} files into {output_file}")

if __name__ == "__main__":
    input_folder = CONST.DATA_FOLDER
    output_file = CONST.GAME_DATABASE
    file_mask = f"games_*.json"
    merge_json_files(input_folder, output_file, file_mask)
