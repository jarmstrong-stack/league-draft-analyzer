"""
    Computes champion pairing synergies

    Takes as input a `league-draft-analyzer` formatted json file and adds inplace
    the computed synergies (win-rate based) of the blue and red team
"""

import sys
import json
import argparse
from itertools import combinations
from collections import defaultdict
from typing import Any

import constants as CONST

### Constants
INPUT_FILE_ARG = "input_file"
OUTPUT_FILE_ARG = "output_file"

def parse_args() -> dict[str,Any]:
    """Parse required args for script"""
    parser = argparse.ArgumentParser("compute_synergy_py")
    parser.add_argument(INPUT_FILE_ARG, help="Which json file to compute synergies.", type=str)
    parser.add_argument(f"-{OUTPUT_FILE_ARG}", help="(Optional) Which json file to output to.", type=str, default=None)
    args = parser.parse_args()

    return {
        INPUT_FILE_ARG: getattr(args, INPUT_FILE_ARG),
        OUTPUT_FILE_ARG: getattr(args, OUTPUT_FILE_ARG)
    }

def main():
    """Entry point of script
        1. Get info from arguments
        2. Initialize
        3. Read data
        4. Compute synergies
        5. Add synergies to data
        6. Write data
    """

    # 1. Get info from argumentsa
    parsed_args:dict = parse_args()
    if parsed_args[OUTPUT_FILE_ARG] is None:
        parsed_args[OUTPUT_FILE_ARG] = parsed_args[INPUT_FILE_ARG]
    print(f"> Parsed args: {parsed_args}")

    # 2. Initialize
    data_to_compute = list()

    # 3. Read data
    with open(parsed_args[INPUT_FILE_ARG], 'r') as f:
        data_to_compute = json.load(f)
    if len(data_to_compute) == 0:
        raise ValueError("Input file given has no info/Could not be read.")
    print(f"> Read input file... len={len(data_to_compute)}")

    # 4. Compute synergies
    pair_synergy = calculate_role_specific_synergy(data_to_compute)
    print(f"> Computed synergies: {len(pair_synergy)}")

    # 5. Add synergies to data
    updated_data = add_synergy_to_data(data_to_compute, pair_synergy)
    print(f"> Synergies added to data")
    
    # 6. Write data
    with open(parsed_args[OUTPUT_FILE_ARG], 'w') as f:
        json.dump(updated_data, f, indent=2)
    print(f"> Wrote data to: {parsed_args[OUTPUT_FILE_ARG]}")

    return 0

def calculate_role_specific_synergy(data):
    """Calculate champion pair synergy based on win rates"""
    # Dictionary to track win rates for champion pairs
    pair_win_count = defaultdict(int)
    pair_game_count = defaultdict(int)

    # Define the role-specific pairs we're interested in
    # (e.g., {Top, Jungle}, {Mid, Jungle}, {ADC, Support})
    role_pairs = [
        ("1", "2"),  # Top and Jungle
        ("2", "3"),  # Jungle and Mid
        ("4", "5")   # ADC and Support
    ]

    # First pass: collect win/loss statistics for each role-specific champion pair
    for game in data:
        blue_team = game[CONST.PICK_DATA][CONST.BLUE_SIDE]
        red_team = game[CONST.PICK_DATA][CONST.RED_SIDE]
        result = game[CONST.GAMERESULT_DATA] 

        # Update win and game counts for blue team role-specific champion pairs
        for role1, role2 in role_pairs:
            pair_blue = (blue_team[role1], blue_team[role2])
            pair_red = (red_team[role1], red_team[role2])

            sorted_blue_pair = tuple(sorted(pair_blue))
            sorted_red_pair = tuple(sorted(pair_red))

            pair_game_count[sorted_blue_pair] += 1
            pair_game_count[sorted_red_pair] += 1

            if result == CONST.BLUE_WIN:
                pair_win_count[sorted_blue_pair] += 1
            elif result == CONST.RED_WIN:
                pair_win_count[sorted_red_pair] += 1

    # Calculate win rate (synergy) for each pair of champions
    pair_synergy = {}
    for pair, games in pair_game_count.items():
        wins = pair_win_count[pair]
        pair_synergy[pair] = wins / games if games > 0 else 0.0

    return pair_synergy

# Add synergy values to each game in the dataset
def add_synergy_to_data(data, pair_synergy):
    for game in data:
        blue_team = list(game[CONST.PICK_DATA][CONST.BLUE_SIDE].values())
        red_team = list(game[CONST.PICK_DATA][CONST.RED_SIDE].values())

        # Calculate the synergy score for the blue team
        blue_synergy = 0.0
        for pair in combinations(blue_team, 2):
            sorted_pair = tuple(sorted(pair))
            blue_synergy += pair_synergy.get(sorted_pair, 0.0)

        # Calculate the synergy score for the red team
        red_synergy = 0.0
        for pair in combinations(red_team, 2):
            sorted_pair = tuple(sorted(pair))
            red_synergy += pair_synergy.get(sorted_pair, 0.0)

        # Add synergy to the game data
        game[CONST.SYNERGY_DATA] = {
            CONST.BLUE_SIDE: round(blue_synergy, 3),
            CONST.RED_SIDE: round(red_synergy, 3)
        }

    return data

if __name__ == "__main__":
    ret:int = main()
    assert isinstance(ret, int)
    sys.exit(ret)
