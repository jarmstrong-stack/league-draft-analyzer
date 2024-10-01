"""
    Computes champion pairing synergies

    Takes as input a `league-draft-analyzer` formatted json file and adds inplace
    the computed synergies (win-rate based) of the blue and red team, or creates a new file
    with the data and synergies if given the optional argument

    How to run:
    $ python3 source/compute_synergy.py <input.json> -output_file <optional_output.json>
"""

import sys
import json
import argparse
from collections import defaultdict
from typing import Any

import constants as CONST

### Constants
INPUT_FILE_ARG = "input_file"
OUTPUT_FILE_ARG = "output_file"

# Define the role-specific pairs we're interested in
# (e.g., {Top, Jungle}, {Mid, Jungle}, {ADC, Support})
all_role_pairs = [
    (CONST.TOP_ROLE, CONST.JGL_ROLE),
    (CONST.TOP_ROLE, CONST.MID_ROLE),
    (CONST.TOP_ROLE, CONST.ADC_ROLE),
    (CONST.TOP_ROLE, CONST.SUP_ROLE),

    (CONST.JGL_ROLE, CONST.MID_ROLE),
    (CONST.JGL_ROLE, CONST.ADC_ROLE),
    (CONST.JGL_ROLE, CONST.SUP_ROLE),

    (CONST.MID_ROLE, CONST.ADC_ROLE),
    (CONST.MID_ROLE, CONST.SUP_ROLE),

    (CONST.ADC_ROLE, CONST.SUP_ROLE),
]

main_role_pairs = [
    (CONST.TOP_ROLE, CONST.JGL_ROLE),
    (CONST.MID_ROLE, CONST.JGL_ROLE),
    (CONST.SUP_ROLE, CONST.JGL_ROLE),
    (CONST.SUP_ROLE, CONST.ADC_ROLE),
]

role_pairs = all_role_pairs 

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

    # 4. Compute synergies and counters
    pair_synergy, counter_values = calculate_role_specific_synergy_and_counters(data_to_compute)
    print(f"> Computed synergies: {len(pair_synergy)}")
    print(f"> Computed counters: {len(counter_values)}")

    # 5. Add synergies to data
    for game in data_to_compute:
        add_synergy_and_counters_to_data(game, pair_synergy, counter_values)
    print(f"> Synergies added to data")
        
    # 6. Write data
    with open(parsed_args[OUTPUT_FILE_ARG], 'w') as f:
        json.dump(data_to_compute, f, indent=2)
    print(f"> Wrote data to: {parsed_args[OUTPUT_FILE_ARG]}")

    return 0

def calculate_role_specific_synergy_and_counters(data, laplace_smoothing_factor=1):
    """Calculate champion pair synergy based on win rates and individual counters based on win rates,
       applying Laplace smoothing to avoid extreme win rates with few games."""
    # Dictionary to track win rates for champion pairs (synergy)
    pair_win_count = defaultdict(int)
    pair_game_count = defaultdict(int)

    # Dictionary to track individual role counters (e.g., Top vs Top)
    counter_win_count = defaultdict(int)
    counter_game_count = defaultdict(int)

    # First pass: collect win/loss statistics for both role-specific champion pairs (synergy) and individual role counters
    for game in data:
        blue_team = game[CONST.PICK_DATA][CONST.BLUE_SIDE]
        red_team = game[CONST.PICK_DATA][CONST.RED_SIDE]
        result = game[CONST.GAMERESULT_DATA]

        # Update win and game counts for blue team role-specific champion pairs (synergy)
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

        # Update win and game counts for individual role counters (e.g., Top vs Top)
        for role in [CONST.TOP_ROLE, CONST.JGL_ROLE, CONST.MID_ROLE, CONST.ADC_ROLE, CONST.SUP_ROLE]:
            blue_champ = blue_team[role]
            red_champ = red_team[role]

            counter_game_count[(blue_champ, red_champ)] += 1
            counter_game_count[(red_champ, blue_champ)] += 1

            if result == CONST.BLUE_WIN:
                counter_win_count[(blue_champ, red_champ)] += 1
            elif result == CONST.RED_WIN:
                counter_win_count[(red_champ, blue_champ)] += 1

    # Calculate win rate (synergy) for each pair of champions with Laplace smoothing
    pair_synergy = {}
    for pair, games in pair_game_count.items():
        wins = pair_win_count[pair]
        # Laplace smoothing: Add `laplace_smoothing_factor` virtual wins and losses
        pair_synergy[pair] = (wins + laplace_smoothing_factor) / (games + 2 * laplace_smoothing_factor)

    # Calculate counter rate for individual roles with Laplace smoothing
    counter_rate = {}
    for matchup, games in counter_game_count.items():
        wins = counter_win_count[matchup]
        # Laplace smoothing: Add `laplace_smoothing_factor` virtual wins and losses
        counter_rate[matchup] = (wins + laplace_smoothing_factor) / (games + 2 * laplace_smoothing_factor)

    return pair_synergy, counter_rate

def add_synergy_and_counters_to_data(game, pair_synergy, counter_rate):
    """Calculate synergy and counters for a specific game and add them to data."""

    blue_team = game[CONST.PICK_DATA][CONST.BLUE_SIDE]
    red_team = game[CONST.PICK_DATA][CONST.RED_SIDE]

    # Initialize dictionaries to store role-specific synergies and counters
    blue_synergy = {}
    red_synergy = {}
    blue_counters = {}
    red_counters = {}

    # Calculate the role-specific synergy score for the blue and red team
    for role1, role2 in role_pairs:
        role_key = f"{role1}_{role2}"  # Create a string key for the role pair
        pair_blue = (blue_team[role1], blue_team[role2])
        pair_red = (red_team[role1], red_team[role2])
        sorted_blue_pair = tuple(sorted(pair_blue))
        sorted_red_pair = tuple(sorted(pair_red))

        blue_synergy[role_key] = pair_synergy.get(sorted_blue_pair, 0.0)
        red_synergy[role_key] = pair_synergy.get(sorted_red_pair, 0.0)

    # Calculate the individual role counter values for the blue and red team
    for role in [CONST.TOP_ROLE, CONST.JGL_ROLE, CONST.MID_ROLE, CONST.ADC_ROLE, CONST.SUP_ROLE]:
        blue_champ = blue_team[role]
        red_champ = red_team[role]

        blue_counters[role] = counter_rate.get((blue_champ, red_champ), 0.0)
        red_counters[role] = counter_rate.get((red_champ, blue_champ), 0.0)

    # Add the computed synergies and counters for both teams
    game[CONST.SYNERGY_DATA] = {
        CONST.BLUE_SIDE: {role: round(synergy, 3) for role, synergy in blue_synergy.items()},
        CONST.RED_SIDE: {role: round(synergy, 3) for role, synergy in red_synergy.items()}
    }

    game[CONST.COUNTER_DATA] = {
        CONST.BLUE_SIDE: {role: round(counter, 3) for role, counter in blue_counters.items()},
        CONST.RED_SIDE: {role: round(counter, 3) for role, counter in red_counters.items()}
    }

if __name__ == "__main__":
    ret:int = main()
    assert isinstance(ret, int)
    sys.exit(ret)
