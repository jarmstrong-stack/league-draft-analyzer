"""
    Helper code to join large ammounts of `league-draft-analyzer` data
    into a single file
"""

import json
import glob

def merge_json_files(input_folder, output_file):
    """Function to load all json files and combine the entries"""
    all_data = []

    # Get all JSON files in the input folder
    json_files = glob.glob(f"{input_folder}/games_*.json")

    # Loop through each file and load its data
    for file in json_files:
        print(f"> Loading {file}...")
        with open(file, 'r') as f:
            data = json.load(f)
            # Check if the file contains a list of entries
            if isinstance(data, list):
                all_data.extend(data)  # Add entries to the combined list

    # Write combined data to a new JSON file
    print(f"> Finished... Exporting to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Successfully merged {len(json_files)} files into {output_file}")

# Usage
input_folder = 'data'
output_file = 'data/all_games.json'
merge_json_files(input_folder, output_file)
