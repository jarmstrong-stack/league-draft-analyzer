"""
    LDA deep neural network module
"""

import json
import yaml
import torch, torch.nn as nn

import compute_synergy as CS
import parse_game_data as PG
import constants as CONST
from Normalizer import Normalizer
from LDAClass import LDAClass

class LDANet(nn.Module, LDAClass):
    """
        Actual neural network for `league-draft-analyzer`

        Loads champ_mapping str to int database.
        Loads ALL game data into memory.
        Pre-computes synergy values.

        `self.champion_count` = total number of champions that were found in data
    """

    # Neural net data
    champ_mapping: dict
    synergy_values: dict
    game_data: list[dict]

    # Feature normalization
    features_to_process: list[str] = [CONST.PICK_DATA, CONST.BAN_DATA, CONST.SYNERGY_DATA, CONST.PATCH_DATA]
    normalizer: Normalizer
    
    def __init__(self, *args, **kwargs) -> None:
        super(LDANet, self).__init__(*args, **kwargs)

        # Instantiate champ_mapping, game_data, synergy_values
        self.load_champ_mapping()
        self.load_game_data()
        self.compute_synergy_values()

        # Instantiate data normalizer
        self.normalizer = Normalizer(self.features_to_process)

        # Define neural net
        self.define()

    @property
    def champion_count(self) -> int:
        """Number of champions found in loaded data"""
        if self.champ_mapping:
            return len(self.champ_mapping)
        return 0       

    def handle_prediction_data(self, data:dict):
        """Handles data before being inputted to the net, fixes bad data
            Translate champion names if isinstance(champ, str)
            Compute synergies if it doesnt have it
            And normalize the data if its not normalized

            Changes data inplace
        """
        self.logger.info(f"Checking for wrongly formatted data:\n{data}")

        # Check for champ name instead of integer
        try:
            # This code makes me wanna marry it, this will automatically turn ANY champ name(str)
            # Into their respective int number, it supports some champs being str and others int
            PG.parse_champs_helper(data[CONST.PICK_DATA][CONST.BLUE_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.PICK_DATA][CONST.RED_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.BAN_DATA][CONST.BLUE_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.BAN_DATA][CONST.RED_SIDE], self.champ_mapping)
        except KeyError:
            pass

        # Check for missing synergy
        if CONST.SYNERGY_DATA not in data:
            CS.add_synergy_to_data(data, self.synergy_values)
        
        # Check for not tensor parsed data
        if not isinstance(data[CONST.PICK_DATA][CONST.BLUE_SIDE], torch.Tensor):
            data = self.normalizer.normalize(data)

        self.logger.info(f"Data after checks:\n{data}")

    def load_champ_mapping(self):
        """Load champ mapping to memory"""
        with open(CONST.CHAMP_TO_INT_DATABASE, 'r', encoding='utf-8') as f:
            yml_content = f.read()
            self.champ_mapping = yaml.safe_load(yml_content)
        self.logger.info(f"Loaded champ mapping... Champions found={len(self.champ_mapping)}")
       
    def load_game_data(self):
        """Loads all game data"""
        with open(CONST.GAME_DATABASE, 'r') as f:
            self.game_data = json.load(f)
        self.logger.info(f"Loaded game data... len={len(self.game_data)}")

    def compute_synergy_values(self):
        """Pre-computes synergy values"""
        if self.game_data == None or len(self.game_data) == 0:
            self.logger.critical("Could not compute synergy values: No game data found.")
            return
        self.synergy_values = CS.calculate_role_specific_synergy(self.game_data)
        self.logger.info(f"Loaded synergy values... len={len(self.synergy_values)}")

    def define(self):
        """Defines neural network architecture"""
        ...
