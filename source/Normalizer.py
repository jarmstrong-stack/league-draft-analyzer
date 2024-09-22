"""
    Data normalizer & pre processer for `league-draft-analyzer`

    Will handle the data before being inputted into LDA neural net
"""

import torch

import constants as CONST
from LDAClass import LDAClass

class Normalizer(LDAClass):
    """
        Handles normalizing and preprocessing data dinamically

        Its purpose is to take in some data of type `dict` and normalize some features
        specified in `features_to_process`.

        It will return a new processed dict with ONLY the features specified.

        Each feature's function to normalize is written the same as the feature name.
        (eg. "picks" will be normalized by self.picks(data))
    """

    features_to_process: list[str]

    def __init__(self, features_to_process: list[str]) -> None:
        self.features_to_process = features_to_process

    def get_processor(self, feature: str):
        """Returns the processor for `feature`, does that by looking into `self` attrs."""
        try:
            processing_function = getattr(self, feature)
            if not callable(processing_function):
                self.logger.error(f"Invalid preprocessor type {type(processing_function)} for feature {feature}.")
                return None
        except AttributeError as e:
            self.logger.error(f"No preprocessor found for feature {e.name}.")
            return None
        return processing_function

    def normalize(self, data:dict) -> dict:
        """Main function that normalizes given `data` and calls each preprocessing function"""
        normalized_data = dict()

        for feature in self.features_to_process:
            processing_function = self.get_processor(feature)
            if processing_function is None: # Means there is no preprocessor for this feature
                self.logger.warning(f"Skipped processing for feature {feature}.")
                continue

            normalized_feature = processing_function(data)
            normalized_data[feature] = normalized_feature

        return normalized_data

    def pick(self, data:dict):
        """picks preprocessor (dict to list)"""
        blue_picks = [data[CONST.PICK_DATA][CONST.BLUE_SIDE][str(i)] for i in range(1, 6)]
        red_picks = [data[CONST.PICK_DATA][CONST.RED_SIDE][str(i)] for i in range(1, 6)]
        return {
            CONST.BLUE_SIDE: torch.tensor(blue_picks), CONST.RED_SIDE: torch.tensor(red_picks)
        }

    def ban(self, data:dict):
        """bans preprocessor"""
        return {
            CONST.BLUE_SIDE: torch.tensor(data[CONST.BAN_DATA][CONST.BLUE_SIDE]),
            CONST.RED_SIDE: torch.tensor(data[CONST.BAN_DATA][CONST.RED_SIDE])
        }

    def patch(self, data:dict):
        """patch preprocessor"""
        return torch.tensor(data[CONST.PATCH_DATA])

    def synergy(self, data:dict):
        """synergy preprocessor (red synergy - blue synergy)"""
        return torch.tensor(round(data[CONST.SYNERGY_DATA][CONST.RED_SIDE] - data[CONST.SYNERGY_DATA][CONST.BLUE_SIDE], 3))

