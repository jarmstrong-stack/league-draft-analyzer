"""
    Lda dataset for training
"""

import random
import torch
from torch.utils.data import Dataset

import constants as CONST
from LDAClass import LDAClass

class LDADataset(Dataset, LDAClass):
    """
    Custom Dataset class for game data.

    Args:
        data (list of dicts): List of game data dictionaries.
        fix_class_discrepency (bool): Flag to know if we should process the dataset evenly
    """

    def __init__(self, data, fix_class_discrepency=True):
        self.data = data
        if fix_class_discrepency:
            self.check_for_result_discrepency()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_data = self.data[idx]
        input_data = self.prepare_input_data(game_data)
        label = float(game_data[CONST.GAMERESULT_DATA])
        return {"data": input_data, "label": torch.tensor(label).to(CONST.DEVICE_CUDA)}

    def prepare_input_data(self, entry:dict):
        """Class to prepare a single entry of `self.data` to tensor type, should be specifically overriden in LDANet"""
        pass

    def check_for_result_discrepency(self):
        """
            Fix any result class differences

            Fixes historic blueside higher winrate, however the model should still
            have knowledge that blueside is stronger
        """
        assert self.data

        blue_win_dataset = list[dict]()
        red_win_dataset = list[dict]()
        for game in self.data:
            if game[CONST.GAMERESULT_DATA] == CONST.BLUE_WIN:
                blue_win_dataset.append(game)
            elif game[CONST.GAMERESULT_DATA] == CONST.RED_WIN:
                red_win_dataset.append(game)

        # Check if wins are evenly present
        # If not we remove from the most present dataset, the oldest entries(patch filter)
        blue_count = len(blue_win_dataset)
        red_count = len(red_win_dataset)
        if blue_count > red_count:
            blue_win_dataset = list(filter(lambda d: d[CONST.PATCH_DATA], blue_win_dataset))
            blue_win_dataset = blue_win_dataset[blue_count-red_count:]
        elif red_count > blue_count:
            red_win_dataset = list(filter(lambda d: d[CONST.PATCH_DATA], red_win_dataset))
            red_win_dataset = red_win_dataset[red_count-blue_count:]
        else:
            return

        # re-define data with even distribuition
        self.data = blue_win_dataset + red_win_dataset
        random.shuffle(self.data)
        assert self.data

