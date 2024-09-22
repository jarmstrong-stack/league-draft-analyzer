"""
    LDA deep neural network module
"""

import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import train
import compute_synergy as CS
import parse_game_data as PG
import constants as CONST
from Normalizer import Normalizer
from LDAClass import LDAClass

class LDANet(nn.Module, LDAClass):
    """
    A neural network class designed for analyzing League of Legends drafts using 
    champion picks, bans, synergy differences, and other game features. The 
    network aims to predict match outcomes based on this input data.

    Attributes:
    -----------
    champ_mapping : dict
        A dictionary mapping champion names (or IDs) to integers for use in the model.
    
    synergy_values : dict
        A dictionary storing pre-computed synergy values for each champion pair in the dataset.
    
    game_data : list[dict]
        A list of dictionaries where each entry represents game data, including picks, bans, 
        synergies, patches, and results.
    
    normalizer : Normalizer
        An instance of the Normalizer class responsible for normalizing input features such as 
        picks, bans, and synergies before they are passed to the neural network.
    
    features_to_process : list[str]
        A list of feature types (e.g., picks, bans, synergy, patch) that are expected to be 
        processed and fed into the neural network.
    
    feature_input_size : dict[str, int]
        A dictionary defining the input size for each feature type, ensuring that the model 
        expects the correct number of inputs for each feature.
    
    input_size : int
        The total size of the input layer for the neural network, calculated by summing the 
        input sizes of the features to be processed.

    Methods:
    --------
    __init__(*args, **kwargs)
        Initializes the LDANet class by loading champion mappings, game data, synergy values, 
        and defining the neural network architecture.
    
    define()
        Defines the architecture of the neural network, consisting of several fully connected 
        layers, ReLU activations, dropout layers for regularization, and a sigmoid output for 
        binary classification.
    
    forward(x)
        The forward pass of the network, applying the defined layers to the input data.
    
    train_lda()
        Prepares the dataset for training and runs the training process using the `train_model` 
        function.
    
    load_lda(weights_path:str)
        Loads pre-trained weights from a given file path into the network's state dictionary.
    
    champion_count()
        Returns the number of unique champions found in the loaded data.
    
    handle_prediction_data(data:dict)
        Preprocesses the input data by ensuring it is properly formatted and normalized. This 
        includes translating champion names into their respective integer representations, 
        computing missing synergy values, padding missing bans, and converting data into tensors.
    
    pad_or_trim_list(lst:list, target_size:int, pad_value=0)
        Pads or trims a list to ensure it has the target size, filling any missing values with 
        `pad_value`.
    
    load_champ_mapping()
        Loads champion-to-integer mappings from a YAML file.
    
    load_game_data()
        Loads all game data from a JSON file into memory.
    
    compute_synergy_values()
        Pre-computes role-specific synergy values based on the loaded game data.
    
    compute_input_size()
        Calculates the total size of the input layer for the network by summing the sizes of 
        the selected features to be processed.
    """

    # Neural net data
    champ_mapping: dict
    synergy_values: dict
    game_data: list[dict]

    # Feature normalization
    normalizer: Normalizer
    features_to_process: list[str] = [CONST.PICK_DATA, CONST.BAN_DATA, CONST.SYNERGY_DATA, CONST.PATCH_DATA]
    feature_input_size: dict[str, int] = { # How much does each feature take to input
        CONST.PICK_DATA: 10,
        CONST.BAN_DATA: 10,
        CONST.TOURNAMENT_DATA: 1,
        CONST.GAMETIME_DATA: 1,
        CONST.PATCH_DATA: 1,
        CONST.TEAMS_DATA: 2,
        CONST.GAMEDATE_DATA: 1,
        CONST.SYNERGY_DATA: 1,
        CONST.GAMERESULT_DATA: 1,
    }

    def __init__(self, *args, **kwargs) -> None:
        super(LDANet, self).__init__(*args, **kwargs)
        
        # Instantiate champ_mapping, game_data, synergy_values
        self.load_champ_mapping()
        self.load_game_data()
        self.compute_synergy_values()
        
        # Instantiate data normalizer
        self.normalizer = Normalizer(self.features_to_process)
        
        # Calculate neural net input size
        self.compute_input_size()
        
        # Define neural net
        self.define()

    def define(self):
        """Defines neural network architecture"""
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        
        # Output layer
        self.output = nn.Linear(16, 1)
        
        # Dropout layers for regularization to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """Forward pass neural net"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        
        x = torch.sigmoid(self.output(x))
        return x

    def train_lda(self):
        """Train lda net"""
        # Setup dataset for training
        lda_dataset = train.GameDataset(self.game_data)
        lda_dataset.prepare_input_data = self.handle_prediction_data # pyright: ignore
        
        # Start training
        self.logger.info(f"Starting to train on {len(self.game_data)} games...")
        train.train_model(self, lda_dataset)

    def load_lda(self, weights_path:str):
        """Loads lda weights"""
        self.load_state_dict(torch.load(weights_path, weights_only=True))
        self.logger.info(f"Successefully loaded weights from \"{weights_path}\"")

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
        # If its already a Tensor, don't bother doing anything
        if isinstance(data, torch.Tensor):
            return data
        
        # Check for champ name instead of integer
        # This will automatically turn ANY champ name(str)
        # Into their respective int number, it supports some champs being str and others int
        try:
            PG.parse_champs_helper(data[CONST.PICK_DATA][CONST.BLUE_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.PICK_DATA][CONST.RED_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.BAN_DATA][CONST.BLUE_SIDE], self.champ_mapping)
            PG.parse_champs_helper(data[CONST.BAN_DATA][CONST.RED_SIDE], self.champ_mapping)
        except KeyError:
            pass
        
        # Check for missing synergy data
        if CONST.SYNERGY_DATA not in data:
            CS.add_synergy_to_data(data, self.synergy_values)
        
        # Check for missed bans, some games can have 
        data[CONST.BAN_DATA][CONST.BLUE_SIDE] = self.pad_or_trim_list(data[CONST.BAN_DATA][CONST.BLUE_SIDE], 5, 0)
        data[CONST.BAN_DATA][CONST.RED_SIDE] = self.pad_or_trim_list(data[CONST.BAN_DATA][CONST.RED_SIDE], 5, 0)
        
        # Check for not tensor parsed data
        if not isinstance(data[CONST.PICK_DATA][CONST.BLUE_SIDE], torch.Tensor):
            data = self.normalizer.normalize(data)
        
        # Transform data in a tensor
        np_data = np.concatenate([
            data[CONST.PICK_DATA][CONST.BLUE_SIDE],
            data[CONST.PICK_DATA][CONST.RED_SIDE],
            data[CONST.BAN_DATA][CONST.BLUE_SIDE],
            data[CONST.BAN_DATA][CONST.RED_SIDE],
            [data[CONST.SYNERGY_DATA]],
            [data[CONST.PATCH_DATA]]
        ])
        data = torch.Tensor(np_data).to(CONST.DEVICE_CUDA) # pyright: ignore
        
        return data

    def pad_or_trim_list(self, lst:list, target_size:int, pad_value=0):
        """Pad or trim a list to ensure it's of a consistent size"""
        if len(lst) < target_size:
            return lst + [pad_value] * (target_size - len(lst))
        return lst[:target_size]

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

    def compute_input_size(self):
        """Compute neural net initial input size given the features to input"""
        self.input_size = 0
        for feature in self.features_to_process:
            self.input_size += self.feature_input_size[feature]
