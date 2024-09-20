"""
    LDA deep neural network module
"""

import torch, torch.nn as nn

import constants as CONST
from LDAClass import LDAClass

class LDANet(nn.Module, LDAClass):
    """
        Actual neural network for `league-draft-analyzer`
    """

    def __init__(self) -> None:
        super(LDANet, self).__init__()

        self.define()

    def define(self):
        """Defines neural network architecture"""
        ...
