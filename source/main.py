"""
    LDA entry-point
"""

from Normalizer import Normalizer
from LDANet import LDANet
import constants as CONST
import test_data as TEST

def main(args:dict) -> int:
    """This is the function that is called on every process after driver.py"""
    net = LDANet()
    return 0
