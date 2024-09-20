"""
    LDA entry-point
"""

from Normalizer import Normalizer
from LDANet import LDANet
import constants as CONST
import test_data as TEST

def main(args:dict) -> int:
    """This is the function that is called on every process after driver.py"""
    d = TEST.PCS_GAME
    f = [CONST.PICK_DATA, CONST.BAN_DATA, CONST.SYNERGY_DATA, CONST.PATCH_DATA]
    n = Normalizer(f)
    p = n.normalize(d)
    
    net = LDANet()
    return 0
