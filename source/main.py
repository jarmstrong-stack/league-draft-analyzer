"""
    LDA entry-point
"""

import logging
logger = logging.getLogger(__name__)

import constants as CONST 
import test_data as TEST
from LDANet import LDANet

def main(args:dict) -> int:
    """This is the function that is called on every process after driver.py"""
    net = LDANet().to(CONST.DEVICE_CUDA)

    match args[CONST.DRIVER_ACTION]:
        case CONST.DRIVER_TRAIN:
            net.train_lda()
        case CONST.DRIVER_PREDICT:
            net.load_lda(CONST.LDA_WEIGHTS_PATH)
            x = net(net.handle_prediction_data(TEST.PCS_GAME))
            print(x)
        case None:
            logger.critical(f"No \"{CONST.DRIVER_ACTION}\" was provided in args.")
    return 0
