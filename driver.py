"""
    Main driver

    Sets up logs & arguments
"""

import source.constants as CONST
import argparse

# Add source/ to syspath
import sys
sys.path.append('./source/')

# Logging
import logging, coloredlogs
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

# commandline args
def parse_args() -> dict:
    """Parse required args for script"""
    parser = argparse.ArgumentParser("league-draft-analyzer")
    parser.add_argument(f'-{CONST.DRIVER_ACTION}', help="Action for the driver to take.", type=str)
    args = parser.parse_args()
    return {
        CONST.DRIVER_ACTION: getattr(args, CONST.DRIVER_ACTION)
    }

# driver
if __name__ == "__main__":
    driver_args = parse_args()
    logger.info(f"Script arguments: {driver_args}")

    from source.main import main
    ret:int = main(driver_args)
    assert isinstance(ret, int)
    sys.exit(ret)
