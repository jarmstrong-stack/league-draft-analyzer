"""
    Base class of all `league-draft-analyzer` related classes
"""

import logging
local_logger = logging.getLogger(__name__)

class LDAClass:
    """Base class for all `league-draft-analyzer` related classes.
        Handles standardized(idk how to spell xdd) logging & other generic stuff
    """

    def __new__(cls, *args, **kwargs):
        cls.logger = logging.getLogger(cls.__name__)
        local_logger.info("Instantiating class %s", cls.__name__)
        
        instance = super(LDAClass, cls).__new__(cls)
        return instance
