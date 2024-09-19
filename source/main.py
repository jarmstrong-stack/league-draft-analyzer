"""
    LDA entry-point
"""

from Normalizer import Normalizer

def main(args:dict) -> int:
    """This is the function that is called on every process after driver.py"""
    d = {"a": "1", "b": "2"}
    f = ["a", "b"]
    n = Normalizer(f)
    p = n.normalize(d)
    return 0
