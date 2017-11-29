"""
Interactive game, Pygame are required
"""

import sys
import argparse

from magent.renderer import PyGameRenderer
from magent.server import ArrangeServer as Server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--message", type=str)
    parser.add_argument("--n_round", type=int, default=1909)
    args = parser.parse_args()

    PyGameRenderer().start(Server(messages=["-iGD'uyPR", 'Jn(#_}'], rnd=args.n_round), grid_size=5)

