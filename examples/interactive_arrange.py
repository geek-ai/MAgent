"""
Interactive game, Pygame are required
"""

import sys

from magent.renderer import PyGameRenderer
from magent.server import ArrangeServer as Server

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interactive_arrange.py messages")
        exit()
    PyGameRenderer().start(Server(messages=sys.argv[1:]), grid_size=5)

