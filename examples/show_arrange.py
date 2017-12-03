"""
Show arrange, pygame are required.
Type messages and let agents to arrange themselves to form these characters
"""


import os
import sys
import argparse
import magent
from magent.renderer import PyGameRenderer
from magent.renderer.server import ArrangeServer as Server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="0: without maze, 1: adding a maze")
    parser.add_argument("--mess", type=str, nargs="+", help="words you wanna print", required=True)
    args = parser.parse_args()

    magent.utility.check_model('arrange')

    PyGameRenderer().start(Server(messages=args.mess, mode=args.mode), grid_size=3.5)
