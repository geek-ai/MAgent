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
    # if len(sys.argv) < 2:
    #     print("Usage: python show_arrange.py messages...")
    #     exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="0: without maze, 1: adding a maze")
    parser.add_argument("--mess", type=str, nargs="+", help="words you wanna print", required=True)
    args = parser.parse_args()

    if not os.path.exists("data/arrange_model/arrange/tfdqn_10.index"):
        magent.utility.download_model("https://od.lk/d/NDFfNjAzNTA3OF8/arrange_game.tar.gz")

    PyGameRenderer().start(Server(messages=args.mess, mode=args.mode), grid_size=3.5)
