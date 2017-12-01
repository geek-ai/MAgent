"""
Show arrange, pygame are required.
Type messages and let agents to arrange themselves to form these characters
"""


import os
import sys

import magent
from magent.renderer import PyGameRenderer
from magent.renderer.server import ArrangeServer as Server

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_arrange.py messages...")
        exit()

    if not os.path.exists("data/arrange_model/arrange/tfdqn_10.index"):
        magent.utility.download_model("https://od.lk/d/NDFfNjAzNTA3OF8/arrange_game.tar.gz")

    PyGameRenderer().start(Server(messages=sys.argv[1:]), grid_size=3.5)
