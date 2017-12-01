"""
Interactive game, Pygame are required.
Act like a general and dispatch your soilders.
"""

import os

import magent
from magent.renderer import PyGameRenderer
from magent.server import BattleServer as Server


if __name__ == "__main__":
    if not (os.path.exists("data/battle_model/trusty-l/tfdqn_0.index")
            and os.path.exists("data/battle_model/trusty-r/tfdqn_0.index")):
        magent.utility.download_model("https://od.lk/s/NDFfNjAzNDk0OV8/battle_game.tar.gz")

    PyGameRenderer().start(Server())
