"""
Interactive game, Pygame are required.
Act like a general and dispatch your soilders.
"""

import os

import magent
from magent.renderer import PyGameRenderer
from magent.renderer.server import BattleServer as Server


if __name__ == "__main__":
    if not (os.path.exists("data/battle_model/trusty-battle-game-l/tfdqn_0.index")
            and os.path.exists("data/battle_model/trusty-battle-game-r/tfdqn_0.index")):
        magent.utility.download_model("https://od.lk/d/NDFfNjA2MTU1N18/battle_model.tar.gz")

    PyGameRenderer().start(Server())
