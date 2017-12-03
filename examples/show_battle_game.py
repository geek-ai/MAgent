"""
Interactive game, Pygame are required.
Act like a general and dispatch your solders.
"""

import os

import magent
from magent.renderer import PyGameRenderer
from magent.renderer.server import BattleServer as Server


if __name__ == "__main__":
    magent.utility.check_model('battle-game')
    PyGameRenderer().start(Server())
