"""
Interactive game, Pygame are required.
Act like a general and dispatch your soilders.
"""

from magent.renderer import PyGameRenderer
from magent.server import BattleServer as Server

if __name__ == "__main__":
    PyGameRenderer().start(Server())
