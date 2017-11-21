"""
Interactive game, Pygame are required
"""

from magent.renderer import PyGameRenderer
from magent.server import BattleServer as Server

if __name__ == "__main__":
    PyGameRenderer().start(Server())
