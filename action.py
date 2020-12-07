# sys

# 3rd
import numpy as np

# ours
from .src.pymod.model import create_model
from .src.pymod.mytypes import TileType
from .src.cymod.feed import Feed


class Action :
    def __init__(self) :
        self.model = create_model()
        self.feed = Feed(1)

    def decide_which_tile_to_discard(self, game, players, player_num) -> (int, bool) :
        self.feed.write_feed_x(game, players, player_num)
        pred = self.model.predict_on_batch(self.feed.feed_x)
        self.feed.clear_feed()

        indexes = np.argsort(-pred[0])
        player = players[player_num]

        for i in indexes :
            if (player.hand[i] > 0) or (i in TileType.REDS and player.reds[i // 10] and player.hand[i+5] > 0) :
                if player.last_got_tile == i : exchanged = True
                else :exchanged = False
                return i, exchanged

