# sys

# 3rd
import numpy as np

# ours
from .src.pymod.model import create_model
from .src.pymod.mytypes import TileType
from .src.cymod.feed import Feed


class Action :
    def __init__(self) :
        self.main_model  = create_model("main")
        self.steal_model = create_model("steal")
        self.feed = Feed(1)


    # 切る牌を決める
    def decide_which_tile_to_discard(self, game, players, player_num) -> (int, bool) :
        self.feed.write_feed_x(game, players, player_num)
        pred = self.main_model.predict(self.feed.feed_x)
        self.feed.clear_feed()

        player = players[player_num]
        indexes =np.argsort(-pred[0][0])
        for i in indexes :
            if (player.hand[i] > 0) or (i in TileType.REDS and player.reds[i // 10] and player.hand[i+5] > 0) :
                if i in TileType.FIVES and player.hand[i] == 1 and player.reds[i // 10] : continue
                if player.last_got_tile == i : exchanged = False
                else : exchanged = True
                return i, exchanged


    # 鳴くかどうか決める
    def decide_to_steal(self, game, players, tile, pos, player_num) -> (int, int, int, int, int, int) :
        self.write_feed_x(game, players, player_num)
        self.write_about_steal_info(tile, pos)
        pred = self.steal_model.predict(self.feed.feed_x)
        self.feed.clear_feed()

        indexes = tuple(np.argsort(-pred[0][0]))
        return indexes


    # リーチするかどうか決める
    # TODO ちゃんと書く
    def decide_to_declare_ready(self, game, players, player_num) -> bool :
        return True


