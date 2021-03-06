# sys
import sys
from typing import List

# 3rd
import numpy as np

# ours
from .src.pymod.model import create_model
from .src.pymod.mytypes import TileType
from .src.cymod.feed import Feed


class Action :
    def __init__(self) :
        self.main_feed = Feed("main", 1)
        self.main_model  = create_model("main")

        self.steal_feed = Feed("steal", 1)
        self.steal_model = create_model("steal")

        self.ready_feed = Feed("ready", 1)
        self.ready_model = create_model("ready")


    # 切る牌を決める
    def decide_which_tile_to_discard(self, game, players, player_num) -> (int, bool) :
        self.main_feed.write_feed_x(game, players, player_num)
        pred = self.main_model.predict(self.main_feed.feed_x)
        self.main_feed.clear_feed()

        player = players[player_num]
        indexes = np.argsort(-pred[0])
        for tile in indexes :
            if (player.hand[tile] > 0) or (tile in TileType.REDS and player.reds[tile // 10] and player.hand[tile+5] > 0) :
                # 赤5 1枚しか持ってないのに黒5を切ろうとするとcontinue
                if tile in TileType.FIVES and player.hand[tile] == 1 and player.reds[tile // 10] : continue

                # 立直宣言したのに聴牌しない打牌をしようとするとcontinue
                if player.has_right_to_one_shot :
                    player.hand[tile] -= 1
                    shanten_nums = game.shanten_calculator.get_shanten_nums(player.hand, 0)
                    player.hand[tile] += 1
                    ready = False
                    for shanten_num in shanten_nums :
                        if shanten_num <= 0 : ready = True
                    if not(ready) : continue

                # ツモ切りかどうかを判定
                # TODO 手出し出来る時でも強制ツモ切りしているのでちゃんとNNが選ぶようにしたい
                if player.last_got_tile == tile : exchanged = False
                else : exchanged = True

                return tile, exchanged


    # 鳴くかどうか決める
    # TODO 赤含みで鳴くかどうかまでAIが決められてない．現状赤含みにできたら全て赤含みにしている．
    # TODO? 一番値が大きかったやつしか返していない．4,3,5,...みたいなときに4(嵌張チー)ができなかったときに3ではなく0を返す．
    def decide_to_steal(self, game, players, tile, pos, player_num) -> (int, int, int, int, int, int, int, int) :
        self.steal_feed.write_feed_x(game, players, player_num)
        self.steal_feed.write_about_steal_info(tile, pos)
        pred = self.steal_model.predict(self.steal_feed.feed_x)
        self.steal_feed.clear_feed()

        indexes = np.argsort(-pred[0])
        action = indexes[0]
        tile1, tile2 = -1, -1

        # チーの場合どの牌で鳴くかも返す
        if action in (3,4,5) :
            if tile in TileType.REDS : tile += 5

            if   action == 3 : tile1, tile2 = tile - 2, tile - 1
            elif action == 4 : tile1, tile2 = tile - 1, tile + 1
            elif action == 5 : tile1, tile2 = tile + 1, tile + 2

            if players[player_num].reds[tile1 // 10] : tile1 -= 5
            elif players[player_num].reds[tile2 // 10] : tile2 -= 5

        return action, 0, 0, 0, 0, 0, tile1, tile2


    # リーチするかどうか決める
    def decide_to_declare_ready(self, game, players, player_num) -> bool :
        self.ready_feed.write_feed_x(game, players, player_num)
        pred = self.main_model.predict(self.ready_feed.feed_x)
        self.ready_feed.clear_feed()

        if pred[0][0] > pred[0][1] : return False
        return True


    # 槓するかどうか決める
    # TODO ちゃんと書く
    def decide_to_kan(self, game, players, player_num, ankan_tiles, kakan_tiles) -> int :
        return -1,


    # 九種九牌を宣言するかどうか決める
    # TODO ちゃんと書く
    def decide_to_declare_nine_orphans(self, game, players, player_num, hand) -> bool :

        # 九種九牌をそもそも宣言できるかどうかの判定
        terminals_num = 0
        for i in (TileType.TERMINALS | TileType.HONORS) :
            if hand[i] > 0 :
                terminals_num += 1

        # 今はとりあえず流局できたらするようにする
        if terminals_num > 8 : return True

        return False


    # 和了るかどうか決める
    # TODO ちゃんと書く
    def decide_win(self, game, players, player_num) -> True :
        return True


