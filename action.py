# sys
from typing import List

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
        if player.has_right_to_one_shot : hand = player.hand[:]
        for tile in indexes :
            if (player.hand[tile] > 0) or (tile in TileType.REDS and player.reds[tile // 10] and player.hand[tile+5] > 0) :
                # 赤5しか持ってないのに黒5を切ろうとするとcontinue
                if tile in TileType.FIVES and player.hand[tile] == 1 and player.reds[tile // 10] : continue

                # 立直宣言したのに聴牌しない打牌をしようとするとcontinue
                if player.has_right_to_one_shot :
                    hand[tile] -= 1
                    shanten_nums = game.shanten_calculator.get_shanten_nums(hand, 0)
                    hand[tile] += 1
                    ready = False
                    for shanten_num in shanten_nums :
                        if shanten_num <= 0 : ready = True
                    if not(ready) : continue

                # ツモ切りかどうかを判定
                if player.last_got_tile == tile : exchanged = False
                else : exchanged = True

                return tile, exchanged


    # 鳴くかどうか決める
    # TODO 赤含みで鳴くかどうかまで決められてない
    def decide_to_steal(self, game, players, tile, pos, player_num) -> (int, int, int, int, int, int) :
        self.feed.write_feed_x(game, players, player_num)
        self.feed.write_about_steal_info(tile, pos)
        pred = self.steal_model.predict(self.feed.feed_x)
        self.feed.clear_feed()

        indexes = tuple(np.argsort(-pred[1][0]))
        return indexes


    # リーチするかどうか決める
    # TODO ちゃんと書く
    def decide_to_declare_ready(self, game, players, player_num) -> bool :
        return True


    # 槓するかどうか決める
    # TODO ちゃんと書く
    def decide_to_kan(self, game, players, player_num:int, ankan_tiles:List[int], kakan_tiles:List[int]) -> int :
        return -1,


    # 九種九牌を宣言するかどうか決める
    # TODO ちゃんと書く
    def decide_to_declare_nine_orphans(self, game, players, player_num:int, hand:List[int]) -> bool :

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
    def decide_win(self, game, players, player_num:int) -> True :
        return True


