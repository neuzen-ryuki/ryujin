#std
import random
from typing import List

# 3rd

# ours
from mytypes import BlockType, TileType


class Player :
    def __init__(self, player_num) :
        self.player_num = player_num                      # プレイヤ番号 スタート時の席と番号の関係(0:起家, 1:南家, 2:西家, 3:北家)
        self.exists = True
        self.score = 25000                                # 点棒


    # 局の初期化
    def init_subgame(self, rotations_num:int) -> None :
        self.hand = [0] * 38                              # 手牌
        self.reds = [False] * 3                           # 自分の手に赤があるかどうか，マンピンソウの順．例）reds[1] is True ==> 手の中に赤5pがある
        self.opened_hand = [0] * 20                       # 副露牌
        self.opened_reds = [False] * 3                    # 副露牌に赤が含まれているか
        self.discarded_tiles = []                         # 河
        self.discarded_tiles_hist = [0] * 38              # 枚数だけ記録する河
        self.discarded_state = []                         # D:ツモ切り，d:手出し , i:"D" ==> discarded_tiles[i]はツモ切られた牌
        self.furiten_tiles = [0] * 38                     # フリテン牌 furiten_tiles[i] > 0 ==> i番の牌は既に自分で切っているか同巡に切られた牌
        self.same_turn_furiten_tiles = []                 # 同巡に切られた牌

        self.has_stealed = False                          # 1度でもポン, ダイミンカン, チーしていればTrueになる
        self.has_declared_ready = False                   # リーチを宣言しているか
        self.has_declared_double_ready = False            # ダブルリーチを宣言しているか
        self.has_right_to_one_shot = False                # 一発があるか
        self.has_right_to_nagashi_mangan = True           # 流し満貫継続中か
        self.is_ready = False                             # テンパイか

        self.opened_sets_num = 0                          # 晒している面子の数(暗槓含む)
        self.kans_num = 0                                 # 槓した回数

        self.players_wind = 31 + (((self.player_num+4) - rotations_num)%4)
        self.last_got_tile = -1                           # 直近でツモった牌 赤牌は番号そのまま(10:赤5筒)
        self.last_discarded_tile = -1                     # 最後に切った牌 赤牌は番号そのまま(10:赤5筒)

        self.dragons_num = 0                              # 大三元パオ判定用
        self.winds_num = 0                                # 大四喜パオ判定用


    # 立直宣言
    def declare_ready(self, is_first_turn:bool) -> None :
        if is_first_turn : self.has_declared_double_ready = True
        else : self.has_declared_ready = True
        self.has_right_to_one_shot = True
        self.score -= 1000


    # 牌を手牌に加える．配牌取得，ツモ，ロンで使う．
    def get_tile(self, tile:int) -> None :
        self.last_got_tile = tile
        if tile in TileType.REDS :
            self.reds[tile//10] = True
            tile += 5
        self.hand[tile] += 1


    # 牌を捨てる
    def discard_tile(self, discarded_tile, exchanged) -> None :
        # 河への記録
        self.discarded_state += [exchanged]
        self.discarded_tiles += [discarded_tile]

        # 赤牌を番号に変換
        if discarded_tile in TileType.REDS :
            self.reds[discarded_tile // 10] = False
            discarded_tile += 5

        # 手牌から切る牌を減らす
        self.hand[discarded_tile] -= 1
        # 切る牌をフリテン牌に記録する
        self.furiten_tiles[discarded_tile] += 1
        # 枚数だけ記録する河に切る牌を記録する
        self.discarded_tiles_hist[discarded_tile] += 1

        # 流し満貫継続中かどうかチェック
        if self.has_right_to_nagashi_mangan and not(discarded_tile in (TileType.TERMINALS | TileType.HONORS)) :
            self.has_right_to_nagashi_mangan = False


    # 流し満貫かチェック
    def check_player_has_right_to_nagashi_mangan(self) -> bool :
        if not(self.last_discarded_tile in (TileType.TERMINALS | TileType.HONORS)) : self.is_nagashi_mangan = False


    # フリテン牌を追加
    def add_furiten_tile(self, tile:int) -> None : self.furiten_tiles[tile] += 1


    # 同巡フリテン牌を追加
    def add_same_turn_furiten_tile(self, tile:int) -> None :
        self.same_turn_furiten_tiles += [tile]
        self.furiten_tiles[tile] += 1


    # 同巡フリテンを解消
    def reset_same_turn_furiten(self) -> None :
        for tile in self.same_turn_furiten_tiles : self.furiten_tiles[tile] -= 1
        self.same_turn_furiten_tiles = []


    # 鳴いて晒した牌を全部戻した手をreturnする
    def put_back_opened_hand(self) -> List[int] :
        hand = self.hand[:]
        for i in range(0,20,5) :
            if self.opened_hand[i] == 0 : break
            elif self.opened_hand[i] == BlockType.OPENED_TRIPLET : hand[self.opened_hand[i+1]] += 3
            elif self.opened_hand[i] == BlockType.OPENED_RUN :
                n = self.opened_hand[i+1]
                hand[n] += 1
                hand[n+1] += 1
                hand[n+2] += 1
            else : hand[self.opened_hand[i+1]] += 4
        return hand


    # 暗槓の処理
    def proc_ankan(self, tile:int) -> None :
        self.opened_hand[self.opened_sets_num * 5] = BlockType.CLOSED_KAN
        self.opened_hand[self.opened_sets_num * 5 + 1] = tile
        self.opened_hand[self.opened_sets_num * 5 + 2] = tile
        self.opened_hand[self.opened_sets_num * 5 + 3] = 0
        self.hand[tile] -= 4
        self.kans_num += 1
        self.opened_sets_num += 1
        if tile in TileType.FIVES :
            self.reds[tile//10] = False
            self.opened_reds[tile//10] = True


    # 加槓の処理
    def proc_kakan(self, tile:int) -> int :
        self.kans_num += 1
        for i in range(4) :
            if self.opened_hand[i*5] == BlockType.OPENED_TRIPLET and self.opened_hand[(i*5)+1] == tile :
                self.opened_hand[i*5] = BlockType.OPENED_KAN
                self.hand[tile] -= 1
                if tile in TileType.FIVES :
                    self.reds[tile//10]= False
                    self.opened_reds[tile//10] = True


    # 大明槓の処理
    def proc_daiminkan(self, tile:int, pos:int) -> None :
        if tile in TileType.REDS :
            self.opened_reds[tile//10] = True
            tile += 5
        elif tile in TileType.FIVES:
            self.reds[tile//10]= False
            self.opened_reds[tile//10] = True

        self.opened_hand[(self.opened_sets_num * 5) + 0] = BlockType.OPENED_KAN
        self.opened_hand[(self.opened_sets_num * 5) + 1] = tile
        self.opened_hand[(self.opened_sets_num * 5) + 2] = tile
        self.opened_hand[(self.opened_sets_num * 5) + 3] = pos
        self.hand[tile] -= 3

        self.kans_num += 1
        self.opened_sets_num += 1
        self.has_stealed = True


    # ポンの処理
    def proc_pon(self, tile:int, pos:int, contain_red:bool) -> int :
        if tile in TileType.REDS :
            self.opened_reds[tile//10] = True
            tile += 5
        if contain_red :
            self.reds[tile//10] = False
            self.opened_reds[tile//10] = True

        self.opened_hand[(self.opened_sets_num * 5) + 0] = BlockType.OPENED_TRIPLET
        self.opened_hand[(self.opened_sets_num * 5) + 1] = tile
        self.opened_hand[(self.opened_sets_num * 5) + 2] = tile
        self.opened_hand[(self.opened_sets_num * 5) + 3] = pos
        self.opened_sets_num += 1
        self.has_stealed = True
        self.hand[tile] -= 2

        if tile in TileType.DRAGONS :
            self.dragons_num += 1
            if self.dragons_num == 3 : return 0 # 大三元のパオであれば0を返す
        elif tile in TileType.HONORS :
            self.winds_num += 1
            if self.winds_num == 4 : return 1 # 大喜四のパオであれば1を返す

        return -1 # パオでなければ-1を返す


    # チーの処理
    def proc_chii(self, tile:int, tile1:int, tile2:int) -> None :
        if tile in TileType.REDS :
            self.opened_reds[tile//10] = True
            tile += 5
        elif tile1 in TileType.REDS :
            self.reds[tile1//10] = False
            self.opened_reds[tile1//10] = True
            tile1 += 5
        elif tile2 in TileType.REDS :
            self.reds[tile2//10] = False
            self.opened_reds[tile2//10] = True
            tile2 += 5

        self.opened_hand[(self.opened_sets_num * 5) + 0] = BlockType.OPENED_RUN
        if tile1 > tile : min_tile = tile
        else : min_tile = tile1
        self.opened_hand[(self.opened_sets_num * 5) + 1] = min_tile
        self.opened_hand[(self.opened_sets_num * 5) + 2] = tile
        self.opened_hand[(self.opened_sets_num * 5) + 3] = 3
        self.opened_sets_num += 1
        self.has_stealed = True
        self.hand[tile1] -= 1
        self.hand[tile2] -= 1


    # 鳴いた後の手出し牌を登録
    def add_tile_to_discard_tiles_after_stealing(self, tile:int) -> None :
        self.opened_hand[((self.opened_sets_num - 1) * 5) + 4] = tile


    # handを標準出力に表示
    def print_hand(self) -> None:
        s_hand = ""
        for i in range(1,38) :
            if i == 10 : s_hand += "m"
            elif i == 20 : s_hand += "p"
            elif i == 30 : s_hand += "s"
            for j in range(self.hand[i]) :
                if i < 30 : s_hand += str(i%10)
                elif i == 31 : s_hand += "東"
                elif i == 32 : s_hand += "南"
                elif i == 33 : s_hand += "西"
                elif i == 34 : s_hand += "北"
                elif i == 35 : s_hand += "白"
                elif i == 36 : s_hand += "発"
                else : s_hand += "中"
        print(s_hand)
