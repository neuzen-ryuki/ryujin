#std
import sys

# 3rd

# ours
from .params import Params as p
from .player import Player
from .mytypes import TileType, ActionType


class Game :
    def __init__(self, xml_root, file_name:str, mode:str, feed=None) :
        self.file_name = file_name
        self.mode = mode
        self.xml_root = xml_root
        self.i_log = 4
        self.players = [Player(i) for i in range(4)]

        self.feed = feed


    # tagによって実行する処理メソッドを切り替える
    def switch_proc(self, item) :
        if   item.tag    == "UN"        : self.proc_UN(item.attrib)
        elif item.tag    == "DORA"      : self.proc_DORA(int(item.attrib["hai"]))
        elif item.tag[0] == "T"         : self.proc_Tsumo(0, int(item.tag[1:]))
        elif item.tag[0] == "U"         : self.proc_Tsumo(1, int(item.tag[1:]))
        elif item.tag[0] == "V"         : self.proc_Tsumo(2, int(item.tag[1:]))
        elif item.tag[0] == "W"         : self.proc_Tsumo(3, int(item.tag[1:]))
        elif item.tag[0] == "D"         : self.proc_Dahai(0, int(item.tag[1:]))
        elif item.tag[0] == "E"         : self.proc_Dahai(1, int(item.tag[1:]))
        elif item.tag[0] == "F"         : self.proc_Dahai(2, int(item.tag[1:]))
        elif item.tag[0] == "G"         : self.proc_Dahai(3, int(item.tag[1:]))
        elif item.tag    == "N"         : self.proc_N(int(item.attrib["who"]), int(item.attrib["m"]))
        elif item.tag    == "INIT"      : self.proc_INIT(item.attrib)
        elif item.tag    == "REACH"     : self.proc_REACH(item.attrib)
        elif item.tag    == "AGARI"     : self.proc_AGARI(item.attrib)
        elif item.tag    == "RYUUKYOKU" : self.proc_RYUUKYOKU(item.attrib)
        elif item.tag    == "BYE"       : self.proc_BYE(int(item.attrib["who"]))


    # 学習と同時にfeedを作る時用のgenerator
    def generate_feed(self) :
        for item in self.xml_root[4:] :
            self.switch_proc(item)
            if p.BATCH_SIZE == self.feed.i_batch :
                yield (self.feed.feed_x, self.feed.feed_y)
                self.feed.init_feed()


    # INITタグの処理
    def proc_INIT(self, attr) :
        # Gameのメンバ変数初期化
        seed = attr["seed"].split(",")
        self.rounds_num = int(seed[0]) // 4
        self.rotations_num = int(seed[0]) % 4
        self.counters_num = int(seed[1])
        self.deposits_num = int(seed[2])
        self.remain_tiles_num = 70                              # 山に残っているツモ牌の数
        self.org_got_tile = -1                                  # ツモ牌番号保存用
        self.doras = [0] * 5                                    # ドラ
        self.dora_indicators = [0] * 5                          # ドラ表示牌
        self.dora_has_opened = [False] * 5                      # ドラの状態, i番目がTrue → dora_indicators[i]がめくれている状態
        self.appearing_tiles = [0] * 38                         # プレイヤ全員に見えている牌， appearing_tiles[i]がj → i番の牌がj枚見えている
        self.appearing_red_tiles = [False] * 3                  # プレイヤ全員に見えている赤牌． 萬子，筒子，索子の順．
        self.pao_info = [-1] * 4                                # パオ記録用
        self.steal_flag = False                                 # 手出し記録用
        self.write_flag = False                                 # 鳴き学習用
        self.is_first_turn = False                              # 1巡目かどうか
        self.prevailing_wind = 31 + self.rounds_num             # 場風にあたる牌番号

        # 鳴き用
        self.tile = -1
        self.tile1 = -1
        self.tile2 = -1
        self.pos = -1
        self.contain_red = False


        # Player関係のメンバ変数を初期化
        ten = attr["ten"].split(",")
        for i in range(4) :
            self.players[i].init_subgame(self.rotations_num)
            self.players[i].score = int(ten[i]) * 100

        # 配牌を配る
        for i in range(4) :
            starting_hand = [self.convert_tile(int(tile)) for tile in attr[f"hai{i}"].split(",")]
            for tile in starting_hand : self.players[i].get_tile(tile)

        # 初期ドラをセット
        indicator = self.convert_tile(int(seed[5]))
        self.open_new_dora(indicator)


    # 新しいドラを開く
    def open_new_dora(self, indicator) -> None :
        for i in range(5) :
            if self.dora_has_opened[i] is False :
                self.dora_has_opened[i] = True
                self.dora_indicators[i] = indicator
                self.appearing_tiles[indicator] += 1
                if indicator in TileType.REDS : indicator = indicator + 5
                if indicator in TileType.NINES : self.doras[i] = indicator - 8
                elif indicator == 34 : self.doras[i] = 31
                elif indicator == 37 : self.doras[i] = 35
                else : self.doras[i] = indicator + 1
                break


    # T, U, V, Wタグの処理
    def proc_Tsumo(self, player_num, tile) :
        # 一つ前の打牌で，誰かが鳴けたけど鳴かなかった時のfeed_steal_yへの書き込み
        if self.write_flag :
            self.feed.write_feed_steal(self, self.players, 0)
            self.write_flag = False

        self.org_got_tile = tile
        self.remain_tiles_num -= 1
        self.players[player_num].has_right_to_one_shot = False
        self.players[player_num].reset_same_turn_furiten()
        self.players[player_num].get_tile(self.convert_tile(tile))


    # D, E, F, Gタグの処理
    def proc_Dahai(self, player_num, tile) :
        if tile in {16, 52, 88} : self.appearing_red_tiles[tile // 36] = True
        discarded_tile = self.convert_tile(tile)
        exchanged = False
        if tile != self.org_got_tile : exchanged = True

        # feed_mainへ書き込み
        if self.mode == "main" and not(self.players[player_num].has_declared_ready) and self.players[player_num].exists :
            self.feed.write_feed_x(self, self.players, player_num)
            self.feed.write_feed_main_y(discarded_tile)
            self.feed.i_batch += 1

        # プレイヤが牌を切る
        self.players[player_num].discard_tile(discarded_tile, exchanged)

        # 捨てられた牌を見えている牌に記録
        self.add_to_appearing_tiles(discarded_tile)

        # 切られた牌を他プレイヤの同巡フリテン牌として記録
        for i in range(1, 4) : self.players[(player_num + i) % 4].add_same_turn_furiten_tile(discarded_tile)

        # 鳴いた後に切った場合, 手出し牌に牌を記録
        if self.steal_flag : self.players[player_num].add_tile_to_discard_tiles_after_stealing(discarded_tile)
        self.steal_flag = False

        # feed_stealへ書き込み
        if self.mode == "steal" and self.remain_tiles_num > 0 :
            # 他のプレイヤが鳴けるかどうか判定する
            action_players_num = 0
            for i in range(1,4) :
                action_player = (player_num + i) % 4
                if self.players[action_player].can_steal(discarded_tile, i) :
                    self.write_flag = True
                    self.feed.write_steal_info(action_player, discarded_tile, (4 + player_num - action_player) % 4, action_players_num)
                    action_players_num += 1


    # 鳴きの処理
    def proc_N(self, player_num:int, mc:int) :
        # mcからどんな鳴きが行われたかを判別
        action = self.analyze_mc(player_num, mc)

        # feed_steal_xyへの書き込み
        if self.write_flag :
            self.feed.write_feed_steal(self, self.players, action)
            self.write_flag = False

        # 鳴きによる変数処理
        if action == 1 : self.proc_pon(player_num, self.pos , self.tile, self.contain_red)
        elif action in {3, 4, 5} : self.proc_chii(player_num, self.tile, self.tile1, self.tile2)
        elif action == 6 : self.proc_kakan(player_num, self.tile)
        elif action == 2 : self.proc_daiminkan(player_num, self.tile, self.pos)
        elif action == 7 : self.proc_ankan(player_num, self.tile)
        else :
            print("error! in proc_N()")
            print("unkown mc")
            sys.exit()


    # Nタグについているmコードを解析してそれぞれの鳴きに対する処理をする
    def analyze_mc(self, player_num:int, mc:int) -> int :
        # チー
        if  (mc & 0x0004) :
            pt = (mc & 0xFC00) >> 10
            r  = pt % 3
            pn = pt // 3
            color = pn // 7
            n  = (color * 10) + (pn % 7) + 1
            run = [n, n+1, n+2]
            pp = [(mc & 0x0018) >> 3, (mc & 0x0060) >> 5, (mc & 0x0180) >> 7]
            for i in range(3) :
                if (run[i] % 10 == 5 and pp[i] == 0) : run[i] -= 5
            if   r == 0 : self.tile, self.tile1, self.tile2 = run[0], run[1], run[2]
            elif r == 1 : self.tile, self.tile1, self.tile2 = run[1], run[0], run[2]
            elif r == 2 : self.tile, self.tile1, self.tile2 = run[2], run[0], run[1]
            return 3 + r

        # ポン
        elif(mc & 0x0008) :
            self.pos = mc & 0x0003
            pt = (mc & 0xFE00) >> 9
            r  = pt % 3
            pn =  pt // 3
            color = pn // 9
            self.tile  = (color * 10) + (pn % 9) + 1
            self.contain_red = False
            if(color != 3 and self.tile % 10 == 5) :
                if ((mc & 0x0060) == 0) : pass
                elif (r == 0)           : self.tile -= 5
                else                    : self.contain_red = True
            return 1

        # 加槓
        elif(mc & 0x0010) :
            self.pos = mc & 0x0003
            pt = (mc & 0xFE00) >> 9
            r  = pt % 3
            pn =  pt // 3
            color = pn // 9
            self.tile  = (color * 10) + (pn % 9) + 1
            return 6

        # 大明槓, 暗槓
        else :
            self.pos = mc & 0x0003
            pt = (mc & 0xFF00) >> 8
            r  = pt % 4
            pn =  pt // 4
            color = pn // 9  # 0:萬子,...
            self.tile  = (color * 10) + (pn % 9) + 1
            action = 2
            if(color != 3 and tile % 10 == 5) :
                if (pos == 0) : action = 7
                elif (r == 0) : self.tile -= 5
            return action


    # チーが行われた時の処理
    def proc_chii(self, player_num:int, tile:int, tile1:int, tile2:int) -> None :
        self.players[player_num].proc_chii(tile, tile1, tile2)

        if tile in TileType.REDS : tile += 5
        elif tile1 in TileType.REDS :
            tile1 += 5
            self.appearing_red_tiles[tile1 // 10] = True
        elif tile2 in TileType.REDS :
            tile2 += 5
            self.appearing_red_tiles[tile2 // 10] = True
        self.appearing_tiles[tile1] += 1
        self.appearing_tiles[tile2] += 1
        self.is_first_turn = False
        self.steal_flag = True
        for i in range(4) : self.players[i].has_right_to_one_shot = False


    # ポンが行われた時の処理
    def proc_pon(self, player_num:int, pos:int, tile:int, contain_red:bool) -> None:
        pao = self.players[player_num].proc_pon(tile, pos, contain_red)
        if pao > -1 : self.set_pao(pao, player_num, (player_num + pos) % 4)
        if tile in TileType.REDS :
            tile += 5
            self.appearing_red_tiles[tile // 10] = True
        if contain_red : self.appearing_red_tiles[tile // 10] = True
        self.appearing_tiles[tile] += 2
        self.is_first_turn = False
        self.steal_flag = True
        for i in range(4) : self.players[i].has_right_to_one_shot = False


    # 加槓が行われた時の処理
    def proc_kakan(self, player_num:int, tile:int) -> None :
        self.players[player_num].proc_kakan(tile)
        if tile in TileType.REDS :
            tile += 5
            self.appearing_red_tiles[tile // 10] = True
        self.appearing_tiles[tile] += 1
        for i in range(4) : self.players[i].has_right_to_one_shot = False


    # 大明槓が行われた時の処理
    def proc_daiminkan(self, player_num:int, tile:int, pos:int) -> None :
        self.players[player_num].proc_daiminkan(tile, pos)
        if tile in TileType.REDS : tile += 5
        self.appearing_tiles[tile] += 3
        self.is_first_turn = False


    # 暗槓が行われた時の処理
    def proc_ankan(self, player_num:int, tile:int) -> None :
        if tile in TileType.FIVES : self.appearing_red_tiles[tile // 10] = True
        self.appearing_tiles[tile] += 4
        self.is_first_turn = False
        for i in range(4) : self.players[i].has_right_to_one_shot = False
        self.players[player_num].proc_ankan(tile)


    # リーチ時の処理
    def proc_REACH(self, attr) :
        if attr["step"] == "2" :
            player_num, step = int(attr["who"]), attr["step"]
            self.ready_flag = True
            self.deposits_num += 1
            self.players[player_num].declare_ready(self.is_first_turn)


    # 和了の処理
    def proc_AGARI(self, attr) :
        # 何もやることなくね
        # 何点で和了ったとかどんな手で和了ったとかは簡単に取ってこれるのでその辺のデータが取りたい時にはここで何かしらやることになる？
        return


    # 流局の処理
    def proc_RYUUKYOKU(self, attr) :
        # 何もやることなくね
        return


    # 槓でドラが開かれる時の処理
    def proc_DORA(self, indicator) :
        self.open_new_dora(self.convert_tile(indicator))


    # playerが回線落ちした時の処理
    def proc_BYE(self, player_num) :
        self.players[player_num].exists = False


    # playerが回線落ちから帰ってきた時の処理
    def proc_UN(self, attr) :
        if "n0" in attr : self.players[0].exists = True
        if "n1" in attr : self.players[1].exists = True
        if "n2" in attr : self.players[2].exists = True
        if "n3" in attr : self.players[3].exists = True


    # xml_logの牌番号をこのプログラムの牌番号に変換
    def convert_tile(self, org_tile:int) -> int:
        if org_tile in {16, 52, 88} : tile = (org_tile // 40) * 10
        else :
            tile = org_tile // 4
            if tile >= 27 : tile += 4
            elif tile >= 18 : tile += 3
            elif tile >= 9 : tile += 2
            else : tile += 1

        return tile


    # 全員に公開されている牌を追加
    def add_to_appearing_tiles(self, tile:int) -> None :
        if tile in TileType.REDS :
            self.appearing_red_tiles[tile // 10] = True
            self.appearing_tiles[tile + 5] += 1
        else : self.appearing_tiles[tile] += 1


    # パオをセット i_ap: 最後の牌を鳴いたプレイヤ, i_dp: 鳴かせたプレイヤ
    def set_pao(self, pao:int, i_ap:int, i_dp:int) -> None :
        self.pao_info[pao*2] = i_ap
        self.pao_info[pao*2+1] = i_dp


