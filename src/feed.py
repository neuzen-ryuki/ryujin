# sys

# 3rd
import numpy as np

# ours
from mytypes import BlockType
from params import Params as p


class Feed :
    def __init__(self) :
        self.i_feed  = 1
        self.i_batch = 0

        self.feed_x_m   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_p   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_s   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_h   = np.zeros((p.BATCH_SIZE, p.HONOR_ROW, p.COL, p.PLANE))
        self.feed_x_aux = np.zeros((p.BATCH_SIZE, p.AUX_INPUT))
        self.feed_x     = [self.feed_x_m, self.feed_x_p, self.feed_x_s, self.feed_x_h, self.feed_x_aux]
        self.feed_y     = np.zeros((p.BATCH_SIZE, p.OUTPUT))

        self.suit  = 0
        self.plane = 0
        self.start = 0
        self.end   = 0


    # feedを書き切ったらをnpzファイルとして吐き出す
    def save_feed(self) :
        # ファイルを保存
        np.savez(p.SAVE_DIR + "feed_%05d" % self.i_feed,
                 m=self.feed_x_m,
                 p=self.feed_x_p,
                 s=self.feed_x_s,
                 h=self.feed_x_h,
                 aux=self.feed_x_aux,
                 y=self.feed_y)
        self.i_feed += 1

        # データ初期化
        self.i_batch = 0
        self.feed_x_m   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_p   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_s   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
        self.feed_x_h   = np.zeros((p.BATCH_SIZE, p.HONOR_ROW, p.COL, p.PLANE))
        self.feed_x_aux = np.zeros((p.BATCH_SIZE, p.AUX_INPUT))
        self.feed_x     = [self.feed_x_m, self.feed_x_p, self.feed_x_s, self.feed_x_h, self.feed_x_aux]
        self.feed_y     = np.zeros((p.BATCH_SIZE, p.OUTPUT))


    # feed_xに情報を書き込む
    def write_feed_x(self, game, players, pn) :
        for suit in range(4) :
            self.suit, self.plane = suit, 0
            if suit == 3 : self.start, self.end = 31, 38
            else         : self.start, self.end = suit*10+1, (suit+1)*10

            player = players[pn]
            self.write_about_hand(player.hand)
            self.write_about_opened_hand(player.opened_hand)
            self.write_about_appearing_tiles(game.appearing_tiles[:], player.hand)
            self.write_about_enemy_players(players, pn)
            self.write_about_doras(game.doras, game.dora_has_opened)
            if suit == 3 : self.write_about_winds(game.rounds_num, player.players_wind)
            else         : self.write_about_reds(player.reds, player.opened_reds)
        self.write_about_aux(game, players, pn)


    # feedにプレイヤの手牌について書き込む
    # 1 plane
    def write_about_hand(self, hand) :
        row = 0
        for i in range(self.start,self.end) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,0]
            else              : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,1]
            row += 1

        self.plane += 1


    # feedにプレイヤの副露手牌について書き込む
    # 1 plane
    def write_about_opened_hand(self, opened_hand) :
        hand = [0] * 38
        for i in range(0,20,5) :
            if   opened_hand[i] == 0 : break
            elif opened_hand[i] == BlockType.OPENED_TRIPLET : hand[opened_hand[i+1]] += 3
            elif opened_hand[i] == BlockType.OPENED_RUN :
                n = opened_hand[i+1]
                hand[n] += 1
                hand[n+1] += 1
                hand[n+2] += 1
            else : hand[opened_hand[i+1]] += 4

        row = 0
        for i in range(self.start, self.end) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,0]
            else              : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,1]
            row += 1

        self.plane += 1


    # feedにプレイヤから見えている牌の枚数について書き込む
    # 1 plane
    def write_about_appearing_tiles(self, appearing_tiles, hand) :
        row = 0
        for i in range(self.start, self.end) :
            if   appearing_tiles[i] + hand[i] == 0 : pass
            elif appearing_tiles[i] + hand[i] == 1 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,0,0,0]
            elif appearing_tiles[i] + hand[i] == 2 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,0,0]
            elif appearing_tiles[i] + hand[i] == 3 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,0]
            else                                   : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,1]
            row += 1

        self.plane += 1


    # feedに他プレイヤの情報(安全牌, 副露牌)について書き込む
    # 6 planes
    def write_about_enemy_players(self, players, pn) :
        for i in range(1,4) :
            ep = (pn + i) % 4
            row = 0
            for j in range(self.start, self.end) :
                if players[ep].furiten_tiles[j] > 0 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = 1
                row += 1
            self.plane += 1
            self.write_about_opened_hand(players[ep].opened_hand)


    # feedにドラの情報について書き込む
    # 1 plane
    def write_about_doras(self, doras, dora_has_opened) :
        for i in range(5) :
            if dora_has_opened[i] is False or doras[i] == 0 : break
            if doras[i] < self.start or doras[i] > self.end : continue
            self.feed_x[self.suit][self.i_batch,doras[i]%10-1,:,self.plane] = 1

        self.plane += 1


    # feedに赤牌の情報を書き込む
    # 2 planes
    def write_about_reds(self, reds, opened_reds) :
        if reds[self.suit] : self.feed_x[self.suit][self.i_batch,4,:,self.plane] = 1
        self.plane += 1

        if opened_reds[self.suit] : self.feed_x[self.suit][self.i_batch,4,:,self.plane] = 1
        self.plane += 1


    # feedに役風牌の情報を書き込む
    # 2 planes
    def write_about_winds(self, rounds_num, players_wind) :
        # 手牌の中の赤
        self.feed_x[self.suit][self.i_batch,rounds_num,:,self.plane] = 1
        self.plane += 1
        # 副露牌の赤
        self.feed_x[self.suit][self.i_batch,(players_wind % 10)-1,:,self.plane] = 1
        self.plane += 1


    # feedに諸々の情報を書き込む
    def write_about_aux(self, game, players, pn) :
        i = 0
        player = players[pn]
        # 自分が何家スタートか
        self.feed_x_aux[self.i_batch,i + player.player_num] = 1
        i += 4
        # 今自分が何家か
        self.feed_x_aux[self.i_batch,i + ((player.players_wind % 10) - 1)] = 1
        i += 4
        # 場
        self.feed_x_aux[self.i_batch,i + game.rounds_num] = 1
        i += 3
        # 局
        self.feed_x_aux[self.i_batch,i + game.rotations_num] = 1
        i += 4
        # 供託の数
        for dn in range(1,9) :
            if game.deposits_num >= dn : self.feed_x_aux[self.i_batch,i + dn] = 1
            else : break
        i += 8
        # 本場の数
        for cn in range(1,9) :
            if game.counters_num >= i : self.feed_x_aux[self.i_batch,i + cn] = 1
            else : break
        i += 8
        # 点数
        for p in range(4) :
            player_num = (pn + p) % 4
            self.feed_x_aux[self.i_batch,i] = players[player_num].score // 100
            i += 1
        # 残り山の枚数
        self.feed_x_aux[self.i_batch,i] = game.remain_tiles_num
        i += 1
        # 残りツモの回数
        self.feed_x_aux[self.i_batch,i] = game.remain_tiles_num // 4
        i += 1
        # 他家のリーチ
        for p in range(1,4) :
            player_num = (pn + p) % 4
            if players[player_num].has_declared_ready : self.feed_x_aux[self.i_batch,i] = 1
            i += 1
            if players[player_num].has_declared_double_ready : self.feed_x_aux[self.i_batch,i] = 1
            i += 1


    # feed_yに情報を書き込む
    def write_feed_y(self, discarded_tile) :
        self.feed_y[self.i_batch][discarded_tile] = 1


