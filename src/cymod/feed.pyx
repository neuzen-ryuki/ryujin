# sys
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../pymod'))

# 3rd
import numpy as np
from termcolor import colored

# ours
from mytypes import BlockType
from params import Params as p

# cython
cimport numpy as np
from libcpp cimport bool


cdef class Feed :
    cdef public str mode
    cdef public int i_feed
    cdef public int i_batch
    cdef public int batch_size
    cdef public int suit
    cdef public int plane
    cdef public int start
    cdef public int end
    cdef public list feed_x
    cdef public list feed_y
    cdef public int[5] steal_info
    cdef public np.ndarray feed_x_m
    cdef public np.ndarray feed_x_p
    cdef public np.ndarray feed_x_s
    cdef public np.ndarray feed_x_h
    cdef public np.ndarray feed_x_aux
    cdef public np.ndarray feed_x_ep1
    cdef public np.ndarray feed_x_ep2
    cdef public np.ndarray feed_x_ep3
    cdef public np.ndarray feed_x_si
    cdef public np.ndarray feed_y_main
    cdef public np.ndarray feed_y_ready
    cdef public np.ndarray feed_y_ep1
    cdef public np.ndarray feed_y_ep2
    cdef public np.ndarray feed_y_ep3
    cdef public np.ndarray feed_y_steal
    cdef public np.ndarray feed_y_red


    def __init__(self, mode, batch_size) :
        self.mode = mode
        self.i_feed  = 1
        self.i_batch = 0
        self.suit  = 0
        self.plane = 0
        self.start = 0
        self.end   = 0
        self.steal_info = [-1] * 5

        if batch_size == 0 : self.batch_size = p.BATCH_SIZE
        else : self.batch_size = batch_size
        self.init_feed()


    # feedを初期化
    cpdef init_feed(self) :
        self.i_batch = 0
        # 入力feed
        self.feed_x_m    = np.zeros((self.batch_size, p.MPS_ROW,   p.COL, p.PLANE)) # m means "manzu"
        self.feed_x_p    = np.zeros((self.batch_size, p.MPS_ROW,   p.COL, p.PLANE)) # p means "pinzu"
        self.feed_x_s    = np.zeros((self.batch_size, p.MPS_ROW,   p.COL, p.PLANE)) # s means "souzu"
        self.feed_x_h    = np.zeros((self.batch_size, p.HONOR_ROW, p.COL, p.PLANE)) # h means "honor"
        self.feed_x_aux  = np.zeros((self.batch_size, p.AUX_INPUT))                 # aux means "auxiliary"
        self.feed_x_ep1  = np.zeros((self.batch_size, p.EP_INPUT))                  # ep means "enemy player", ep1 is right player
        self.feed_x_ep2  = np.zeros((self.batch_size, p.EP_INPUT))                  # ep means "enemy player", ep2 is opposite player
        self.feed_x_ep3  = np.zeros((self.batch_size, p.EP_INPUT))                  # ep means "enemy player", ep3 is left player
        self.feed_x_si   = np.zeros((self.batch_size, p.SI_INPUT))                  # si means "steal information"

        self.feed_x = [self.feed_x_m,
                       self.feed_x_p,
                       self.feed_x_s,
                       self.feed_x_h,
                       self.feed_x_aux,
                       self.feed_x_ep1,
                       self.feed_x_ep2,
                       self.feed_x_ep3]
        if self.mode == "steal" : self.feed_x.append(self.feed_x_si)

        # 出力feed
        self.feed_y_main  = np.zeros((self.batch_size, p.MAIN_OUTPUT))
        self.feed_y_ready = np.zeros((self.batch_size, p.READY_OUTPUT))
        self.feed_y_ep1   = np.zeros((self.batch_size, p.EP_OUTPUT))
        self.feed_y_ep2   = np.zeros((self.batch_size, p.EP_OUTPUT))
        self.feed_y_ep3   = np.zeros((self.batch_size, p.EP_OUTPUT))
        self.feed_y_steal = np.zeros((self.batch_size, p.STEAL_OUTPUT))
        self.feed_y_red   = np.zeros((self.batch_size, p.RED_OUTPUT))
        if self.mode == "steal" :
            self.feed_y = [self.feed_y_steal,
                           self.feed_y_red,
                           self.feed_y_ep1,
                           self.feed_y_ep2,
                           self.feed_y_ep3]
        else :
            self.feed_y = [self.feed_y_main,
                           self.feed_y_ready,
                           self.feed_y_ep1,
                           self.feed_y_ep2,
                           self.feed_y_ep3]


    # feed, i_batchをゼロクリア
    cpdef clear_feed(self) :
        for x in self.feed_x : x[:] = 0
        for y in self.feed_y : y[:] = 0
        self.i_batch = 0


    # メインのfeed_x,yを書き込む
    cpdef write_feed_main(self, game, players, int player_num, int discarded_tile, ready_flag) :
        self.write_feed_x(self, self.players, player_num)
        self.write_feed_y_main(discarded_tile)
        self.write_feed_y_ready(1 if self.ready_flag else 0)


    # 牌やその他の情報をfeed_xに書き込む
    cpdef write_feed_x(self, game, players, int player_num) :
        cdef int suit

        player = players[player_num]
        for suit in range(4) :
            self.suit, self.plane = suit, 0
            if suit == 3 : self.start, self.end = 31, 38
            else         : self.start, self.end = suit*10+1, (suit+1)*10

            self.write_about_hand(player.hand)
            self.write_about_opened_hand(player.opened_hand)
            self.write_about_appearing_tiles(game.appearing_tiles, player.hand)
            self.write_about_enemy_players(players, player_num)
            self.write_about_doras(game.doras, game.dora_has_opened)
            if suit == 3 : self.write_about_winds(game.rounds_num, player.players_wind)
            else         : self.write_about_reds(player.reds, player.opened_reds)

        self.write_about_aux(game, players, player_num)
        self.write_feed_eps(players, player_num)


    # 鳴きに関するfeed_x,yを書き込む
    cpdef write_feed_steal(self, game, players, int action, bool contain_red) :
        cdef int i

        if self.steal_info[2] >= 0 :
            self.write_feed_x(game, players, self.steal_info[2])
            self.write_about_steal_info(self.steal_info[4], self.steal_info[3])
            if action in {1,2} :
                self.write_feed_y_steal(action)
                self.write_feed_y_red(1 if contain_red else 0)
            else :
                self.write_feed_y_steal(0)
                self.write_feed_y_red(0)
            self.i_batch += 1

        if self.steal_info[0] >= 0 and not(action in {1,2}) and self.i_batch < self.batch_size :
            self.write_feed_x(game, players, self.steal_info[0])
            self.write_about_steal_info(self.steal_info[4], self.steal_info[1])
            self.write_feed_y_red(contain_red)
            if action in {3, 4, 5} :
                self.write_feed_y_steal(action)
                self.write_feed_y_red(1 if contain_red else 0)
            else :
                self.write_feed_y_steal(0)
                self.write_feed_y_red(0)
            self.i_batch += 1

        self.steal_info = [-1] * 5


    # 手牌読みに関するfeed_x,yを書き込む
    cpdef write_feed_eps(self, players, int player_num) :
        cdef int i, ep_num

        for i in range(1,4) :
            ep_num = (player_num + i) % 4 # ep means "enemy player"
            player = players[ep_num]
            if   i == 1 : self.write_about_ep(player, ep_num, self.feed_x_ep1[self.i_batch], self.feed_y_ep1[self.i_batch])
            elif i == 2 : self.write_about_ep(player, ep_num, self.feed_x_ep2[self.i_batch], self.feed_y_ep2[self.i_batch])
            elif i == 3 : self.write_about_ep(player, ep_num, self.feed_x_ep3[self.i_batch], self.feed_y_ep3[self.i_batch])


    # 鳴きに関する情報を一旦書き込んで保存する
    cpdef write_steal_info(self, int player_num, int tile, int pos, int i) :
        self.steal_info[4] = tile
        self.steal_info[i * 2] = player_num
        self.steal_info[i * 2 + 1] = pos


    # 1行を書き込む
    cdef int write_row(self, int tiles_num, int row) :
        if   tiles_num == 0 : pass
        elif tiles_num == 1 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,0,0,0]
        elif tiles_num == 2 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,0,0]
        elif tiles_num == 3 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,0]
        else                : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = [1,1,1,1]
        row += 1
        return row


    # feedにプレイヤの手牌について書き込む
    # 1 plane
    cpdef write_about_hand(self, hand) :
        cdef int i, row
        row = 0
        for i in range(self.start,self.end) : row = self.write_row(hand[i], row)
        self.plane += 1


    # feedにプレイヤの副露手牌について書き込む
    # 1 plane
    cpdef  write_about_opened_hand(self, opened_hand) :
        cdef int i, row
        cdef int[38] hand

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
        for i in range(self.start, self.end) : row = self.write_row(hand[i], row)
        self.plane += 1


    # feedにプレイヤから見えている牌の枚数について書き込む
    # 1 plane
    cpdef write_about_appearing_tiles(self, appearing_tiles, hand) :
        cdef int i,row
        row = 0
        for i in range(self.start, self.end) : row = self.write_row(appearing_tiles[i] + hand[i], row)
        self.plane += 1


    # feedに他プレイヤの情報(安全牌, 副露牌)について書き込む
    # 6 planes
    cpdef write_about_enemy_players(self, players, int player_num) :
        cdef int i, ep
        cdef int[20] opened_hand
        for i in range(1,4) :
            ep = (player_num + i) % 4
            row = 0
            for j in range(self.start, self.end) :
                if players[ep].furiten_tiles[j] > 0 : self.feed_x[self.suit][self.i_batch,row,:,self.plane] = 1
                row += 1
            self.plane += 1
            opened_hand = players[ep].opened_hand
            self.write_about_opened_hand(opened_hand)


    # feedにドラの情報について書き込む
    # 1 plane
    cpdef void write_about_doras(self, doras, dora_has_opened) :
        cdef int i
        for i in range(5) :
            if dora_has_opened[i] is False or doras[i] == 0 : break
            if doras[i] < self.start or doras[i] > self.end : continue
            self.feed_x[self.suit][self.i_batch,doras[i]%10-1,:,self.plane] = 1

        self.plane += 1


    # feedに赤牌の情報を書き込む
    # 2 planes
    cpdef write_about_reds(self, reds, opened_reds) :
        if reds[self.suit] : self.feed_x[self.suit][self.i_batch,4,:,self.plane] = 1
        self.plane += 1

        if opened_reds[self.suit] : self.feed_x[self.suit][self.i_batch,4,:,self.plane] = 1
        self.plane += 1


    # feedに役風牌の情報を書き込む
    # 2 planes
    cdef void write_about_winds(self, int rounds_num, int players_wind) :
        # 手牌の中の赤
        self.feed_x[self.suit][self.i_batch,rounds_num,:,self.plane] = 1
        self.plane += 1
        # 副露牌の赤
        self.feed_x[self.suit][self.i_batch,(players_wind % 10)-1,:,self.plane] = 1
        self.plane += 1


    # feedに鳴きの付属情報（出た牌，出た場所）を書き込む
    cpdef write_about_steal_info(self, int tile, int pos) :
        cdef int i
        i = 0
        self.feed_x_si[self.i_batch,tile] = 1
        i += 38
        self.feed_x_si[self.i_batch,pos + i] = 1


    # feedに諸々の情報を書き込む
    cpdef write_about_aux(self, game, players, int player_num) :
        cdef int i, dn, cn, p

        i = 0
        player = players[player_num]
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
            player_num = (player_num + p) % 4
            self.feed_x_aux[self.i_batch,i] = players[player_num].score // 100
            i += 1
        # 残り山の枚数
        self.feed_x_aux[self.i_batch,i] = game.remain_tiles_num
        i += 1
        # 残りツモの回数
        self.feed_x_aux[self.i_batch,i] = game.remain_tiles_num // 4
        i += 1
        # 他家のリーチ，一発
        for p in range(1,4) :
            player_num = (player_num + p) % 4
            if players[player_num].has_declared_ready : self.feed_x_aux[self.i_batch,i] = 1
            i += 1
            if players[player_num].has_declared_double_ready : self.feed_x_aux[self.i_batch,i] = 1
            i += 1
            if players[player_num].has_right_to_one_shot : self.feed_x_aux[self.i_batch,i] = 1
            i += 1


    # feedに手牌読み用の情報を書き込む
    cpdef write_about_ep(self, player, int player_num, x, y) :
        cdef int wi

        # feed_xに河の情報を書き込む
        tiles = player.discarded_tiles
        states = player.discarded_state
        for i, (tile, state) in enumerate(zip(tiles, states)) :
            wi = i * 76 + (tile if state else tile + 38)
            x[wi] = 1
        wi = 1900
        x[wi + player.ready_turn_num] = 1
        wi += 25

        # feed_xに鳴きの情報を書き込む
        hand = [0] * 38
        opened_hand = player.opened_hand
        for i in range(0,20,5) :
            tile = opened_hand[i+1]
            if opened_hand[i] == 0 : break
            elif opened_hand[i] == BlockType.OPENED_TRIPLET : hand[tile] += 3
            elif opened_hand[i] == BlockType.OPENED_RUN :
                hand[tile] += 1
                hand[tile+1] += 1
                hand[tile+2] += 1
            else : hand[tile] += 4

        for i in range(1,38) :
            if i % 10 == 0 : continue
            elif hand[i] == 0 : pass
            elif hand[i] == 1 : x[wi] = 1
            elif hand[i] == 2 : x[wi:wi+1] = 1
            elif hand[i] == 3 : x[wi:wi+2] = 1
            elif hand[i] == 4 : x[wi:wi+3] = 1
            wi += 4

        for i in range(0,20,5) :
            if opened_hand[i] != 0 :
                x[wi+opened_hand[i+1]] = 1
                x[wi+opened_hand[i+4]+38] = 1
            wi += 76

        # feed_yに正解の手を書き込む
        hand = player.hand
        wi = 0
        for i in range(1,38) :
            if i % 10 == 0 : continue
            elif hand[i] == 0 : pass
            elif hand[i] == 1 : y[wi] = 1
            elif hand[i] == 2 : y[wi:wi+1] = 1
            elif hand[i] == 3 : y[wi:wi+2] = 1
            elif hand[i] == 4 : y[wi:wi+3] = 1
            wi += 4


    # feed_y_mainに正解ラベルを書き込む
    cpdef write_feed_y_main(self, int i) :
        self.feed_y_main[self.i_batch,i] = 1


    # feed_y_readyに正解ラベルを書き込む
    cpdef write_feed_y_ready(self, int i) :
        self.feed_y_ready[self.i_batch,i] = 1


    # feed_y_stealに正解ラベルを書き込む
    cpdef write_feed_y_steal(self, int i) :
        self.feed_y_steal[self.i_batch,i] = 1


    # feed_y_mainに正解ラベルを書き込む
    cpdef write_feed_y_red(self, int i) :
        self.feed_y_red[self.i_batch,i] = 1

