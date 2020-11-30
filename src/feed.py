# sys

# 3rd
import numpy as np

# ours
from mytypes import BlockType
from params import Params as p


class Feed :

    VERSION       = "v0.0.1"
    SAVE_PATH     = "../data/feed/2019/"
    BATCH_SIZE    = 5000
    MPS_ROW     = 9
    ROW_HONOR   = 7
    COL         = 4
    MPS_PLANE   = 2
    PLANE_HONOR = 2
    OUTPUT      = 38


    def __init__(self) :
        self.i_feed  = 1
        self.i_batch = 0
        self.feed_x_m = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.MPS_PLANE))
        self.feed_x_p = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.MPS_PLANE))
        self.feed_x_s = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.MPS_PLANE))
        self.feed_x_h = np.zeros((p.BATCH_SIZE, p.HONOR_ROW, p.COL, p.HONOR_PLANE))
        self.feed_y   = np.zeros((p.BATCH_SIZE, p.OUTPUT))
        self.feed_x_mps = [self.feed_x_m, self.feed_x_p, self.feed_x_s]


    # feedを書き切ったらをnpzファイルとして吐き出す
    def save_feed(self) :
        # ファイルを保存
        np.savez(self.SAVE_PATH + "feed_%05d" % self.i_feed, m=self.feed_x_m, p=self.feed_x_p, s=self.feed_x_s, h=self.feed_x_h, y=self.feed_y)
        self.i_feed += 1

        # データ初期化
        self.i_batch = 0
        self.feed_x_m = np.zeros((self.BATCH_SIZE, self.MPS_ROW,   self.COL, self.MPS_PLANE))
        self.feed_x_p = np.zeros((self.BATCH_SIZE, self.MPS_ROW,   self.COL, self.MPS_PLANE))
        self.feed_x_s = np.zeros((self.BATCH_SIZE, self.MPS_ROW,   self.COL, self.MPS_PLANE))
        self.feed_x_h = np.zeros((self.BATCH_SIZE, self.HONOR_ROW, self.COL, self.HONOR_PLANE))
        self.feed_y   = np.zeros((self.BATCH_SIZE, self.OUTPUT))


    # feed_yに情報を書き込む
    def write_feed_y(self, discarded_tile) :
        self.feed_y[self.i_batch][discarded_tile] = 1

    # feed_x_mpsに情報を書き込む
    def write_feed_x_mps(self, game, player, mode) :
        plane = 0
        plane = self.write_about_hand_mps(player.hand, plane, mode)
        plane = self.write_about_opened_hand_mps(player.opened_hand, plane, mode)


    # feedにプレイヤの手牌のマンピンソウについて書き込む
    def write_about_hand_mps(self, hand, plane, mode) :
        col = 0
        row = 0
        for i in range(mode*10+1,(mode+1)*10) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,1,0]
            else              : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,1,1]
            row += 1

        plane += 1
        return plane


    # feedにプレイヤの副露手牌のマンピンソウについて書き込む
    def write_about_opened_hand_mps(self, opened_hand, plane, mode) :
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
        col = 0
        for i in range(mode*10+1,(mode+1)*10) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,1,0]
            else              : self.feed_x_mps[mode][self.i_batch,row,:,plane] = [1,1,1,1]
            row += 1

        plane += 1
        return plane


    # feed_x_hに情報を書き込む
    def write_feed_x_h(self, game, player) :
        plane = 0
        plane = self.write_about_hand_h(player.hand, plane)
        plane = self.write_about_opened_hand_h(player.opened_hand, plane)


    # feedにプレイヤの手牌の字牌について書き込む
    def write_about_hand_h(self, hand, plane) :
        col = 0
        row = 0
        for i in range(31,38) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x_h[self.i_batch,row,:,plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,1,0]
            else              : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,1,1]
            row += 1

        plane += 1
        return plane


    # feedにプレイヤの副露手牌の字牌について書き込む
    def write_about_opened_hand_h(self, opened_hand, plane) :
        hand = [0] * 38
        for i in range(0,20,5) :
            if   opened_hand[i] == 0 : break
            elif opened_hand[i] == BlockType.OPENED_TRIPLET :
                hand[opened_hand[i+1]] += 3
            elif opened_hand[i] == BlockType.OPENED_RUN :
                n = opened_hand[i+1]
                hand[n] += 1
                hand[n+1] += 1
                hand[n+2] += 1
            else : hand[opened_hand[i+1]] += 4

        row = 0
        col = 0
        for i in range(31,38) :
            if   hand[i] == 0 : pass
            elif hand[i] == 1 : self.feed_x_h[self.i_batch,row,:,plane] = [1,0,0,0]
            elif hand[i] == 2 : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,0,0]
            elif hand[i] == 3 : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,1,0]
            else              : self.feed_x_h[self.i_batch,row,:,plane] = [1,1,1,1]
            row += 1

        plane += 1
        return plane




