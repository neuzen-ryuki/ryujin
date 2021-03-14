# std
import os
import sys
import random
import xml.etree.ElementTree as et

# 3rd
import numpy as np
from termcolor import colored

# ours
from pymod.params import Params as p
from pymod.game import Game
from pymod.player import Player
from cymod.feed import Feed
from cymod.shanten import ShantenNumCalculator
from mytools import fntime


@fntime
def create_val(mode) :
    val_x_m     = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_p     = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_s     = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_h     = np.zeros((p.VAL_SIZE, p.HONOR_ROW, p.COL, p.PLANE))
    val_x_aux   = np.zeros((p.VAL_SIZE, p.AUX_INPUT))
    val_x_ep1   = np.zeros((p.VAL_SIZE, p.EP_INPUT))
    val_x_ep2   = np.zeros((p.VAL_SIZE, p.EP_INPUT))
    val_x_ep3   = np.zeros((p.VAL_SIZE, p.EP_INPUT))
    val_x_si    = np.zeros((p.VAL_SIZE, p.SI_INPUT))

    val_y_main  = np.zeros((p.VAL_SIZE, p.MAIN_OUTPUT))
    val_y_ready = np.zeros((p.VAL_SIZE, p.READY_OUTPUT))
    val_y_ep1   = np.zeros((p.VAL_SIZE, p.EP_OUTPUT))
    val_y_ep2   = np.zeros((p.VAL_SIZE, p.EP_OUTPUT))
    val_y_ep3   = np.zeros((p.VAL_SIZE, p.EP_OUTPUT))
    val_y_steal = np.zeros((p.VAL_SIZE, p.STEAL_OUTPUT))

    feed = Feed(mode, 2000)
    shanten_calculator = ShantenNumCalculator()
    game = Game(mode, sc=shanten_calculator, feed=feed)
    dir_components = os.listdir(p.VAL_XML_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.VAL_XML_DIR, f))]
    i = 0
    total_situation_num = 0
    for file_name in files :
        # xmlを読み込む
        try :
            tree = et.parse(p.VAL_XML_DIR + file_name)
        except :
            print(file_name)
            continue
        root = tree.getroot()

        # 1半荘分feedに書き込む
        game.init_game(root, file_name)
        game.read_log()

        # 1半荘からランダムな1局面をval_xyに保存
        total_situation_num += feed.i_batch
        if feed.i_batch == 0 : continue
        i_rnd = random.randint(0,(feed.i_batch - 1))
        val_x_m[i]     = feed.feed_x_m[i_rnd]
        val_x_p[i]     = feed.feed_x_p[i_rnd]
        val_x_s[i]     = feed.feed_x_s[i_rnd]
        val_x_h[i]     = feed.feed_x_h[i_rnd]
        val_x_aux[i]   = feed.feed_x_aux[i_rnd]
        val_x_ep1[i]   = feed.feed_x_ep1[i_rnd]
        val_x_ep2[i]   = feed.feed_x_ep2[i_rnd]
        val_x_ep3[i]   = feed.feed_x_ep3[i_rnd]
        val_x_si[i]    = feed.feed_x_si[i_rnd]

        val_y_main[i]  = feed.feed_y_main[i_rnd]
        val_y_ready[i] = feed.feed_y_ready[i_rnd]
        val_y_ep1[i]   = feed.feed_y_ep1[i_rnd]
        val_y_ep2[i]   = feed.feed_y_ep2[i_rnd]
        val_y_ep3[i]   = feed.feed_y_ep3[i_rnd]
        val_y_steal[i] = feed.feed_y_steal[i_rnd]
        i += 1
        print(f"i: {i}, rnd_i: {i_rnd}, i_batch:{feed.i_batch}")

        # save
        if i == p.VAL_SIZE :
            try : os.makedirs(p.VAL_DIR)
            except : pass
            np.savez(f"{p.VAL_DIR}/val_{mode}",
                     m=val_x_m,
                     p=val_x_p,
                     s=val_x_s,
                     h=val_x_h,
                     aux=val_x_aux,
                     ep1=val_x_ep1,
                     ep2=val_x_ep2,
                     ep3=val_x_ep3,
                     si=val_x_si,
                     ym=val_y_main,
                     yr=val_y_ready,
                     yep1=val_y_ep1,
                     yep2=val_y_ep2,
                     yep3=val_y_ep3,
                     ys=val_y_steal)
            print("DONE!")
            break

        # feed初期化
        feed.clear_feed()

    print(total_situation_num)


if __name__ == "__main__" :
    args = sys.argv
    if len(args) == 2 : mode = args[1]
    else :
        print("Usage : " + colored("$ python create_val.py {mode}", "yellow"))
        sys.exit()
    create_val(mode)
