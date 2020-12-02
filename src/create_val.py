# std
import os
import sys
import random
import xml.etree.ElementTree as et

# 3rd
import numpy as np

# ours
from params import Params as p
from log_game import Game
from log_player import Player
from cymod.feed import Feed
from mytools import fntime


@fntime
def create_val() :
    val_x_m   = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_p   = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_s   = np.zeros((p.VAL_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_h   = np.zeros((p.VAL_SIZE, p.HONOR_ROW, p.COL, p.PLANE))
    val_x_aux = np.zeros((p.VAL_SIZE, p.AUX_INPUT))
    val_y     = np.zeros((p.VAL_SIZE, p.OUTPUT))

    feed = Feed()
    dir_components = os.listdir(p.VAL_XML_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.VAL_XML_DIR, f))]
    i = 0
    for file_name in files :
        try :
            tree = et.parse(p.VAL_XML_DIR + file_name)
        except :
            print(file_name)
            continue

        root = tree.getroot()
        game = Game(root, file_name, feed_mode=True, feed=feed)
        game.read_log()
        i_rnd = random.randint(0,(feed.i_batch - 1))
        val_x_m[i]   = feed.feed_x_m[i_rnd]
        val_x_p[i]   = feed.feed_x_p[i_rnd]
        val_x_s[i]   = feed.feed_x_s[i_rnd]
        val_x_h[i]   = feed.feed_x_h[i_rnd]
        val_x_aux[i] = feed.feed_x_aux[i_rnd]
        val_y[i]     = feed.feed_y[i_rnd]
        i += 1
        print(f"i: {i}, rnd_i: {i_rnd}")
        if i == p.VAL_SIZE :
            np.savez(p.DIR + f"val{p.VAL_SIZE}",
                     m=val_x_m,
                     p=val_x_p,
                     s=val_x_s,
                     h=val_x_h,
                     aux=val_x_aux,
                     y=val_y)
            print("DONE!")
            sys.exit()
        feed.init_feed()


if __name__ == "__main__" :
    create_val()
