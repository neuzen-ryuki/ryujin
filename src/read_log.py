# std
import os
import sys
import xml.etree.ElementTree as et

# 3rd

# ours
from log_game import Game
from log_player import Player
from feed import Feed
from mytools import fntime


@fntime
def proc_month(files) :
    for file_name in files :
        try :
            tree = et.parse(path + file_name)
        except :
            print(file_name)
            continue

        root = tree.getroot()
        # game = Game(root, file_name, feed_mode=False, feed=None) # feedは作らない時
        game = Game(root, file_name, feed_mode=True, feed=feed) # feedを作りたい時
        game.read_log()


if __name__ == "__main__" :
    year = int(input("Input year : "))
    feed = Feed()
    for month in range(1, 13) :
        path = f"../data/xml/{year}/{month:02}/"
        dir_components = os.listdir(path)
        files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
        print(f"year:{year}, month:{month}")
        proc_month(files)
