# sys
import sys
import os
import xml.etree.ElementTree as et

# 3rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from termcolor import colored

# ours
from pymod.params import Params as p
from pymod.game import Game
from pymod.model import create_model, load_model
from cymod.feed import Feed
from cymod.shanten import ShantenNumCalculator


# 一旦feedを作ってから学習させる時にfitに渡すgenerator
def load_feed() :
    temp = np.zeros((p.BATCH_SIZE, p.STEAL_OUTPUT))
    dir_components = os.listdir(p.FEED_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.FEED_DIR, f))]
    for epoch in range(p.EPOCH) :
        for file_name in files :
            feed = np.load(f"{p.FEED_DIR}/{file_name}")
            yield [feed["m"], feed["p"], feed["s"], feed["h"], feed["aux"]], [feed["y"], temp]


# feedを作りながら学習させる時にfitに渡すgenerator
def generate_feed(mode:str) :
    feed = Feed(mode=mode, batch_size=p.BATCH_SIZE)
    shanten_calculator = ShantenNumCalculator()
    game = Game(mode, sc=shanten_calculator ,feed=feed)
    for epoch in range(p.EPOCH) :
        for year in range(2019, (2019-p.YEARS_NUM), -1) :
            print("")
            print(colored(f"epoch:{epoch}, year:{year}", "green", attrs=["bold"]))
            for month in range(1, 12) : # validation用に12月のログは使わないようにしている
                path = f"{p.XML_DIR}/{year}/{month:02}/"
                dir_components = os.listdir(path)
                files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
                for file_name in files :
                    try : tree = et.parse(path + file_name)
                    except : continue
                    root = tree.getroot()
                    game.init_game(root, file_name)
                    yield from game.generate_feed()


if __name__ ==  "__main__" :
    # create model
    args = sys.argv
    if len(args) == 3 and args[1] == "new" and args[2] in {"main", "steal", "ready"} :
        mode = args[2]
        model = create_model(mode)
    elif len(args) == 3 and args[1] == "load" :
        mode = args[2]
        model = load_model(mode)
    else :
        print(colored("Usage :", "green", attrs=["bold"]))
        print("    If you want to create a new model   : " + colored("$ python learn.py new {\"main\", \"steal\", \"ready\"} ", "yellow"))
        print("    If you want to load a trained model : " + colored("$ python learn.py load {\"main\", \"steal\", \"ready\"} ", "yellow"))
        sys.exit()

    # load data for validation
    val = np.load(f"{p.VAL_DIR}/val_{mode}.npz")
    val_x = [val["m"],
             val["p"],
             val["s"],
             val["h"],
             val["aux"],
             val["ep1"],
             val["ep2"],
             val["ep3"]]
    val_y = [val["ym"],
             val["yr"],
             val["yep1"],
             val["yep2"],
             val["yep3"]]
    if mode == "steal" :
        val_x.append(val["si"])
        val_y = [val["ys"],
                 val["yred"],
                 val["yep1"],
                 val["yep2"],
                 val["yep3"]]

    # setting up learning records
    try : os.makedirs(p.SAVED_DIR + f"/{mode}")
    except : pass
    saved_file_name = p.SAVED_DIR + f"/{mode}/" + "{epoch}-{val_loss:.6f}.h5"
    cbf1 = keras.callbacks.ModelCheckpoint(filepath=saved_file_name,
                                           save_weights_only=False,
                                           monitor="val_loss")
    cbf2 = keras.callbacks.CSVLogger(f"{p.RESULT_DIR}/{mode}_history.csv")

    # learning
    # 途中でgenerate_feed()がfeedを吐かなくなってもsaveするようtry-exceptで制御
    # try :
    #     model.fit(
    #         generate_feed(mode),
    #         validation_data=(val_x, val_y),
    #         steps_per_epoch=p.VALIDATE_SPAN,
    #         epochs=p.ENDLESS,
    #         verbose=1,
    #         callbacks=[cbf1, cbf2])
    # except : pass
    # model.save(p.SAVED_DIR + "/{mode}_last.h5")

    ## for debug
    model.fit(
        generate_feed(mode),
        validation_data=(val_x, val_y),
        steps_per_epoch=p.VALIDATE_SPAN,
        epochs=p.ENDLESS,
        verbose=1,
        callbacks=[cbf1, cbf2])
