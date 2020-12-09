# sys
import sys
import os
import xml.etree.ElementTree as et

# 3rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ours
from pymod.params import Params as p
from pymod.game import Game
from pymod.model import create_model
from cymod.feed import Feed


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
def generate_feed() :
    feed = Feed()
    for year in range(2019, (2019-p.YEARS_NUM), -1) :
        for month in range(1, 12) : # val_dataを作る用に12月のログは使わないようにしている
            path = f"{p.XML_DIR}/{year}/{month:02}/"
            dir_components = os.listdir(path)
            files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
            for file_name in files :
                try : tree = et.parse(path + file_name)
                except : continue
                root = tree.getroot()
                game = Game(root, file_name, feed=feed)
                yield from game.generate_feed()


if __name__ ==  "__main__" :
    # create
    model = create_model()

    # load data for validation
    load_name = ""
    if   p.MAIN_MODE : load_name = "val_main"
    elif p.MAIN_MODE : load_name = "val_steal"
    elif p.MAIN_MODE : load_name = "val_ready"
    val = np.load(f"{p.VAL_DIR}/{load_name}.npz")
    val_x = [val["m"], val["p"], val["s"], val["h"], val["si"], val["aux"]]
    val_y = [val["my"], val["sy"], val["ry"]]

    # setting up learning records
    saved_file_name = p.SAVED_DIR + "/{epoch:02d}-{val_loss:.6f}.h5"
    cbf1 = keras.callbacks.ModelCheckpoint(filepath=saved_file_name,
                                           save_weights_only=False,
                                           monitor="val_loss")
    cbf2 = keras.callbacks.CSVLogger(f"{p.RESULT_DIR}/result_history.csv")

    # learning
    # 途中でgenerate_feed()がfeedを吐かなくなってもsaveするようtry-exceptで制御
    try :
        model.fit(
            generate_feed(),
            validation_data=(val_x, val_y),
            steps_per_epoch=p.VALIDATE_SPAN,
            epochs=(p.TOTAL_BATCHS_NUM // p.VALIDATE_SPAN) * p.EPOCH,
            verbose=1,
            callbacks=[cbf1, cbf2])
    except :
        model.save(saved_file_name)

