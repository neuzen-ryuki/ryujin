# sys
import os
import xml.etree.ElementTree as et

# 3rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ours
from params import Params as p
from cymod.feed import Feed
from log_game import Game


# 一旦feedを作ってから学習させる時にfitに渡すgenerator
def load_feed() :
    dir_components = os.listdir(p.FEED_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.FEED_DIR, f))]
    for epoch in range(p.EPOCH) :
        for file_name in files :
            feed = np.load(f"{p.FEED_DIR}{file_name}")
            yield [feed["m"], feed["p"], feed["s"], feed["h"], feed["aux"]], feed["y"]


# feedを作りながら学習させる時にfitに渡すgenerator
def generate_feed() :
    feed = Feed()
    for year in range(2019, 2017, -1) :
        # val_dataを作る用に12月のファイルは使わないようにしている
        for month in range(1, 12) :
            path = f"../data/xml/{year}/{month:02}/"
            dir_components = os.listdir(path)
            files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
            for file_name in files :
                try : tree = et.parse(path + file_name)
                except : continue
                root = tree.getroot()
                game = Game(root, file_name, feed_mode=True, feed=feed)
                yield from game.generate_feed()


# Neural Networkモデルを構築
def create_model() :
    ## 数牌処理CNN
    mps_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="mps_input")
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv1")(mps_inputs)
    x = layers.BatchNormalization(name="mpsConv1_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv2")(x)
    x = layers.BatchNormalization(name="mpsConv2_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv3")(x)
    x = layers.BatchNormalization(name="mpsConv3_BN")(x)
    mps_outputs = layers.Flatten(name="mps_Flatten")(x)

    mps_model = keras.Model(mps_inputs, mps_outputs)
    m_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="m_input")
    p_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="p_input")
    s_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="s_input")

    m_outputs = mps_model(m_inputs)
    p_outputs = mps_model(p_inputs)
    s_outputs = mps_model(s_inputs)

    ## 字牌処理CNN
    h_inputs = keras.layers.Input(shape=(p.HONOR_ROW, p.COL, p.PLANE), name="honor_input")
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv1")(h_inputs)
    x = layers.BatchNormalization(name="honorConv1_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv2")(x)
    x = layers.BatchNormalization(name="honorConv2_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv3")(x)
    x = layers.BatchNormalization(name="honorConv3_BN")(x)
    h_outputs = layers.Flatten(name="honor_Flatten")(x)

    ## MLP
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    dense_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, h_outputs, aux_inputs])
    x = layers.Dense(p.UNITS, activation="relu", name="MLP1")(dense_inputs)
    x = layers.BatchNormalization(name="MLP1_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="MLP2")(x)
    x = layers.BatchNormalization(name="MLP2_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="MLP3")(x)
    x = layers.BatchNormalization(name="MLP3_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="MLP4")(x)
    dense_outputs = layers.BatchNormalization(name="MLP4_BN")(x)

    ## 出力
    outputs = layers.Dense(p.OUTPUT, activation="softmax", name="main_output")(dense_outputs)
    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs], outputs=outputs)

    return model


if __name__ ==  "__main__" :
    # create and setting up model
    model = create_model()
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=["categorical_accuracy"])

    # load data for validation
    val = np.load(f"{p.DIR}val.npz")

    # setting up learning records
    fpath = p.DIR + "weights.{epoch:02d}-{val_loss:.6f}.hdf5"
    cbf1 = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor="val_loss", mode="auto")
    cbf2 = keras.callbacks.CSVLogger(p.DIR + "result_history.csv")

    # learning
    model.fit(
        generate_feed(),
        steps_per_epoch=p.VALIDATE_SPAN,
        validation_data=([val["m"], val["p"], val["s"], val["h"], val["aux"]], val["y"]),
        epochs=int((p.TOTAL_BATCHS_NUM // p.VALIDATE_SPAN) * p.EPOCH),
        verbose=1,
        callbacks=[cbf1,cbf2])
