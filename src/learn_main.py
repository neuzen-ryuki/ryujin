# sys
import os

# 3rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ours
from params import Params as p


def generate_feed() :
    dir_components = os.listdir(p.FEED_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.FEED_DIR, f))]

    for epoch in range(p.EPOCH) :
        for file_name in files :
            feed = np.load(f"{p.FEED_DIR}{file_name}")
            yield [feed["m"], feed["p"], feed["s"], feed["h"]], feed["y"]


def create_model() :
    ## 数牌処理CNN
    mps_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.MPS_PLANE), name="mps_input")
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv1")(mps_inputs)
    x = layers.BatchNormalization(name="mpsConv1_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv2")(x)
    x = layers.BatchNormalization(name="mpsConv2_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv3")(x)
    x = layers.BatchNormalization(name="mpsConv3_BN")(x)
    mps_outputs = layers.Flatten(name="mps_Flatten")(x)

    mps_model = keras.Model(mps_inputs, mps_outputs)
    m_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.MPS_PLANE), name="m_input")
    p_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.MPS_PLANE), name="p_input")
    s_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.MPS_PLANE), name="s_input")

    m_outputs = mps_model(m_inputs)
    p_outputs = mps_model(p_inputs)
    s_outputs = mps_model(s_inputs)

    ## 字牌処理CNN
    h_inputs = keras.layers.Input(shape=(p.HONOR_ROW, p.COL, p.HONOR_PLANE), name="honor_input")
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv1")(h_inputs)
    x = layers.BatchNormalization(name="honorConv1_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv2")(x)
    x = layers.BatchNormalization(name="honorConv2_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv3")(x)
    x = layers.BatchNormalization(name="honorConv3_BN")(x)
    h_outputs = layers.Flatten(name="honor_Flatten")(x)

    ## MLP
    dense_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, h_outputs])
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP1")(dense_inputs)
    x = layers.BatchNormalization(name="MAIN_MLP1_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP2")(x)
    x = layers.BatchNormalization(name="MAIN_MLP2_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP3")(x)
    dense_outputs = layers.BatchNormalization(name="MAIN_MLP3_BN")(x)

    ## 出力
    outputs = layers.Dense(p.OUTPUT, activation="softmax", name="main_output")(dense_outputs)
    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs], outputs=outputs)

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
    val = np.load(f"{p.VAL_DIR}val.npz")

    # setting up learning records
    fpath = p.SAVE_DIR + "weights.{epoch:02d}-{val_loss:.6f}.hdf5"
    cbf1 = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor="val_loss", mode="auto")
    cbf2 = keras.callbacks.CSVLogger(p.SAVE_DIR + "result_history.csv")

    # learning
    model.fit(
        generate_feed(),
        steps_per_epoch=p.VALIDATE_SPAN,
        validation_data=([val["m"], val["p"], val["s"], val["h"]], val["y"]),
        epochs=int((p.FEED_FILES_NUM // p.VALIDATE_SPAN) * p.EPOCH),
        verbose=1,
        callbacks=[cbf1,cbf2])
