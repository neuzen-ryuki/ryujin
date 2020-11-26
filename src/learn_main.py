# sys

# 3rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ours
from params.Params import *


def create_model() :
    ## 数牌処理CNN
    mps_inputs = keras.layers.Input(shape=(N_ROW_MPS, N_COL, N_PLANE_MPS), name="mps_input")
    x = layers.Conv2D(N_CH_MPS, (3,2), activation="relu", name="mpsConv1")(mps_inputs)
    x = layers.BatchNormalization(name="mpsConv1_BN")(x)
    x = layers.Conv2D(N_CH_MPS, (3,2), activation="relu", name="mpsConv2")(x)
    x = layers.BatchNormalization(name="mpsConv2_BN")(x)
    x = layers.Conv2D(N_CH_MPS, (3,2), activation="relu", name="mpsConv3")(x)
    x = layers.BatchNormalization(name="mpsConv3_BN")(x)
    mps_outputs = layers.Flatten(name="mps_Flatten")(x)

    mps_model = keras.Model(mps_inputs, mps_outputs)
    m_inputs = keras.layers.Input(shape=(N_ROW_MPS, N_COL, N_PLANE_MPS), name="m_input")
    p_inputs = keras.layers.Input(shape=(N_ROW_MPS, N_COL, N_PLANE_MPS), name="p_input")
    s_inputs = keras.layers.Input(shape=(N_ROW_MPS, N_COL, N_PLANE_MPS), name="s_input")

    m_outputs = mps_model(m_inputs)
    p_outputs = mps_model(p_inputs)
    s_outputs = mps_model(s_inputs)

    ## 字牌処理CNN
    char_inputs = keras.layers.Input(shape=(N_ROW_CHAR, N_COL, N_PLANE_CHAR), name="char_input")
    x = layers.Conv2D(N_CH_CHAR, (3,2), activation="relu", name="charConv1")(char_inputs)
    x = layers.BatchNormalization(name="charConv1_BN")(x)
    x = layers.Conv2D(N_CH_CHAR, (3,2), activation="relu", name="charConv2")(x)
    x = layers.BatchNormalization(name="charConv2_BN")(x)
    x = layers.Conv2D(N_CH_CHAR, (3,2), activation="relu", name="charConv3")(x)
    x = layers.BatchNormalization(name="charConv3_BN")(x)
    char_outputs = layers.Flatten(name="char_Flatten")(x)

    ## MLP
    auxiliary_inputs = keras.layers.Input(shape=(N_AUX_INPUT, ), name="auxiliary_input")
    dense_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, char_outputs, auxiliary_inputs])
    x = layers.Dense(N_UNITS, activation="relu", name="MAIN_MLP1")(dense_inputs)
    x = layers.BatchNormalization(name="MAIN_MLP1_BN")(x)
    x = layers.Dense(N_UNITS, activation="relu", name="MAIN_MLP2")(x)
    x = layers.BatchNormalization(name="MAIN_MLP2_BN")(x)
    x = layers.Dense(N_UNITS, activation="relu", name="MAIN_MLP3")(x)
    x = layers.BatchNormalization(name="MAIN_MLP3_BN")(x)
    x = layers.Dense(N_UNITS, activation="relu", name="MAIN_MLP4")(x)
    x = layers.BatchNormalization(name="MAIN_MLP4_BN")(x)
    x = layers.Dense(N_UNITS, activation="relu", name="MAIN_MLP5")(x)
    dense_outputs = layers.BatchNormalization(name="MAIN_MLP5_BN")(x)

    ## 出力
    outputs = layers.Dense(N_OUTPUT, activation="softmax", name="main_output")(dense_outputs)
    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, char_inputs, auxiliary_inputs], outputs=outputs)

    return model


if __name__ ==  "__main__" :
    model = create_model()
