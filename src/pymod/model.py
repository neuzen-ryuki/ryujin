# sys
import sys

# 3rd
from tensorflow import keras
from tensorflow.keras import layers
from termcolor import colored

# ours
from .params import Params as p


# Neural Networkモデルを構築
def create_model(mode:str) :
    ## 学習済みモデルをロードするかどうか決める
    load = input(colored("Load the trained model? (Y/n): ","yellow", attrs=["bold"]))
    if load == "Y" :
        file_name = input(colored("Input .h5 file name (mode:{mode}) : ","yellow", attrs=["bold"]))
        model = keras.models.load_model(f"{p.SAVED_DIR}/{file_name}")
        return model
    else : print(colored("Creating the model...","yellow", attrs=["bold"]))

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

    ## COMMON_MLP
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    common_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, h_outputs, aux_inputs])
    x = layers.Dense(p.COMMON_UNITS1, activation="relu", name="COMMON_MLP")(common_inputs)
    x = layers.BatchNormalization(name="COMMON_BN")(x)
    x = layers.Dense(p.COMMON_UNITS2, activation="relu", name="COMMON_OUT")(x)
    common_outputs = layers.BatchNormalization(name="COMMON_OUT_BN")(x)

    ## MAIN_MLP
    main_inputs = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP")(common_outputs)
    x = layers.BatchNormalization(name="MAIN_BN")(main_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_OUT")(x)
    x = layers.BatchNormalization(name="MAIN_OUT_BN")(x)
    main_outputs = layers.Dense(p.MAIN_OUTPUT, activation="softmax", name="main_output")(x)

    ## STEAL_MLP
    steal_info = keras.layers.Input(shape=(p.SI_INPUT, ), name="steal_tile")
    steal_inputs = layers.concatenate([common_outputs, steal_info])
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_MLP")(steal_inputs)
    x = layers.BatchNormalization(name="STEAL_BN")(steal_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_OUT")(x)
    x = layers.BatchNormalization(name="STEAL_OUT_BN")(x)
    steal_outputs = layers.Dense(p.STEAL_OUTPUT, activation="softmax", name="steal_output")(x)

    ## READY_MLP
    ready_inputs = layers.Dense(p.UNITS, activation="relu", name="READY_MLP")(common_outputs)
    x = layers.BatchNormalization(name="READY_BN")(ready_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="READY_OUT")(x)
    x = layers.BatchNormalization(name="READY_OUT_BN")(x)
    ready_outputs = layers.Dense(p.READY_OUTPUT, activation="softmax", name="ready_output")(x)

    ## create_model
    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, steal_info, aux_inputs],
                        outputs=[main_outputs, steal_outputs, ready_outputs])

    ## モデルの形を出力
    keras.utils.plot_model(model, f"{p.RESULT_DIR}/ryujin_model.png", show_shapes=True)

    ## setting up model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model

