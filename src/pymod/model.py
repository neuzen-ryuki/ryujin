# sys
import sys

# 3rd
from tensorflow import keras
from tensorflow.keras import layers
from termcolor import colored

# ours
from .params import Params as p


# 数牌処理CNNを構築
def create_mps_network() :
    mps_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="mps_input")
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv1")(mps_inputs)
    x = layers.BatchNormalization(name="mpsConv1_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv2")(x)
    x = layers.BatchNormalization(name="mpsConv2_BN")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="mpsConv3")(x)
    x = layers.BatchNormalization(name="mpsConv3_BN")(x)
    mps_outputs = layers.Flatten(name="mps_Flatten")(x)

    mps_model = keras.Model(mps_inputs, mps_outputs)
    m_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="m_inputs")
    p_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="p_inputs")
    s_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="s_inputs")

    m_outputs = mps_model(m_inputs)
    p_outputs = mps_model(p_inputs)
    s_outputs = mps_model(s_inputs)

    return m_inputs, p_inputs, s_inputs, m_outputs, p_outputs, s_outputs


# 字牌処理CNNを構築
def create_honor_network() :
    h_inputs = keras.layers.Input(shape=(p.HONOR_ROW, p.COL, p.PLANE), name="honor_input")
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv1")(h_inputs)
    x = layers.BatchNormalization(name="honorConv1_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv2")(x)
    x = layers.BatchNormalization(name="honorConv2_BN")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="honorConv3")(x)
    x = layers.BatchNormalization(name="honorConv3_BN")(x)
    h_outputs = layers.Flatten(name="honor_Flatten")(x)

    return h_inputs, h_outputs


# 各モデル共通部分のnetworkを構築
def create_common_network() :
    m_inputs, p_inputs, s_inputs, m_outputs, p_outputs, s_outputs = create_mps_network()
    h_inputs, h_outputs = create_honor_network()
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    common_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, h_outputs, aux_inputs])
    x = layers.Dense(p.COMMON_UNITS1, activation="relu", name="COMMON_MLP")(common_inputs)
    x = layers.BatchNormalization(name="COMMON_BN")(x)
    x = layers.Dense(p.COMMON_UNITS2, activation="relu", name="COMMON_OUT")(x)
    common_outputs = layers.BatchNormalization(name="COMMON_OUT_BN")(x)

    return m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, common_outputs


# 打牌判断モデルを構築
def create_main_model() -> keras.Model :
    m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, common_outputs = create_common_network()
    main_inputs = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP")(common_outputs)
    x = layers.BatchNormalization(name="MAIN_BN")(main_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_OUT")(x)
    x = layers.BatchNormalization(name="MAIN_OUT_BN")(x)
    main_outputs = layers.Dense(p.MAIN_OUTPUT, activation="softmax", name="main_output")(x)

    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs], outputs=main_outputs)
    return model


# 副露判断モデルを構築
def create_steal_model() -> keras.Model :
    m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, common_outputs = create_common_network()
    steal_info = keras.layers.Input(shape=(p.SI_INPUT, ), name="steal_tile")
    steal_inputs = layers.concatenate([common_outputs, steal_info])
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_MLP")(steal_inputs)
    x = layers.BatchNormalization(name="STEAL_BN")(steal_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_OUT")(x)
    x = layers.BatchNormalization(name="STEAL_OUT_BN")(x)
    steal_outputs = layers.Dense(p.STEAL_OUTPUT, activation="softmax", name="steal_output")(x)

    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, steal_info], outputs=steal_outputs)
    return model


# 副露判断モデルを構築
def create_ready_model() -> keras.Model :
    m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, common_outputs = create_common_network()
    ready_inputs = layers.Dense(p.UNITS, activation="relu", name="READY_MLP")(common_outputs)
    x = layers.BatchNormalization(name="READY_BN")(ready_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="READY_OUT")(x)
    x = layers.BatchNormalization(name="READY_OUT_BN")(x)
    ready_outputs = layers.Dense(p.READY_OUTPUT, activation="softmax", name="ready_output")(x)

    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs], outputs=ready_outputs)
    # model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs], outputs=common_outputs)
    return model


# modeに対応するモデルを構築
def create_model(mode:str) -> keras.Model :
    if   mode == "main"  : model = create_main_model()
    elif mode == "steal" : model = create_steal_model()
    elif mode == "ready" : model = create_ready_model()


    try : os.makedirs(p.RESULT_DIR)
    except : pass
    keras.utils.plot_model(model, f"{p.RESULT_DIR}/ryujin_model.png", show_shapes=True)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model

