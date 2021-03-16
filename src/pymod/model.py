# sys
import os
import sys

# 3rd
import tensorflow as tf
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


# 手牌読み用MLPを構築
def create_ep_network() :
    ep_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep_input")
    x = layers.Dense(p.EP_UNITS1, activation="relu", name="EP1_MLP")(ep_inputs)
    x = layers.BatchNormalization(name="EP1_BN")(x)
    x = layers.Dense(p.EP_UNITS2, activation="relu", name="EP2_MLP")(x)
    x = layers.BatchNormalization(name="EP2_BN")(x)
    x = layers.Dense(p.EP_UNITS3, activation="relu", name="EP3_MLP")(x)
    x = layers.BatchNormalization(name="EP3_BN")(x)
    x = layers.Dense(p.EP_OUTPUT, activation="relu", name="EP_OUT")(x)
    ep_outputs = layers.BatchNormalization(name="EP_OUT_BN")(x)

    ep_model = keras.Model(ep_inputs, ep_outputs)
    ep1_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep1_inputs")
    ep2_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep2_inputs")
    ep3_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep3_inputs")

    ep1_outputs = ep_model(ep1_inputs)
    ep2_outputs = ep_model(ep2_inputs)
    ep3_outputs = ep_model(ep3_inputs)

    return ep1_inputs, ep2_inputs, ep3_inputs, ep1_outputs, ep2_outputs, ep3_outputs


# 各モデル共通部分のnetworkを構築
def create_common_network() :
    m_inputs, p_inputs, s_inputs, m_outputs, p_outputs, s_outputs = create_mps_network()
    h_inputs, h_outputs = create_honor_network()
    ep1_inputs, ep2_inputs, ep3_inputs, ep1_outputs, ep2_outputs, ep3_outputs = create_ep_network()
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    common_inputs = layers.concatenate([m_outputs,
                                        p_outputs,
                                        s_outputs,
                                        h_outputs,
                                        ep1_outputs,
                                        ep2_outputs,
                                        ep3_outputs,
                                        aux_inputs])
    x = layers.Dense(p.COMMON_UNITS1, activation="relu", name="COMMON_MLP")(common_inputs)
    x = layers.BatchNormalization(name="COMMON_BN")(x)
    x = layers.Dense(p.COMMON_UNITS2, activation="relu", name="COMMON_OUT")(x)
    common_outputs = layers.BatchNormalization(name="COMMON_OUT_BN")(x)

    return m_inputs, p_inputs, s_inputs, h_inputs, ep1_inputs, ep2_inputs, ep3_inputs ,aux_inputs, common_outputs, ep1_outputs, ep2_outputs, ep3_outputs


# mainモデルを構築
def create_main_model() :
    # 数牌処理CNNを構築
    mps_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="mps_input")
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="MPS_Conv1")(mps_inputs)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="MPS_Conv2")(x)
    x = layers.Conv2D(p.MPS_CH, (3,2), activation="relu", name="MPS_Conv3")(x)
    mps_outputs = layers.Flatten(name="MPS_Flatten")(x)
    mps_model = keras.Model(inputs=mps_inputs, outputs=mps_outputs, name="mps_model")

    m_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="m_input")
    p_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="p_input")
    s_inputs = keras.layers.Input(shape=(p.MPS_ROW, p.COL, p.PLANE), name="s_input")

    m_outputs = mps_model(m_inputs)
    p_outputs = mps_model(p_inputs)
    s_outputs = mps_model(s_inputs)

    # 字牌処理CNNを構築
    h_inputs = keras.layers.Input(shape=(p.HONOR_ROW, p.COL, p.PLANE), name="honor_input")
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="HONOR_Conv1")(h_inputs)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="HONOR_Conv2")(x)
    x = layers.Conv2D(p.HONOR_CH, (3,2), activation="relu", name="HONOR_Conv3")(x)
    h_outputs = layers.Flatten(name="honor_Flatten")(x)

    # 手牌読み用MLPを構築
    ep_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep_input")
    x = layers.Dense(p.EP_UNITS1, activation="relu", name="EP_MLP1")(ep_inputs)
    x = layers.Dense(p.EP_UNITS2, activation="relu", name="EP_MLP2")(x)
    x = layers.Dense(p.EP_UNITS3, activation="relu", name="EP_MLP3")(x)
    ep_outputs = layers.Dense(p.EP_OUTPUT, activation="sigmoid", name="ep_output")(x)
    ep_model = keras.Model(inputs=ep_inputs, outputs=ep_outputs, name="ep_model")

    ep1_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep1_input")
    ep2_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep2_input")
    ep3_inputs = keras.layers.Input(shape=(p.EP_INPUT, ), name="ep3_input")

    ep1_outputs = ep_model(ep1_inputs)
    ep2_outputs = ep_model(ep2_inputs)
    ep3_outputs = ep_model(ep3_inputs)

    # 各NN結合後のNNを構築
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    common_inputs = layers.concatenate([m_outputs,
                                        p_outputs,
                                        s_outputs,
                                        h_outputs,
                                        aux_inputs,
                                        ep1_outputs,
                                        ep2_outputs,
                                        ep3_outputs])
    x = layers.Dense(p.COMMON_UNITS1, activation="relu", name="COMMON_MLP1")(common_inputs)
    x = layers.BatchNormalization(name="COMMON_BN1")(x)
    x = layers.Dense(p.COMMON_UNITS2, activation="relu", name="COMMON_MLP2")(x)
    x = layers.BatchNormalization(name="COMMON_BN2")(x)
    x = layers.Dense(p.COMMON_UNITS3, activation="relu", name="COMMON3_MLP3")(x)
    common_outputs = layers.BatchNormalization(name="COMMON_BN3")(x)

    # 打牌判断出力NNを構築
    main_inputs = layers.Dense(p.MAIN_UNITS1, activation="relu", name="MAIN_MLP1")(common_outputs)
    x = layers.BatchNormalization(name="MAIN_BN1")(main_inputs)
    x = layers.Dense(p.MAIN_UNITS2, activation="relu", name="MAIN_MLP2")(x)
    x = layers.BatchNormalization(name="MAIN_BN2")(x)
    x = layers.Dense(p.MAIN_UNITS3, activation="relu", name="MAIN_MLP3")(x)
    x = layers.BatchNormalization(name="MAIN_BN3")(x)
    main_outputs = layers.Dense(p.MAIN_OUTPUT, activation="softmax", name="main_output")(x)

    # 立直判断出力NNを構築
    ready_inputs = layers.Dense(p.READY_UNITS1, activation="relu", name="READY_MLP1")(common_outputs)
    x = layers.BatchNormalization(name="READY_BN1")(ready_inputs)
    x = layers.Dense(p.READY_UNITS2, activation="relu", name="READY_MLP2")(x)
    x = layers.BatchNormalization(name="READY_BN2")(x)
    x = layers.Dense(p.READY_UNITS3, activation="relu", name="READY_MLP3")(x)
    x = layers.BatchNormalization(name="READY_BN3")(x)
    ready_outputs = layers.Dense(p.READY_OUTPUT, activation="softmax", name="ready_output")(x)

    # 各種設定
    ep_opt = keras.optimizers.SGD()
    opt = keras.optimizers.Adam()
    gpu_is_available = tf.config.experimental.list_physical_devices("GPU")
    if gpu_is_available :
        print(colored(f"START LEARNING WITH GPU", "green", attrs=["bold"]))
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    else :
        print(colored(f"START LEARNING WITHOUT GPU", "yellow", attrs=["bold"]))

    ins = [m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, ep1_inputs, ep2_inputs, ep3_inputs]
    outs = [main_outputs, ready_outputs, ep1_outputs, ep2_outputs, ep3_outputs]
    model = keras.Model(inputs=ins, outputs=outs)
    model.compile(
        optimizer=opt,
        loss=[["categorical_crossentropy"],
              ["categorical_crossentropy"],
              ["binary_crossentropy"],
              ["binary_crossentropy"],
              ["binary_crossentropy"]],
        metrics=[["accuracy"],
                 ["accuracy"],
                 ["binary_crossentropy"],
                 ["binary_crossentropy"],
                 ["binary_crossentropy"]])

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
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model


# 立直判断モデルを構築
def create_ready_model() -> keras.Model :
    m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs, common_outputs = create_common_network()
    ready_inputs = layers.Dense(p.UNITS, activation="relu", name="READY_MLP")(common_outputs)
    x = layers.BatchNormalization(name="READY_BN")(ready_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="READY_OUT")(x)
    x = layers.BatchNormalization(name="READY_OUT_BN")(x)
    ready_outputs = layers.Dense(p.READY_OUTPUT, activation="softmax", name="ready_output")(x)

    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, aux_inputs], outputs=ready_outputs)
    return model


# modeに対応するモデルを構築
def create_model(mode:str) -> keras.Model :
    if   mode == "main"  : model = create_main_model()
    elif mode == "steal" : model = create_steal_model()
    elif mode == "ready" : model = create_ready_model()

    try : os.makedirs(p.RESULT_DIR)
    except : pass
    keras.utils.plot_model(model, f"{p.RESULT_DIR}/{mode}_model.png", show_shapes=True)
    model.summary()

    return model


def load_model(mode:str) -> keras.Model :
    if   mode == "main"  : model = keras.models.load_model(p.MAIN_MODEL)
    elif mode == "steal" : model = keras.models.load_model(p.STEAL_MODEL)
    elif mode == "ready" : model = keras.models.load_model(p.READY_MODEL)

    return model


