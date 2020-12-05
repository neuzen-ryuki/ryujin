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
from params import Params as p
from cymod.feed import Feed
from log_game import Game


# 一旦feedを作ってから学習させる時にfitに渡すgenerator
def load_feed() :
    temp = np.zeros((p.BATCH_SIZE, p.STEAL_OUTPUT))
    dir_components = os.listdir(p.FEED_DIR)
    files = [f for f in dir_components if os.path.isfile(os.path.join(p.FEED_DIR, f))]
    for epoch in range(p.EPOCH) :
        for file_name in files :
            feed = np.load(f"{p.FEED_DIR}{file_name}")
            yield [feed["m"], feed["p"], feed["s"], feed["h"], feed["aux"]], [feed["y"], temp]


# feedを作りながら学習させる時にfitに渡すgenerator
def generate_feed() :
    feed = Feed()
    for year in range(2019, 2010, -1) :
        # val_dataを作る用に12月のファイルは使わないようにしている
        for month in range(1, 12) :
            path = f"../data/xml/{year}/{month:02}/"
            dir_components = os.listdir(path)
            files = [f for f in dir_components if os.path.isfile(os.path.join(path, f))]
            for file_name in files :
                try : tree = et.parse(path + file_name)
                except : continue
                root = tree.getroot()
                game = Game(root, file_name, feed=feed)
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

    ## COMMON_DENSE
    aux_inputs = keras.layers.Input(shape=(p.AUX_INPUT, ), name="aux_input")
    common_inputs = layers.concatenate([m_outputs, p_outputs, s_outputs, h_outputs, aux_inputs])
    x = layers.Dense(p.UNITS, activation="relu", name="COMMON_MLP")(common_inputs)
    x = layers.BatchNormalization(name="COMMON_BN")(x)
    x = layers.Dense(p.UNITS, activation="relu", name="COMMON_OUT")(x)
    common_outputs = layers.BatchNormalization(name="COMMON_OUT_BN")(x)

    ## MAIN_DENSE
    main_inputs = layers.Dense(p.UNITS, activation="relu", name="MAIN_MLP")(common_outputs)
    x = layers.BatchNormalization(name="MAIN_BN")(main_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="MAIN_OUT")(x)
    x = layers.BatchNormalization(name="MAIN_OUT_BN")(x)
    main_outputs = layers.Dense(p.MAIN_OUTPUT, activation="softmax", name="main_output")(x)

    ## STEAL_DENSE
    steal_info = keras.layers.Input(shape=(p.SI_INPUT, ), name="steal_tile")
    steal_inputs = layers.concatenate([common_outputs, steal_info])
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_MLP")(steal_inputs)
    x = layers.BatchNormalization(name="STEAL_BN")(steal_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="STEAL_OUT")(x)
    x = layers.BatchNormalization(name="STEAL_OUT_BN")(x)
    steal_outputs = layers.Dense(p.STEAL_OUTPUT, activation="softmax", name="steal_output")(x)

    ## READY_DENSE
    ready_inputs = layers.Dense(p.UNITS, activation="relu", name="READY_MLP")(common_outputs)
    x = layers.BatchNormalization(name="READY_BN")(ready_inputs)
    x = layers.Dense(p.UNITS, activation="relu", name="READY_OUT")(x)
    x = layers.BatchNormalization(name="READY_OUT_BN")(x)
    ready_outputs = layers.Dense(p.READY_OUTPUT, activation="softmax", name="ready_output")(x)

    ## create_model
    model = keras.Model(inputs=[m_inputs, p_inputs, s_inputs, h_inputs, steal_info, aux_inputs],
                        outputs=[main_outputs, steal_outputs, ready_outputs])

    ## モデルの形を出力
    # keras.utils.plot_model(model, f"{p.DIR}model/ryujin_model.png", show_shapes=True)
    # sys.exit()

    # setting up model
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(
        optimizer=adam,
        loss=categorical_crossentropy,
        metrics=["accuracy"],
        )

    return model


if __name__ ==  "__main__" :
    # create
    model = create_model()
    # load data for validation
    # val = np.load(f"{p.DIR}val.npz")

    val_x_m   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_p   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_s   = np.zeros((p.BATCH_SIZE, p.MPS_ROW,   p.COL, p.PLANE))
    val_x_h   = np.zeros((p.BATCH_SIZE, p.HONOR_ROW, p.COL, p.PLANE))
    val_x_si  = np.zeros((p.BATCH_SIZE, p.SI_INPUT)) # si means "steal info"
    val_x_aux = np.zeros((p.BATCH_SIZE, p.AUX_INPUT))
    val_x     = [val_x_m, val_x_p, val_x_s, val_x_h, val_x_si, val_x_aux]

    val_main_y  = np.zeros((p.BATCH_SIZE, p.MAIN_OUTPUT))
    val_steal_y = np.zeros((p.BATCH_SIZE, p.STEAL_OUTPUT))
    val_ready_y = np.zeros((p.BATCH_SIZE, p.READY_OUTPUT))
    val_y = [val_main_y, val_steal_y, val_ready_y]

    # setting up learning records
    fpath = p.DIR + "model/weights.{epoch:02d}-{val_loss:.6f}.hdf5"
    cbf1 = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor="val_loss", mode="auto")
    cbf2 = keras.callbacks.CSVLogger(f"{p.DIR}result_history.csv")

    # learning
    model.fit(
        generate_feed(),
        validation_data=(val_x, val_y),
        steps_per_epoch=p.VALIDATE_SPAN,
        epochs=int((p.TOTAL_BATCHS_NUM // p.VALIDATE_SPAN) * p.EPOCH),
        verbose=1,
        callbacks=[cbf1,cbf2])
