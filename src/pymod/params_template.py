
### 編集してparams.pyに改名して使う

import os
import sys

class ConstMeta(type):
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError(f"Can\'t rebind const ({name})")
        else: self.__setattr__(name, value)


class Params(metaclass=ConstMeta) :
    # 学習関連
    EPOCH            = 2
    YEARS_NUM        = 9
    VAL_SIZE         = 100000
    VALIDATE_SPAN    = 5000
    BATCH_SIZE       = 1000
    ENDLESS          = 100000 # feed_generator()がデータを切らさないくらいの十分大きな数

    # Feedの形式関連
    MPS_ROW          = 9
    HONOR_ROW        = 7
    COL              = 4
    PLANE            = 12
    AUX_INPUT        = 46
    SI_INPUT         = 42
    MAIN_OUTPUT      = 38
    STEAL_OUTPUT     = 6
    READY_OUTPUT     = 2

    # モデルのパラメータ数関連
    MPS_CH           = 128
    HONOR_CH         = 128
    COMMON_UNITS1    = 768
    COMMON_UNITS2    = 512
    UNITS            = 256

    # ファイル保存関連
    VERSION          = "temp"
    DATA_DIR         = os.path.join(os.path.dirname(__file__), "../../data")
    DIR              = f"{DATA_DIR}/{VERSION}"
    XML_DIR          = f"{DATA_DIR}/xml"
    FEED_DIR         = f"{DIR}/feed"
    VAL_DIR          = f"{DIR}/val"
    MODEL_DIR        = f"{DIR}/model"
    RESULT_DIR       = f"{DIR}/result"
    SAVED_DIR        = f"{DIR}/saved"
    VAL_XML_DIR      = f"{XML_DIR}/val/"

    # モデル関連
    MAIN_MODEL       = SAVED_DIR + "/main/"  + ""
    STEAL_MODEL      = SAVED_DIR + "/steal/" + ""
    READY_MODEL      = SAVED_DIR + "/ready/" + ""