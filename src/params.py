class ConstMeta(type):
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError(f'Can\'t rebind const ({name})')
        else: self.__setattr__(name, value)


class Params(metaclass=ConstMeta) :
    # 学習関連
    FEED_FILES_NUM  = 10000
    BATCH_SIZE      = 5000
    VALIDATE_SPAN   = 2000
    EPOCH           = 3

    # Feedの形式関連
    MPS_ROW         = 9
    HONOR_ROW       = 7
    COL             = 4
    PLANE           = 12
    AUX_INPUT       = 43
    OUTPUT          = 38

    # モデルのパラメータ数関連
    MPS_CH          = 128
    HONOR_CH        = 128
    UNITS           = 256

    # ファイル保存関連
    VERSION         = "v0.2"
    YEAR            = "2019"
    SAVE_DIR        = f"../data/{VERSION}/model/"
    FEED_DIR        = f"../data/{VERSION}/feed/{YEAR}/"
    VAL_DIR         = f"../data/{VERSION}/val/"
