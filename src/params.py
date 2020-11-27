class ConstMeta(type):
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError(f'Can\'t rebind const ({name})')
        else: self.__setattr__(name, value)


class Params(metaclass=ConstMeta) :
    FEED_FILES_NUM  = 10000
    BATCH_SIZE      = 5000
    VALIDATE_SPAN   = 2000
    EPOCH           = 3

    MPS_ROW         = 9
    MPS_PLANE       = 2
    MPS_CH          = 128

    HONOR_ROW       = 7
    HONOR_CH        = 128
    HONOR_PLANE     = 2

    UNITS           = 256

    COL             = 4
    OUTPUT          = 38

    YEAR            = "2019"
    VERSION         = "v0.1"
    SAVE_DIR        = f"../data/{VERSION}/model/"
    FEED_DIR        = f"../data/{VERSION}/feed/{YEAR}/"
    VAL_DIR         = f"../data/{VERSION}/val/"
