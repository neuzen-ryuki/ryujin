class ConstMeta(type):
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError(f'Can\'t rebind const ({name})')
        else: self.__setattr__(name, value)

class Params(metaclass=ConstMeta) :
    BATCH_SIZE    = 5000
    N_ROW_MPS     = 9
    N_ROW_HONOR   = 7
    N_COL         = 4
    N_PLANE_MPS   = 2
    N_PLANE_HONOR = 2
    N_OUTPUT      = 38
