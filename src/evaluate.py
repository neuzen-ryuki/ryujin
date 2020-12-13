# sys

# 3rd
import numpy as np

# ours
from pymod.params import Params as p
from pymod.model import create_model
from mytools import fntime

@fntime
def evaluate():
    # load data for validation
    model = create_model()
    load_name = ""
    if   p.MAIN_MODE : load_name = "val_main"
    elif p.MAIN_MODE : load_name = "val_steal"
    elif p.MAIN_MODE : load_name = "val_ready"
    val = np.load(f"{p.VAL_DIR}/{load_name}.npz")
    val_x = [val["m"], val["p"], val["s"], val["h"], val["si"], val["aux"]]
    val_y = [val["my"], val["sy"], val["ry"]]

    model.evaluate(val_x, val_y)


if __name__ == "__main__" :
    evaluate()

