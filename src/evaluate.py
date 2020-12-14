# sys

# 3rd
import numpy as np
from termcolor import colored

# ours
from pymod.params import Params as p
from pymod.model import create_model
from mytools import fntime

@fntime
def evaluate():
    mode = input(colored("Input the mode. (main, steal, ready): ","yellow", attrs=["bold"]))
    model = create_model(mode)
    val = np.load(f"{p.VAL_DIR}/val_{mode}.npz")
    val_x = [val["m"], val["p"], val["s"], val["h"], val["si"], val["aux"]]
    val_y = [val["my"], val["sy"], val["ry"]]

    model.evaluate(val_x, val_y)


if __name__ == "__main__" :
    evaluate()

