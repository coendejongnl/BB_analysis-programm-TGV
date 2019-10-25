import directoryGetter
import calculations as calc
import dataGetters as getters
import dataReader
import pathlib

import os
import sys

from cycler import cycler #used to cycle through styles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from function_file import  *

import Tkinter
import tkFileDialog
import os

root = Tkinter.Tk()
root.withdraw() #use to hide tkinter window

currdir = os.getcwd()
tempdir = tkFileDialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
if len(tempdir) > 0:
    print(1)