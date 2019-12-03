'''
Program that performs analysis on data from the Blue Battery Pilot Project in Delft
Requires that all data from the BMS are contained in a single folder and are unmodified (same name, columns, etc.)
Author: Thomas Richards
Version: 1.0
Author: Coen de Jong
Version: 1.1
'''

# ======================================= Imports =================================================
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

# ================================================================================================
dataDir = ''    # This global variable will store the path to the folder 


if __name__ == '__main__':
    loadParams()
    dataDir=importFiles()
    pathlib.Path(str(dataDir)+"data_analysis").mkdir(parents=True, exist_ok=True)
    pathlib.Path(str(dataDir)+"data_analysis/figures").mkdir(parents=True, exist_ok=True)
    
    menu()
