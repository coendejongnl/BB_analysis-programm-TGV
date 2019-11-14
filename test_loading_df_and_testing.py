import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import interpolate
from timeit import default_timer as timer
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


t1=timer()

#dataDir="C:\\Users\\Media Markt\\Google Drive\\1 stage BlueBattery\\python\\data trial 1\\2019-09-02 10-26\\"
#
#df=pd.read_csv(dataDir + 'CT01_CONDUCTIVITY.bcp', header=None, delimiter='\t')
dataDir=file_path
df=pd.read_csv(dataDir , header=None, delimiter='\t')
t2=timer()

#print(t2-t1)

values=df.sort_values([3]).values[:,4]
time=df.sort_values([3]).values[:,3].astype(dtype="datetime64")

dtime=np.abs(np.diff(time).astype(dtype="float")/1000)
dtime=np.append(dtime,1)
time=np.cumsum(dtime)











t2=timer()

#print(t2-t1)