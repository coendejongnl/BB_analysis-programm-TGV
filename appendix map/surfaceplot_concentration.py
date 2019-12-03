import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d
from matplotlib import cm

df=pd.read_excel("ConcentrationCalibration.xlsx")

cond=df["Conductivity (mS/cm)"].values
conc=df["Concentration (M)"].values
temp=df["Temperature C"].values

cond


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(cond, temp, conc,  edgecolors='grey', alpha=0.5,cmap=cm.coolwarm,antialiased=True)
ax.scatter(cond, temp, conc)
plt.xlabel("conductivity")
plt.ylabel("temperature")
ax.set_zlabel('concentration')

plt.show()

#plt.figure()
#plt.contour(cond,temp,conc)