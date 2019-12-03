import numpy as np
import matplotlib.pyplot as plt
n=int(10e6)
c=.2
d1=np.random.rand()
d2=np.random.rand()
a=np.random.normal(d1,2,n)
b=np.random.normal(d2,c,n)
c1=a*b
plt.figure(1)
plt.cla()
plt.hist(a,density=True,stacked=True)
plt.hist(b,density=True,stacked=True,alpha=.5,zorder=5)
#plt.hist(c1,density=True,stacked=True,alpha=.1,zorder=10)

plt.figure(2)
plt.title(str(round(d1,4))+"\t"+str(round(d2,4)))
plt.hist2d(a,b,bins=400)
plt.xlim(d2-4*c,d2+4*c)
plt.ylim(d2-4*c,d2+4*c)