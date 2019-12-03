import numpy as np
import matplotlib.pyplot as plt


N=int(100)
a=np.random.rand(N)

#method cumsum into mean
b=np.cumsum(a)
b2=np.mean(b)

d=[]
for j,i in enumerate(a):
    d.append(i*(len(a)-j)/(len(a)))
d2=np.cumsum(d)
#method sum 
c=np.sum(a)
plt.figure(1)
plt.cla()
plt.scatter(np.arange(len(a)),a,label="data")
plt.plot(b,label="cumsum")
plt.plot([0,len(a)],[c,c],label="sum")
plt.plot([0,len(a)],[b2,b2],label="method cumsum mean")
plt.plot(d2,label="current method in program")
plt.legend()
plt.grid()