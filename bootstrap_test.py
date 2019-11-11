import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import random
import scipy, scipy.stats
# fake xy data:
a = -0.5
b = 1
n = 200
x = np.random.randn(n)
y = a*x + b + 1*np.random.randn(n)


plt.plot(x,y,'o', zorder=2)
data = zip(x,y)
n = len(x)


list_slopes = []
constant=[]
for i in range(1000):
#    sampled_data = [ np.random.rand for i in range(n) ]
    rand_value=np.random.randint(n,size=n)
    x_s,y_s = x[rand_value],y[rand_value]

    
    m_s, b_s, r, p, err = scipy.stats.linregress(x_s,y_s)
    ymodel = m_s*x_s + b_s
    list_slopes.append(m_s)
    constant.append(b_s)
    
    plt.plot(x_s,ymodel,'r-',zorder=1,alpha=0.01)

#plt.fill_between(xd, yl,yu, alpha=0.3, facecolor='blue',edgecolor='none',zorder=5)
plt.xlabel("X"); plt.ylabel('Y');plt.show()


a=np.array(constant)
b=np.array(list_slopes)

a_mean=np.mean(a)
a_std=np.std(a)

b_mean=np.mean(b)
b_std=np.std(b)




