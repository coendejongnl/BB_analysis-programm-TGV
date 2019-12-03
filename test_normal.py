import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def double_integral(a,b,title="test"):
    x,y=np.meshgrid(a,b)
    z=x*y
    z=z.flatten()
    z= z[~np.isnan(z)]

    plt.figure(1)
    plt.cla()
    n,d,e=plt.hist(z,weights=np.full(len(z),1/len(z)))
    plt.xlim(0,1)
    plt.xlabel("efficiency [-]")
    plt.ylabel("probability [-]")
#    plt.title(r"$ {0:.2f}\pm {1:.2f}$".format(np.mean(z),np.std(z)))
    plt.tight_layout()
    print(np.mean(z))
    print(np.std(z))
    
if __name__ == '__main__':
#    n=1000
#    a=np.random.beta(3,5,n)
#    b=np.random.beta(5,3,n)
    df=pd.read_excel("efficiencies.xlsx",error_bad_lines=False , encoding='utf-8')
    df1=df.to_numpy()
    number=1
    a=df1[:,number*2-2]
    b=df1[:,number*2-1]
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    plt.hist(a,bins=10,weights=np.full(len(a),1/len(a)))
    plt.hist(b,alpha=.3,bins=10,weights=np.full(len(b),1/len(b)))
    plt.xlim(0,1)
    plt.xlabel("efficiency [-]")
    plt.ylabel("probability [-]")
#    plt.hist2d(a,b,100)
    plt.subplot(2,1,2)
#    plt.hist(a*b,bins=100,density=True,stacked=True)
    plt.xlim(0,1)
    x,y=np.meshgrid(a,b)
    z=x*y
    z=z.flatten()
    z= z[~np.isnan(z)]
    plt.hist(z,alpha=0.3,density=True,stacked=True,zorder=5)
    
    plt.show()
    
    double_integral(a,b)
