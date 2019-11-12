import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import scipy, scipy.stats
import matplotlib.pyplot as plt


def bootstrap_method_linear_fit(x,y,confidence_interval=0.95,samples=300):
    
    list_slopes = []
    constant=[]
    
    for i in range(samples):
        rand_value=np.random.randint(len(x),size=len(x))
        x_s,y_s=x[rand_value],y[rand_value]
        m_s, b_s, r, p, err = scipy.stats.linregress(x_s,y_s)
        list_slopes.append(m_s)
        constant.append(b_s)
        
    a=np.array(constant)
    b=np.array(list_slopes)
     
    a_mean,a_std=mean_confidence_interval(a, confidence_interval)
    b_mean,b_std=mean_confidence_interval(b, confidence_interval)
    
    
    
    return(a_mean,a_std,b_mean,b_std)




def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return (m,h)

def bootstrap_linear_fit_plot_test(x,y,title_plot="test",confidence_interval=0.95,samples=300):
    #linear fit without bootstrap
    m_s_1, b_s_1, r, p, err = scipy.stats.linregress(x,y)
    
    
    #making list for values slopes and constants 
    list_slopes = []
    constant=[]
    
    for i in range(samples):
        rand_value=np.random.randint(len(x),size=len(x))
        x_s,y_s=x[rand_value],y[rand_value]
        m_s, b_s, r, p, err = scipy.stats.linregress(x_s,y_s)
        list_slopes.append(m_s)
        constant.append(b_s)
        
    a=np.array(constant)
    b=np.array(list_slopes)
    
    # calculating the mean and the certainty/confidence interval
    a_mean,a_std=mean_confidence_interval(a, confidence_interval)
    b_mean,b_std=mean_confidence_interval(b, confidence_interval)
    
    # preparing data by doing some array modification
    x_new=np.expand_dims(np.arange(0,50,0.1),axis=1)
    b=np.expand_dims(b,axis=1)
    
    # all bootstrap fits
    y_new=a+x_new*b.T
    
    # linear fit with bootstrap
    y_fit=a_mean+x_new*b_mean
    
    # linear fit without bootstrap
    y_lin=b_s_1+m_s_1*x_new
    
    
    plt.scatter(x,y,c=np.linspace(0.1,1,len(x))[::-1],cmap="gray",label="raw data",zorder=20)
    plt.plot(x_new,y_fit,c='b',label="fit bootstrap",zorder=15)
    plt.plot(x_new,y_lin,c="g",label="lin fit",zorder=15)
    plt.plot(x_new,y_new,c="r",alpha=0.2,zorder=10)
    plt.title(r"{0}:   y=({1:.2f}$\pm${2:.2f})+({3:.2f}$\pm${4:.2f})*x)".format(str(title_plot),a_mean,a_std,b_mean,b_std))
    plt.xlim([0,max(x*1.1)])
    plt.ylim([0,max(y*1.1)])
    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", borderaxespad=1)
#    plt.tight_layout()
#    plt.grid()
    
    
    
    
    

if __name__ == '__main__':
    start=time.time()
    N=100
    x=np.random.rand(N)

    y=np.random.rand(N)-0.5+10+2*x
    bootstrap_linear_fit_plot_test(x,y)
#    a_mean,a_std,b_mean,b_std=bootstrap_method_linear_fit(x,y)
    end=time.time()
    print(end-start)
    
    