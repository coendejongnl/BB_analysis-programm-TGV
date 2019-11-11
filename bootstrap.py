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
    
    x_new=np.expand_dims(np.arange(0,50,0.1),axis=1)
    b=np.expand_dims(b,axis=1)
    y_new=a+x_new*b.T
    plt.scatter(x,y,c="b",label="raw data")
    plt.plot(x_new,y_new,c="r",alpha=0.2,zorder=-10)
    plt.title(r"{0}:   y=({1:.2f}$\pm${2:.2f})+({3:.2f}$\pm${4:.2f})*x)".format(str(title_plot),a_mean,a_std,b_mean,b_std))
    plt.xlim([0,max(x*1.1)])
    plt.ylim([0,max(y*1.1)])
    plt.legend()

if __name__ == '__main__':
    start=time.time()
    N=100
    x=np.random.rand(N)

    y=np.random.rand(N)-0.5+10+2*x
    bootstrap_linear_fit_plot_test(x,y)
#    a_mean,a_std,b_mean,b_std=bootstrap_method_linear_fit(x,y)
    end=time.time()
    print(end-start)
    
    