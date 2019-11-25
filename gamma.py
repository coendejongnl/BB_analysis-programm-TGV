import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def concentration_to_gamma(concentration):
    """This function takes the concentration as an input and uses a interpolation function to create the correct gamma as an output"""
    molality=np.array([0.01,0.02,0.05,0.1,0.2, 0.3,0.5,0.7,1])
    gamma=np.array([0.904,0.875,0.824,0.781,0.734,0.709,0.68,0.664,0.65])
    f=interp1d(molality,gamma)
    
    gamma_new=f(concentration)
    return(gamma_new)

    



if __name__ == '__main__':
    
    molality=np.array([0,0.01,0.02,0.05,0.1,0.2, 0.3,0.5,0.7,1])
    gamma=np.array([1,0.904,0.875,0.824,0.781,0.734,0.709,0.68,0.664,0.65])
    plt.plot(molality,gamma,label="raw data")
    
    f=interp1d(molality,gamma)
    x=np.arange(0,1,.01)
    plt.plot(x,f(x),"rx")