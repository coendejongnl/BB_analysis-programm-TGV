import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def simple_monte_carlo(a,size_sample,labels1):
    
    #random samples
    y=np.empty(shape=(len(a),size_sample))
    for i,x in enumerate(a):
        b=np.random.choice(x,size=size_sample)
        y[i,:]=b
    
    #check which ones wins
    final_results=np.zeros(shape=(len(a),size_sample))
    for i,j in enumerate(range(len(a)),1):
        amax=np.amax(y,axis=0,keepdims=False)
        amax=np.tile(amax,(len(a),1))
        final_results=np.where(amax==y,i,final_results)
        y=np.where(final_results!=0,0,y)
    #counting
    
    
    table=np.zeros(shape=(len(a),len(a)))
     
     
    for i in range(len(a)):
        single_array=final_results[i,:]
        for j in range(len(a)):
            print(i)
            print(j)
            
            table[i,j]=np.sum(np.where(single_array==j+1,1,0))
            
    table=table/size_sample
    print(table)
        
    return(table)
    


    
if __name__ == '__main__':
    a=[]
    n=1000000
    labels=["a","b","c"]
    for i in np.arange(3):
        length=np.random.randint(15,50)
        b=np.random.rand(length)
        a.append(b)
        
    plt.figure(1)
    plt.clf()    
    odds=simple_monte_carlo(a,n,labels)

    for i,j in enumerate(tqdm(a)):
        plt.subplot(2,2,i+1)
        plt.hist(j)
        
#    plt.subplot(2,2,4)
#    plt.pie(odds,autopct='%1.1f%%')
    
    print(odds)
    
    