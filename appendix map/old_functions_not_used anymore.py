# ======================================= Imports =================================================
import directoryGetter
import calculations as calc
import dataGetters as getters
import bootstrap 
import dataReader
import pathlib
from datetime import datetime
import dill  

import os
import sys

from cycler import cycler #used to cycle through styles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import scipy, scipy.stats



def resistance():
    
    #################
    #This function is created to determine the total resistance
    #################
    
    
    ## give inputs of all the important data
    try:
        print("Fujifilm | type 10 | perm=0.97 [0.95,0.99])")
        ##https://www.fujifilmmembranes.com/images/IEM_brochure_1_1_-final_small_size.pdf      
        permselectivity=float(input("what is the permselectivity of the membranes? (0-1)\n"))
    except:
        permselectivity=1
    
    while True:
        sensorNum = input('Enter stack number (select from {})\n'.format(calc.usedStacks))
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)
            if sensorNum not in calc.usedStacks:
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid stack number.  Please enter a value in {}'.format(calc.usedStacks, sensorNum))
    
    windows=int(input("what is the window to determine the resistance? (only integers)\n"))
    

    
    ## loading data
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    filterData=True
    
    
    # waterlevels only 1 - 3 are important
    level3 = getters.getLevelData(dataDir, 3) 
    # conductivity
    conductivities = np.array([getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)])
    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*293.15) for i in range(len(conductivities))] 

    
    ## determening boundaries
    
    sensorUsed=sensorNum
    includeCycles=True
    if includeCycles:
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)
    
    ## obtaining V_open by taking voltage before cycle begins. taking np.mean(window=100sec)
    
    def V_open(voltage,bound,window=windows): 
        V=voltage[np.arange(bound-window,bound)]
        V=V[V!=0]
        V_mean=np.mean(V)
        return(V_mean)
    
    V_open_c=np.array([])
    V_open_d=np.array([])
    
    for i in range(int(cb.shape[0])):
        V_open_c=np.append(V_open_c,V_open(SV,cb[i,0],window=windows))
        V_open_d=np.append(V_open_d,V_open(LV,db[i,0],window=windows))
        

    ## calculating the open voltage using emperical data (concentration)
    
    def open_voltage_function(C_salt,C_fresh,permselectivity):
        if len(C_salt)>len(C_fresh):
            C_salt=C_salt[0:len(C_fresh)]
        else:
            C_fresh=C_fresh[0:len(C_salt)]
        
        R=8.314
        T=293.15
        F=96485.3329
        gamma=0.68
        
        x=C_salt/C_fresh
        
        E_0=R*T/F*np.log(gamma*x)
        E_open=E_0*2*512*permselectivity
        return(E_open)
    
    
    V_open_5=open_voltage_function(concentrations[1],concentrations[5],permselectivity)
    V_open_3=open_voltage_function(concentrations[1],concentrations[3],permselectivity)
    
    

    
    tank3_diff=np.diff(level3)
    
    print(V_open_5.shape)
    print(V_open_3.shape)
    print(tank3_diff.shape)    
    
    [V_open_3,V_open_5,tank3_diff]=make_same_len([V_open_3,V_open_5,tank3_diff])
    print(V_open_5.shape)
    print(V_open_3.shape)
    print(tank3_diff.shape)
    Nernst_voltage=np.where(tank3_diff<=0,V_open_3,V_open_5)
    
    Nernst_voltage_c=np.array([])
    Nernst_voltage_d=np.array([])

    for i in range(int(cb.shape[0])):
        Nernst_voltage_c=np.append(Nernst_voltage_c,np.mean(Nernst_voltage[cb[i,0]:cb[i,0]+windows]))
        Nernst_voltage_d=np.append(Nernst_voltage_d,np.mean(Nernst_voltage[db[i,0]:db[i,0]+windows]))
    

        
    ### 3th method for resistance and open voltage
    #using a linear fit function on a window of charge discharge cycle as V=V_open (+-)R*I
    # V(x)=A (+-)B*x 
    #x=I        A=V_open        B=R
    voltage_c_lin_fit=np.array([])
    voltage_c_lin_fit_std=np.array([])

    voltage_d_lin_fit=np.array([])
    voltage_d_lin_fit_std=np.array([])

    
    resistance_c_lin_fit=np.array([])
    resistance_d_lin_fit=np.array([])
    resistance_c_lin_fit_std=np.array([])
    resistance_d_lin_fit_std=np.array([])
    
    ## test for errors in code
    def check_for_valid(array):
        print(str(array))
        
        if np.any(np.logical_or(array==np.nan,array==np.inf)):
            print("window contains nan or inf\n")
        else:
            if np.any(np.where(array!=0)):
                
                print("no errors in array\n")
            else:
                print(str(array)+": contains only zeros\n")
    
    
    ## actual fitting and bootstrapping
    for i in range(int(cb.shape[0])):
        ## fitting function
        print(i)
#        try:
        V,V_std,R,R_std=bootstrap.bootstrap_method_linear_fit(SC[cb[i,0]:cb[i,0]+windows],SV[cb[i,0]:cb[i,0]+windows],confidence_interval=0.95,samples=300)
            
#            V,R=np.polynomial.polynomial.Polynomial.fit(SC[cb[i,0]:cb[i,0]+windows],SV[cb[i,0]:cb[i,0]+windows],1).coef
#        except:
#            check_for_valid(SC[cb[i,0]:cb[i,0]+windows])
#            check_for_valid(SV[cb[i,0]:cb[i,0]+windows])
#            V,V_std,R,R_std=[np.nan,np.nan,np.nan,np.nan]

            
        print(i)
        try:
            V2,V2_std,R2,R2_std=bootstrap.bootstrap_method_linear_fit(LC[db[i,0]:db[i,0]+windows],LV[db[i,0]:db[i,0]+windows],confidence_interval=0.95,samples=300)
#            V2,R2=np.polynomial.polynomial.Polynomial.fit(LC[db[i,0]:db[i,0]+windows],LV[db[i,0]:db[i,0]+windows],1).coef
        except:
            check_for_valid(LC[db[i,0]:db[i,0]+windows])
            check_for_valid(LV[db[i,0]:db[i,0]+windows])
            V2,V2_std,R2,R2_std=[np.nan,np.nan,np.nan,np.nan]
        ## appending
        voltage_c_lin_fit=np.append(voltage_c_lin_fit,V)
        voltage_d_lin_fit=np.append(voltage_d_lin_fit,V2)
        voltage_c_lin_fit_std=np.append(voltage_c_lin_fit_std,V_std)
        voltage_d_lin_fit_std=np.append(voltage_d_lin_fit_std,V2_std)
        
        resistance_c_lin_fit=np.append(resistance_c_lin_fit,R)
        resistance_d_lin_fit=np.append(resistance_d_lin_fit,R2)
        resistance_c_lin_fit_std=np.append(resistance_c_lin_fit_std,R_std)
        resistance_d_lin_fit_std=np.append(resistance_d_lin_fit_std,R2_std)
        
    
    ## test to check if the shapes of the arguments are correct
    

    print(V_open_c.shape)
    print(V_open_d.shape)
    print(Nernst_voltage.shape)
    print(Nernst_voltage_c.shape)
    print(Nernst_voltage_d.shape)
    print(cb)
    print(db)
    

#    
    ## create an DataFrame
    
    data={
#            "resistance C":Resistance_c,
#          "resistance D":Resistance_d  ,
#          "voltage C":Voltage_c,
#          "voltage d":Voltage_d,
#          "open V C": open_voltage_c,
#          "open V d": open_voltage_d, 
#          "current c":Current_c,
#          "current d":Current_d,
#          "Nernst voltage C":Nernst_voltage_c,
#          "Nernst voltage D":Nernst_voltage_d,
#          "N resistance C": Resistance_c2,
#          "N resistance D": Resistance_d2,
#          "resistance 2 C":resistance_c_3,
#          "resistance 2 d":resistance_d_3,
#          "resistance 3 C":voltage_c_3,
#          "resistance 3 D":voltage_d_3,
#          "open voltage C":voltage_c_4,
#          "open voltage d":voltage_d_4,
          "open voltage lin c":voltage_c_lin_fit,
          "open voltage lin c std":voltage_c_lin_fit_std,
          "open voltage lin d":voltage_d_lin_fit,
          "open voltage lin d std":voltage_d_lin_fit_std,
          "resistance lin c":resistance_c_lin_fit,
          "resistance lin c std":resistance_c_lin_fit_std,
          "resistance lin d":resistance_d_lin_fit,
          "resistance lin d std":resistance_d_lin_fit_std
              }
    
    df= pd.DataFrame.from_dict(data, orient='index')

    
    df=df.transpose()
    print(df)
    print(Nernst_voltage)
    
    ## save to excel
    save_to_excel(df,"resistance",dataDir,sensorNum)
    
    fig=plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    ## voltage with error plot
    plt.subplot(2,1,1)
    plt.errorbar(np.arange(len(voltage_c_lin_fit))+0.2,voltage_c_lin_fit,yerr=voltage_c_lin_fit_std,fmt='o',label="charging")
    plt.errorbar(np.arange(len(voltage_d_lin_fit))-0.2,voltage_d_lin_fit,yerr=voltage_d_lin_fit_std,fmt='o',label="discharging")
    plt.xlabel("cycle")
    plt.ylabel(r"Voltage")
    plt.xticks(np.arange(0,len(voltage_c_lin_fit)+0.1,1))
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)

    
    ## resistance with error plot
    plt.subplot(2,1,2)
    plt.errorbar(np.arange(len(resistance_c_lin_fit))+0.2,resistance_c_lin_fit,yerr=resistance_c_lin_fit_std,fmt='o',label="charging")
    plt.errorbar(np.arange(len(resistance_d_lin_fit))-0.2,resistance_d_lin_fit,yerr=resistance_d_lin_fit_std,fmt='o',label= "discharging")
    plt.xticks(np.arange(0,len(resistance_c_lin_fit)+0.1,1))
    plt.xlabel("cycle")
    plt.ylabel(r"$\Omega$")
    plt.legend(bbox_to_anchor=(1.02,0.5), loc="center left", borderaxespad=0)
    
    
    
    
# open voltage test with nernst equation
#    plt.plot(V_open_5, label="Nernst cond 5")
#    plt.plot(V_open_3,label="Nernst cond 3")    
##    plt.plot(Nernst_voltage,label="Nernst",alpha=0.4)
#
#    plt.xlabel("time")
#    plt.ylabel("voltage")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    
def resistance_test_linear_plot():
    
    while True:
        sensorNum = input('Enter stack number (select from {})\n'.format(calc.usedStacks))
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)
            if sensorNum not in calc.usedStacks:
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid stack number.  Please enter a value in {}'.format(calc.usedStacks, sensorNum))

    windows=int(input("what is the window to determine the resistance? (only integers)\n"))
    
    test_cycles=input("what are the cycles you want to check? \n [seperated by a comma (,)] \n")
    test_cycles=np.array(test_cycles.split(sep=","),dtype=int)

    
    ## loading data
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
        
    sensorUsed=sensorNum
    includeCycles=True
    
    if includeCycles:
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)
    
    
    for a,i in enumerate(test_cycles):
        
        if a%3==0:
            plt.figure(a)
        plt.subplot(3,2,(a%3)*2+1)
        bootstrap.bootstrap_linear_fit_plot_test(SC[cb[i,0]:cb[i,0]+windows],SV[cb[i,0]:cb[i,0]+windows],title_plot="charge cycle "+str(i))
        plt.subplot(3,2,(a%3)*2+2)
        bootstrap.bootstrap_linear_fit_plot_test(LC[db[i,0]:db[i,0]+windows],LV[db[i,0]:db[i,0]+windows],title_plot="discharge cycle "+str(i))
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    
    
def resistance_evolution():
    #this method calculates the resistance for  a windown of points    
    #assumptions V_open is constant and resistance is evolving
    #V_open is calculated using the first points with   V!=0
    #linear fit is used for simplicity

#    dataDir="C:\\Users\\Media Markt\\Google Drive\\1 stage BlueBattery\\python\\data trial 1\\2019-09-02 10-26\\"

    while True:
        sensorNum = input('Enter stack number (select from {})\n'.format(calc.usedStacks))
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)
            if sensorNum not in calc.usedStacks:
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid stack number.  Please enter a value in {}'.format(calc.usedStacks, sensorNum))

    windows=int(input("what is the window to determine the resistance? (only integers)\n"))
    
    test_cycles=input("what are the cycles you want to check? \n [seperated by a comma (,)] \n")
    try:
        test_cycles=np.array(test_cycles.split(sep=","),dtype=int)
        test_cycles1=False
    except:
        test_cycles1=True
        
    testing=input("want to do the testing of the open voltage? (Y/y)\n") in "Yy"

    
    ## loading data
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    
    ## in this function it is really important to have the raw data
        # a new function has been added to do this without any filtering of the load current and time.
        
    sensorUsed=sensorNum
    includeCycles=True
    
    if includeCycles:
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)
        print(db)
    if test_cycles1:
        test_cycles=np.arange(cb.shape[0])
    
    
    SC,SV,ST,LC,LV,LT=getters.getPCVT_raw(dataDir,sensorNum)
        
    
    
    ##function which calculates the open voltage by taking all points which have a current below the mean ( the BMS is going to constant current so a lot of the data is centered. So all points below this are at the start of the phase.)
    def open_voltage_and_resistance_evolution(I,V,T,boundaries,window,title="test"):
        # modify data with zero voltage elements as this means the load or supply is not turned on yet.
        window_1=np.where(np.logical_and(boundaries<=T,boundaries+window>=T))

        # first correct window
        I=I[window_1]
        V=V[window_1]
        
        # second points of interest
        I=I[V!=0]
        V=V[V!=0]
        
        #data for lin fit
        I_lin=I[0:4]
        V_lin=V[0:4]

        try:
            ohm, V_open_start, r, p, err = scipy.stats.linregress(I_lin.astype(float),V_lin.astype(float))
        except:
            V_open_start=np.nan
            print(I_lin.shape)
        #use V_open_start to calculate the resistance through time
        try:
            resistance=np.abs((V-V_open_start)/I)
            plt.plot(np.arange(len(resistance)),resistance,label="resistance",c="b")
            plt.ylabel(r"resistance $(\Omega)$")
            plt.xlabel("time (s)")
            plt.legend()
        except:
            pass

        plt.title(title+": open voltage = {0:.1f} V".format(V_open_start))
                    
    
    for a,i in enumerate(test_cycles):
        if a%3==0:
            fig=plt.figure(a)
            fig.subplots_adjust(left=0.04, bottom=None, right=None, top=None, wspace=0.5, hspace=0.4)

        plt.subplot(3,2,(a%3)*2+1)
        open_voltage_and_resistance_evolution(SC,SV,ST,cb[i-1,0],windows,title="charge cycle {0}".format(str(i)))
        

        plt.subplot(3,2,(a%3)*2+2)
        open_voltage_and_resistance_evolution(LC,LV,LT,db[i-1,0],windows,title="discharge cycle {0}".format(str(i)))
        
        
    #this is a test function to check which points are taken to determine the open voltage, which is used for all the calculations regarding resistance. (previous testing showed that the open voltage was negative for a lot of cases. So this is the debugging)
    
    def open_voltage_and_resistance_evolution_test(I,V,T,boundaries,window,title="test"):
        # first correct window
        window_1=np.where(np.logical_and(boundaries<=T,boundaries+window>=T))
        I=I[window_1]
        V=V[window_1]
        plt.scatter(I,V,c="r",label="discarded data")
        
        # second points of interest
        I=I[V!=0]
        V=V[V!=0]
        
        #data for lin fit
        I_lin=I[0:4]
        V_lin=V[0:4]
        
        plt.scatter(I_lin,V_lin,c="k",label="data fit")
        
        #calculate mean I and take points below mean to calculate V_open
        
        try:
            ohm, V_open_start, r, p, err = scipy.stats.linregress(I_lin.astype(float),V_lin.astype(float))
            I_test=np.arange(0,6,0.1)
            V_test=V_open_start+I_test*ohm
            plt.plot(I_test,V_test,c="b",label="linear fit")
            plt.xlabel("I")
            plt.ylabel("V")
            plt.xlim([0,max(I)*1.2])
            plt.ylim([0,max(V)*1.2])
            plt.legend()
            plt.title(title+ r"   V={0:.2f}$\pm$ I*{1:.2f}".format(V_open_start,ohm))
        except:
            print(title+ " didnt work")
        
    
    if testing:
        for a,i in enumerate(test_cycles): 
            if a%3==0:
                fig=plt.figure(a+1)
                fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
            plt.subplot(3,2,(a%3)*2+1)
            open_voltage_and_resistance_evolution_test(SC,SV,ST,cb[i,0],windows,title="charge cycle {0}: test lin fit".format(i))

            plt.subplot(3,2,(a%3)*2+2)
            open_voltage_and_resistance_evolution_test(LC,LV,LT,db[i,0],windows,title="discharge cycle {0}: test lin fit".format(i))


    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    
    
def resistance_delta_time():
    
    while True:
        sensorNum = input('Enter stack number (select from {})\n'.format(calc.usedStacks))
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)
            if sensorNum not in calc.usedStacks:
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid stack number.  Please enter a value in {}'.format(calc.usedStacks, sensorNum))

    windows=int(input("what is the window to determine the resistance? (only integers)\n"))
    
    test_cycles=input("what are the cycles you want to check? \n [seperated by a comma (,)] \n")
    try:
        test_cycles=np.array(test_cycles.split(sep=","),dtype=int)
        test_cycles1=False
    except:
        test_cycles1=True
        

    
    ## loading data
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    
    ## in this function it is really important to have the raw data
        # a new function has been added to do this without any filtering of the load current and time.
        
    sensorUsed=sensorNum
    
    
    chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)

    
    
    SC,SV,ST,LC,LV,LT=getters.getPCVT_raw(dataDir,sensorNum)
    
    def delta_resistance(I,V,T,boundaries,window=100,title="test"):
        # first correct window
        window_1=np.where(np.logical_and(boundaries<=T,boundaries+window>=T))
        I=I[window_1]
        V=V[window_1]
        
        # second points of interest
        I=I[V!=0]
        V=V[V!=0]
        
        diff=np.append(np.diff(I),np.nan)
        new_window=np.where(diff<0.3)
        I=I[new_window]
        V=V[new_window]
        
        delta_R=(np.append(np.diff(V),np.nan))/I
        
        plt.plot(delta_R,label="delta resistance", c="b")
        plt.plot(np.cumsum(delta_R[1:]))
        plt.xlabel("time(s)")
        plt.ylabel(r"resistance ($\Omega$)")
        plt.title(title)
        
    
    for a,i in enumerate(test_cycles): 
        
            if a%3==0:
                fig=plt.figure(a+1)
                fig.subplots_adjust(left=0.05, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
            plt.subplot(3,2,(a%3)*2+1)
            delta_resistance(SC,SV,ST,cb[i-1,0],windows,title="charge cycle {0}: resistance".format(str(i)))
            plt.subplot(3,2,(a%3)*2+2)
            delta_resistance(LC,LV,LT,db[i-1,0],windows,title="discharge cycle {0}: resistance".format(str(i)))


    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
