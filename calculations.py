import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ================================Constants for the Blue Battery Setup===============================
memArea = None  # I believe this represents the total volume of the system in m^3
totV = None        # Rough total volume of water in the tanks combined
usedStacks = None
# ==================================================================================================
import dataGetters as getters


# Sets the values of the global parameters from the csv data
def setParams(area, vol, loads, writeCSV=True):
    global memArea, totV, usedStacks
    memArea = area * 1e-4   # Convert from cm^2 -> m^2 for global variable
    totV = vol
    usedStacks = list(loads)

    if writeCSV:
        # Because dataframes require inputs of they same size, fill area and vol columns with None
        areaList = [area]
        volList = [vol]
        for i in range(len(usedStacks) - 1):
            areaList.append(None)
            volList.append(None)
        updatedDf = pd.DataFrame(data={'Membrane Area (cm^2)': areaList,
                                       'Total Volume (m^3)': volList,
                                       'Loads / Supplies Used': usedStacks
                                       })
        updatedDf.to_csv(getters.paramPath, index=None)


# Do all of the energy/power efficiency and density calculations given data
def getEfficiencies(currentC, voltageC, powerC, currentD, voltageD, powerD, printing=False):

    chargingPower = np.sum(powerC)
    dischargingPower = np.sum(powerD)
    # Roundtrip efficiency = energy discharged / energy charged
    chargingPower=np.where(chargingPower!=0,chargingPower,np.nan)
    roundtrip = abs(dischargingPower / chargingPower)

    totalCurrentC = np.sum(currentC)    # Adding up all the currents gives a measure of total charge
    totalCurrentD = np.sum(currentD)    # Just like integrating
    # Coulombic effiency = Discharged charge / charged charge
    totalCurrentC=np.where(totalCurrentC!=0,totalCurrentC,np.nan)

    coulombic = abs(totalCurrentD / totalCurrentC)
    if printing:
        print('Roundtrip efficiency = {}'.format(roundtrip))
        print('Coulombic efficiency = {}'.format(coulombic))

    # Voltage efficiency == roundtrip / coulombic
    coulombic1=np.where(coulombic!=0,coulombic,np.nan)
    VE =roundtrip / coulombic1

    # Power Density and power density efficiency
    # Power density == power / memArea

    chargePD = np.mean(powerC / memArea)
    dischargePD = np.mean(powerD / memArea)

    # Energy density
    # energy of discharge / tot volume
    dischargeEnergy = np.zeros(len(powerD))

#    for i in range(len(powerD)):  # Integrate
#        dischargeEnergy[i] = np.sum(powerD[:i])
#        
    #### this should be many times quicker
    dischargeEnergy=np.cumsum(powerD)

    dischargeEnergykWh = dischargeEnergy / (1000 * 3600)

    energyDensity = np.mean(dischargeEnergykWh / totV)

    return roundtrip, coulombic, VE, chargePD, dischargePD, energyDensity


# Test a linear regression fitting conducitivites to concentrations and plot
def plotCondToConcFit(condData, concData):
    slope, intercept, r_value, p_value, std_err = linregress(condData, concData)
    plt.plot(condData, concData, label='Raw Data')
    plt.plot(condData, slope * condData + intercept, label='Linear Fit')
    plt.legend()
    plt.show()


# Convert conducitivity data to concentration data using measurements made in a standard file
def makeConcentrationModel(calibrationPath):    # TODO: THe current values provided don't make any sense, but this implementation should work for anything
    conc, temp, cond = getters.getConcentrationCalcData(calibrationPath)
    X = np.array([cond, temp]).transpose()  # Needs to be a column vector for the model to recognize
    Y = conc

    poly_features = PolynomialFeatures(degree=10)
    poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X, Y)
    return model


# Converts arrays of conductivity and temperature data to concentrations using a polynomial fit from a calibration standard
def getConcentration(cond_data, temp_data):
    calibrationPath = 'ConcentrationCalibration.xlsx'
    model = makeConcentrationModel(calibrationPath)
    inputs = np.array([cond_data, temp_data]).transpose()   # Needs transposed

    conc_predict = model.predict(inputs)
    conc_predict = np.clip(conc_predict, 0, np.inf)  # Don't allow negative numbers
    return conc_predict

def brand_specifications(a="f"):
    if a in "Ff":
        number_of_cells=512
        surface_area=0.22**2
        L=.22
        resistance_m2_A=1.7/100**2
        resistance_m2_C=2/100**2
        width_flow_canal=165e-6
        perm_selectivity=0.97
        
    if a in "Ee":
        number_of_cells=512
        surface_area=0.22**2
        L=.22
        ##9.7 ohm/cm**2
        resistance_m2_A=9.7/100**2
        resistance_m2_C=0/100**2
        width_flow_canal=165e-6
        perm_selectivity=0.965
    return(number_of_cells,surface_area,L,resistance_m2_A,resistance_m2_C,width_flow_canal,perm_selectivity)
    
    
def theoretical_resistance(flowrate_s_raw,flowrate_r_raw,concentration_s,concentration_f,current,a="f"):
    
    #############fuji film resistance############
    number_of_cells,surface_area,L,resistance_m2_A,resistance_m2_C,width_flow_canal,perm_selectivity=brand_specifications(a)
    
    ##same length function
    def same_length(a):
        for i in a:
#            print(len(i))
            try:
                min_len=np.min([len(i),min_len])
                
            except:
                min_len=len(i)
                
        new_list=[]
        for i in a:
            new_list.append(np.array(i[0:min_len]))
#            print(len(i[0:min_len]))

        return(new_list)
    ################R membranes############
    resistance_A=number_of_cells*(resistance_m2_A/surface_area)
    resistance_C=number_of_cells*(resistance_m2_C/surface_area)
    
    R_membranes=resistance_A+resistance_C
    
    
    ####################R_c###################
    alpha=0.97
    R=8.314
    T=293.15
    F=96485
    J=current/surface_area ## current /density
    flowrate_r=flowrate_r_raw/(1000*60)
    flowrate_s=flowrate_s_raw/(1000*60)
    
    q_r=flowrate_r/width_flow_canal/512#m**2/s
    q_s=flowrate_s/width_flow_canal/512
    
    C_r=concentration_f*1000#mol/m**3
    C_s=concentration_s*1000
    
    (J,C_r,C_s,q_r,q_s)=same_length((J,C_r,C_s,q_r,q_s))
    
    delta_ar=1+np.divide(J*L,F*q_r* C_r)
    delta_as=1-np.divide(J*L,F*q_s* C_s)
    
    with np.errstate(all='ignore'):
        R_c=np.where(J!=0,alpha*R*T/(F*J)*np.log(delta_ar/delta_as)*512,np.nan)
    
    return(R_c,R_membranes)
    
    
def open_voltage_function(C_salt,C_fresh,permselectivity):
    if len(C_salt)>len(C_fresh):
        C_salt=C_salt[0:len(C_fresh)]
    else:
        C_fresh=C_fresh[0:len(C_salt)]
    
    R=8.314
    T=293.15
    F=96485.3329
    gamma=1
    
    x=C_salt/C_fresh
    
    E_0=R*T/F*np.log(gamma*x)
    E_open=E_0*2*512*permselectivity
    return(E_open)
    
def concentration_polarization_resistance(I,V,time,boundary,window,V_theoretical,charging=True,title="test"):
    window_of_interrest=np.where(np.logical_and(time>boundary,time<boundary+window))
    I=I[window_of_interrest]
    V=V[window_of_interrest]
    plt.scatter(I,V,label="discarded data",c="r",zorder=-10)
    
    ## V_theoretical mean in window
    V_theoretical=np.mean(V_theoretical[boundary:boundary+window])
    #discard values of voltage zero
    I=I[V!=0]
    V=V[V!=0]
    
    #discard values of I lower than 2 as this will greatly increase the resistance if it is below or above the Vopen
    V=V[I>0.5]
    I=I[I>0.5]
    try:
        plt.scatter(I,V,label="data used",c="g",zorder=0)
        plt.title(title)
        plt.xlabel("I")
        plt.ylabel("V")
        
        x=np.arange(0,7,1)
        x=np.expand_dims(x,1)
        if charging:
            resistance=(V-V_theoretical)/I
            
    
        else:
            resistance=-(V-V_theoretical)/I
    
    
        
        try:
            
            R_ohmic=min(resistance)
            R_non_ohmic=max(resistance)-min(resistance)
            
            R_non_ohmic_index=np.argmax(resistance)
            R_ohmic_index=np.argmin(resistance)
        except:
            R_ohmic=np.nan
            R_non_ohmic=np.nan
            
        plt.scatter(0,V_theoretical,c="b", label="open voltage")
        x=np.full(V.shape,0)
        x=np.column_stack((x,I)).T
        y=np.full(V.shape,V_theoretical)
        y=np.column_stack((y,V)).T
        
        plt.plot(x,y,zorder=-5,alpha=.2,c="r")
        plt.scatter(I[np.array([R_non_ohmic_index,R_ohmic_index])],V[np.array([R_non_ohmic_index,R_ohmic_index])],c="c",zorder=10,label="import data")
#        plt.legend()
    except:
        print("error has occured in:"+title)
    return(R_ohmic,R_non_ohmic,V_theoretical)

    