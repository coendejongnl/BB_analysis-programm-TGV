import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import pandas as pd
from scipy.interpolate import interp1d


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
    if len(cond_data)<len(temp_data):
        temp_data=temp_data[0:len(cond_data)]
    else:
        cond_data=cond_data[0:len(temp_data)]

    
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
        """simple array formatter which makes all arrays in list a the same length"""
        min_len=None #define as None to raise an error first loop
        
        for i in a:
            try:
                min_len=np.min([len(i),min_len])
                
            except:
                min_len=len(i)
                
        new_list=[]
        for i in a:
            new_list.append(np.array(i[0:min_len]))

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
        R_c=np.where(J!=0,alpha*R*T/(F*J)*np.log(delta_ar/delta_as)*1024,np.nan)
    
    return(R_c,R_membranes)
    
    
def open_voltage_function(C_salt,C_fresh,permselectivity):
    if len(C_salt)>len(C_fresh):
        C_salt=C_salt[0:len(C_fresh)]
    else:
        C_fresh=C_fresh[0:len(C_salt)]
    
    R=8.314
    T=293.15
    F=96485.3329
    gamma=np.divide(concentration_to_gamma(C_salt),concentration_to_gamma(C_fresh))
    
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
        plt.legend()
    except:
        print("error has occured in:"+title)
    return(R_ohmic,R_non_ohmic,V_theoretical)


def concentration_to_gamma(concentration):
    """This function takes the concentration as an input and uses a interpolation function to create the correct gamma as an output"""
    molality=np.array([0,0.01,0.02,0.05,0.1,0.2, 0.3,0.5,0.7,1,5])
    gamma=np.array([1,0.904,0.875,0.824,0.781,0.734,0.709,0.68,0.664,0.65,0.65])
    f=interp1d(molality,gamma)
    gamma_new=f(concentration)
    return(gamma_new)



    

def simple_monte_carlo(a,size_sample,labels1):
    
    """This function takes from a few random distribution the samples and checks by sheer numbers which one is the likeliest to have the maximum value using a monte carlo simulation.
    
    """
    
    #random samples
    print("random sampling")
    y=np.empty(shape=(len(a),size_sample))
    for i,x in enumerate(a):
        b=np.random.choice(x,size=size_sample)
        y[i,:]=b
    
    #check which ones wins
    print("check the order of winners")
    final_results=np.zeros(shape=(len(a),size_sample))
    for i,j in enumerate(range(len(a)),1):
        amax=np.amax(y,axis=0,keepdims=False)
        amax=np.tile(amax,(len(a),1))
        final_results=np.where(amax==y,i,final_results)
        y=np.where(final_results!=0,0,y)
    #counting
    
    print("table")
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
    
def double_integral(a,b,title="test"):
    x,y=np.meshgrid(a,b)
    z=x*y
    z=z.flatten()
    plt.figure(1)
    plt.cla()
    n,d,e=plt.hist(z,bins=10,weights=np.full(len(z),1/len(z)))
    plt.xlim(0,1)
    plt.xlabel("efficiency [-]")
    plt.ylabel("probability [-]")
    plt.title(r"$N dis {0:.2f}\pm {1:.2f}$".format(np.mean(z),np.std(z)))
    