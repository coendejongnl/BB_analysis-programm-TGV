# ======================================= Imports =================================================
import directoryGetter
import calculations as calc
import dataGetters as getters
import dataReader

import os
import sys

from cycler import cycler #used to cycle through styles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


#LC=None
#SC=None
#

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink', 'tab:purple', 'tab:brown',
       'tab:gray', 'tab:olive',   'navy', 'tan', 'black',
      'lightgreen', 'lightcoral', 'cadetblue']
cc = cycler(linestyle=[ '--', '-.',':','-']) * (cycler(color=colors))     

plt.rc('axes', prop_cycle=cc)


# grid on for all plots
plt.rcParams['axes.grid'] = True
#transparancy plots 
alpha1=0.8  
alpha2=1


def save_to_excel(df,name,dataDir,sensorNum):
    """This function saves a dataframe to excel and stores it in de directory of the data in the folder (data_analysis)"""
    
    
    while True:
        try:
            df.to_excel(str(dataDir)+"data_analysis/"+str(name)+"{}.xlsx".format(sensorNum), sheet_name=str(name)+'_sensor_{}'.format(sensorNum), index=False)
            break
        except PermissionError:
            print('Attempting to Overwrite an Open Excel Workbook.\nPlease Close the Workbook or Change the Name of the Destination File')
            fileName = input('Input new file name\n')
            if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
                fileName += '.xlsx'
                    
    
## this function makes all the arrays in the list the same size, which is a function used a lot !!!
def make_same_len(list_x):
    a=[] #just to solve an error warning in python
    for i in list_x:
        try:
            a=min(a,len(i))
        except:
            a=len(i)
     
    x=[]
    for j in list_x:
        x.append(np.array(j[0:a]))
        
    return(x)
    
    
    
    

def makeCombinedPlots():
    def plotSensor(sensorNum):
        # Load all of the data necessary
        supplyCurrentC, supplyCurrentD = dataReader.getSegmentedSupplyCurrent(dataDir, sensorNum)
        loadCurrentC, loadCurrentD = dataReader.getSegmentedLoadCurrent(dataDir, sensorNum)

        supplyVoltageC, supplyVoltageD = dataReader.getSegmentedSupplyVoltage(dataDir, sensorNum)
        loadVoltageC, loadVoltageD = dataReader.getSegmentedLoadVoltage(dataDir, sensorNum)

        supplyPowerC, supplyPowerD = dataReader.getSegmentedSupplyPower(dataDir, sensorNum)
        loadPowerC, loadPowerD = dataReader.getSegmentedLoadPower(dataDir, sensorNum)

        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)

        totalCycles = np.min([chargeCycles, dischargeCycles])
        if (totalCycles == 0):     # Meaning the load/supply at this number were not in use during experiment
            print('Detected no cycles for stack {}. Please select an active stack'.format(sensorNum))
            makeCombinedPlots()     # Ask user to input a new sensorNum and try again

        # Create plot skeletons to be drawn over with proper data
        fig = plt.figure()
        fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        currentChargingPlot = fig.add_subplot(3, 2, 1)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Supply Current')
        currentDischargingPlot = fig.add_subplot(3, 2, 2)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Load Current')

        voltageChargingPlot = fig.add_subplot(3, 2, 3)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Supply Voltage')
        voltageDischargingPlot = fig.add_subplot(3, 2, 4)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (v)')
        plt.title('Load Voltage')

        powerChargingPlot = fig.add_subplot(3, 2, 5)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.title('Supply Power')
        powerDischargingPlot = fig.add_subplot(3, 2, 6)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.title('Load Power')

        # Ask the user which cycles to plot
        print('Detected {} cycles in all'.format(totalCycles))
        cycleList = input('''Which cycles would you like to plot?'''
                          '''(Enter integers in Range {}-{} '''
                          '''separated by spaces) or Press ENTER\n to plot all\n'''.format(1, totalCycles)
                          )

        if cycleList == '':
            cycles = np.arange(1, totalCycles + 1)
        else:
            cycles = []
            currentNumStr = ''
            for i, char in enumerate(cycleList):
                # import pdb
                # pdb.set_trace()
                if char == ' ':
                    continue
                try:    # Make sure that the input can be represented as an integer
                    int(char)
                except ValueError:
                    continue

                currentNumStr += char  # Append the string integer to the currentNum
                if i == len(cycleList) - 1 or cycleList[i + 1] == ' ':  # If this is the last digit or the input is a space
                    currentNum = int(currentNumStr)
                    if not 1 <= currentNum <= totalCycles:
                        print('{} is out of the range of cycles.  Excluding')
                        continue
                    cycles.append(currentNum)
                    currentNumStr = ''  # Clear the current number for use on next iteration

        # Draw curves in each plot

        # Explicitly define uniform colors so that each plot's cycles have the same color
      
        for i in cycles:   # Plot the charging cycles first
            try:
                chargeTime = np.arange(0, len(supplyCurrentC[i - 1]))
            except IndexError:
                print('Cycle {} not found for charging'.format(i))
                continue
            currentChargingPlot.plot(chargeTime, supplyCurrentC[i - 1], label='Cycle {}'.format(i), alpha=alpha1)
            voltageChargingPlot.plot(chargeTime, supplyVoltageC[i - 1], label='Cycle {}'.format(i), alpha=alpha1)
            powerChargingPlot.plot(chargeTime, supplyPowerC[i - 1], label='Cycle {}'.format(i),  alpha=alpha1)
        for j in cycles:
            try:
                dischargeTime = np.arange(0, len(loadCurrentD[j - 1]))
            except IndexError:
                print('Cycle {} not found for discharging'.format(j))
                continue
            currentDischargingPlot.plot(dischargeTime, loadCurrentD[j - 1], label='Cycle {}'.format(j),  alpha=alpha1)
            voltageDischargingPlot.plot(dischargeTime, loadVoltageD[j - 1], label='Cycle {}'.format(j),  alpha=alpha1)
            powerDischargingPlot.plot(dischargeTime, loadPowerD[j - 1], label='Cycle {}'.format(j),  alpha=alpha1)
        for plot in [currentChargingPlot, voltageChargingPlot, powerChargingPlot, currentDischargingPlot,
                     voltageDischargingPlot, powerDischargingPlot]:
            plot.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
       
        window =fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        window2=window
        window.x1=window.x1/2
        fig.savefig(str(dataDir+"data_analysis/figures/supply.png"),bbox_inches=window.expanded(1.05,1.05))
        window2.x0=window2.x1
        window2.x1=window2.x1*2
        fig.savefig(str(dataDir+"data_analysis/figures/load.png"),bbox_inches=window.expanded(1.05,1.05))


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
    plotSensor(sensorNum)


# Plot the values of each level sensor
def makeLevelPlots():
    
            
    

    # Option to shade the background with charging / discharging regions
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorUsed)

        
        
    levels = [getters.getLevelData(dataDir, i) for i in range(1, 5)]
    for i in range(len(levels)):                    # Since this data is not necessarily the same size, pad with zeros or slice
        if len(levels[i]) > len(levels[0]):
            levels[i] = levels[i][:len(levels[0])]
        elif len(levels[i]) < len(levels[0]):
            levels[i] = np.hstack((levels[i], np.zeros(len(levels[0]) - len(levels[i]))))

    levels = np.array(levels)   # Now that each row is the same length, we can make into a numpy array
    totalLevel = np.sum(levels, axis=0)     # Sum the rows -> add the contributions from each sensor together
    levels = np.vstack((levels, totalLevel))

    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
  
    
    
    fig.suptitle('Water Levels')
    for i, level in enumerate(levels, 1):
        title = 'Level Sensor {}'.format(i)
        if i == len(levels):    # This is the combined level plot
            title = 'Total Volume'
            fig.add_subplot(3, 2, 6)    # Let the total plot take up twice the normal width
        else:
            fig.add_subplot(3, 2, i)
        plt.plot(level,ls="-")
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Volume ($m^3$)')
        if includeCycles:
            dataReader.colorPlotCycles(dataDir, sensorUsed)  # Color the background with cycle status for extra info



    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()  


def makeConductivityPlots():
    # Give the user the option to save the cleaned data using the savgol filter from getters
    filterData = input('Filter Data? (y/n)\n') in 'yY'
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
    
    conductivities = np.array([getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)])

   

        

    if includeCycles:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorUsed)

        

    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.suptitle('Conductivity Data')
    for i, cond in enumerate(conductivities, 1):
        if i==6 or i==3:
            fig.add_subplot(3, 2, 4)
            plt.title('Conducitivity Sensor %s & %s' % (str(3),str(6)))

            
        if i==4 or i==5:
            fig.add_subplot(3, 2, 6)
            plt.title('Conducitivity Sensor %s & %s' % (str(4),str(5)))
        if i==1 or i==2:
            fig.add_subplot(3, 2, 2)
            plt.title('Conducitivity Sensor %s & %s' % (str(1),str(2)))

        plt.plot(cond,ls="-.",alpha=alpha2,label=str(i))
        plt.xlabel('Time (s)')
        plt.ylabel('Conductivity ($\\frac{mS}{cm}$)')
        plt.legend()
        if includeCycles and i%2==0:
            dataReader.colorPlotCycles(dataDir, sensorUsed)  # Color the background
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


# Gather data from a measurement including several cycles and output all relevant information,
# including efficiencies, power/energy densities, etc.
def makeEfficiencies():
    save = input('Save Data to an Excel File? (y/n)\n') in 'yY'
#    if save:
#        fileName = input('Enter Desired File Name\n')
#        if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
#            fileName += '.xlsx'

    # Calculate efficiencies for each load / supply that is being used
    
    for sensorNum in calc.usedStacks:  
        print("checking "+ str(sensorNum) + " for cycles")
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if not chargeCycles or not dischargeCycles:     # If the load/supply was not in use
            continue

        # Load all of the data necessary segmented into charging / discharging
        supplyCurrentC, supplyCurrentD = dataReader.getSegmentedSupplyCurrent(dataDir, sensorNum)
        loadCurrentC, loadCurrentD = dataReader.getSegmentedLoadCurrent(dataDir, sensorNum)

        supplyVoltageC, supplyVoltageD = dataReader.getSegmentedSupplyVoltage(dataDir, sensorNum)
        loadVoltageC, loadVoltageD = dataReader.getSegmentedLoadVoltage(dataDir, sensorNum)

        supplyPowerC, supplyPowerD = dataReader.getSegmentedSupplyPower(dataDir, sensorNum)
        loadPowerC, loadPowerD = dataReader.getSegmentedLoadPower(dataDir, sensorNum)

        dt = 1  # Time interval (seconds) between data points (I think this is a constant always)
        timeC, timeD = dataReader.getSegmentedTime(dataDir, sensorNum, dt)

        df = pd.DataFrame(columns=['Cycle Number', 'Roundtrip Efficiency', 'Coulombic Efficiency',
                                   'Voltage Efficiency', 'Charge Power Density (Wm^-2)', 'Discharge Power Density (Wm^-2)',
                                   'Power Density Ratio', 'Energy Density (kWh / m^3)', 'Charge Time (hours)',
                                   'Discharge Time (hours)'])

        for cycle in range(min([chargeCycles, dischargeCycles])):
            print(cycle)
            roundtrip, coulombic, VE, chargePD, dischargePD, energyDensity = calc.getEfficiencies(supplyCurrentC[cycle],
                                                                                                  supplyVoltageC[cycle],
                                                                                                  supplyPowerC[cycle],
                                                                                                  loadCurrentD[cycle],
                                                                                                  loadVoltageD[cycle],
                                                                                                  loadPowerD[cycle])
            chargePD=np.where(chargePD!=0,chargePD,np.nan)
            PDratio = dischargePD / chargePD

            df.loc[cycle] = [str(int(cycle + 1)), np.round(roundtrip, 3), np.round(coulombic, 3),
                             np.round(VE, 3), np.round(chargePD, 3), np.round(dischargePD, 3),
                             np.round(PDratio, 3), np.round(energyDensity, 3), np.round(timeC[cycle], 3),
                             np.round(timeD[cycle], 3)
                             ]    # Add the efficiency info for current cycle to the bottom of the dataframe

        print('--------------------------------------------------------')
        print('Data for Load/Supply {}'.format(sensorNum))
        for col in df.columns:  # Print out all of our data nicely
            print('{}\n{}\n'.format(col, df[col].to_string(index=False)))

        if save:
            save_to_excel(df,"Load + Supply",dataDir,sensorNum)


        fig = plt.figure()
        fig.suptitle('Load / Supply {} Efficiencies'.format(sensorNum))
        fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

        fig1=fig.add_subplot(3, 2, 1)
        
        
        
        plt.xlabel('number of cycles')
        plt.ylabel('Efficiency')
        
        
        plt.plot(df['Cycle Number'].values, df['Voltage Efficiency'].values, 'P',label="VE")
        plt.plot(df['Cycle Number'].values, df['Roundtrip Efficiency'].values, 'o',label="RTE")

        plt.plot(df['Cycle Number'].values, df['Coulombic Efficiency'].values, 'v',label="CE")
        fig1.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        
        fig2=fig.add_subplot(3, 2, 3)
        plt.xlabel('number of cycles')
        plt.xlabel('number of cycles')



        fig2.plot(df['Cycle Number'].values, df['Discharge Power Density (Wm^-2)'].values, 'P')
        fig2.set_ylabel(r'Discharge Power Density $\frac{W}{m^2}$')
        for tl in fig2.get_yticklabels():
            tl.set_color('tab:blue')
        
        fig22 = fig2.twinx()
        fig22.plot(df['Cycle Number'].values, df['Energy Density (kWh / m^3)'].values, 'o',color="r")
#        fig22.legend(bbox_to_anchor=(1.04,0.5), loc="center right", borderaxespad=0)
        fig22.set_ylabel(r'Energy Density $\frac{kWh}{ m^3}$')
        fig22.grid(linestyle='dotted')
        for tl in fig22.get_yticklabels():
            tl.set_color('r')
                                  
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()


# Generate IV Curves using the max values of each charge / discharge cycle
def makeIVCurves():
    # Take only peak voltage and current from each cycle
    def makePlot(dataDir, sensorNum):
        # Load all of the data necessary
        supplyCurrentC, supplyCurrentD = dataReader.getSegmentedSupplyCurrent(dataDir, sensorNum)
        loadCurrentC, loadCurrentD = dataReader.getSegmentedLoadCurrent(dataDir, sensorNum)

        supplyVoltageC, supplyVoltageD = dataReader.getSegmentedSupplyVoltage(dataDir, sensorNum)
        loadVoltageC, loadVoltageD = dataReader.getSegmentedLoadVoltage(dataDir, sensorNum)

        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)

        totalCycles = np.min([chargeCycles, dischargeCycles])
        if totalCycles == 0:
            print('No charge/discharge cycles found for load/supply {}'.format(sensorNum))
            return

        fig = plt.figure()
        fig.add_subplot(111)
        for cycle in range(totalCycles):
            plt.plot(np.max(loadVoltageD[cycle]), np.max(loadCurrentD[cycle]), 'o', label='Cycle {}'.format(cycle + 1))
        plt.legend( )
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

    while True:
        sensorNum = input('Enter Load/Supply Number\n')
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)
            if not 1 <= sensorNum <= 5 or sensorNum == 3:
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid load/supply number.  Please enter a value in between 1-5'.format(sensorNum))
    makePlot(dataDir, sensorNum)





# Make plots with the output from each flow sensor
def makeFlowPlots():
    

        
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
        sensorNum =sensorUsed
    
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        
        
        
        
        
    flows = np.array([getters.getFlowData(dataDir, i) for i in range(1, 3)])
    for i in range(len(flows)):
        flows[i]= savgol_filter(flows[i], 1001, 3)

    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.suptitle('Flow Rate')
    titles=["Salt stream","Fresh stream", ]
    for i, flow in enumerate(flows, 1):
#        title = 'Pump {}'.format(i)
        title=titles[i-1]
        fig.add_subplot(3, 2, int((i-1)*2+1))
        plt.plot(flow)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Flow Rate ($\\frac{L}{min}$)')
        if includeCycles:
            dataReader.colorPlotCycles(dataDir, sensorUsed)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def makePressurePlots():
    filterData = input('Filter Data? (y/n)\n') in 'yY'



    pressures = [getters.getPressureData(dataDir, i, filtering=filterData) for i in [1, 2, 3]]    # Already smoothed data
    # Give the user the option to save the cleaned data using the savgol filter from getters
    if filterData:
        output = input('Output Cleaned Data to Excel? (y/n)\n')
        if output == 'y' or output == 'Y':
            fileName = input('Enter a name for the file:\n')
            if '.xlsx' not in fileName: 
                fileName += '.xlsx'
            data = {'Time (s)': pd.read_csv(os.path.join(dataDir, 'CT01_CONDUCTIVITY.bcp'), delimiter='\t').to_numpy()[:, 0],
                    'Pressure Sensor 1 (bar)': pressures[0], 'Pressure Sensor 2 (mBar)': pressures[1], 'Pressure Sensor 3 (mBar)': pressures[2]}
            formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same length
            df = pd.DataFrame(formatData)
            df.to_excel(fileName, index=False)

    includeCycles = input('Include Cycles in Background? (y/n)\n')
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorUsed)
    
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.suptitle('Pressure Levels')
    for i, pressure in enumerate(pressures, 1):
        fig.add_subplot(3, 2, int(i*2))
        plt.plot(pressure)
        plt.title('Pressure Sensor {}'.format(i))
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (Bar)')
        if includeCycles:
            dataReader.colorPlotCycles(dataDir, sensorUsed)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


# Plot data from each temperature sensor
def makeTempPlots():
    filterData = input('Filter Data (y/n)\n') in 'yY'


    temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
    if filterData:
        output = input('Output Cleaned Data to Excel? (y/n)\n')
        if output == 'y' or output == 'Y':
            fileName = input('Enter a name for the file:\n')
            if '.xlsx' not in fileName:
                fileName += '.xlsx'
            data = {}
            data['Time (s)'] = pd.read_csv(os.path.join(dataDir, 'CT01_CONDUCTIVITY.bcp'), delimiter='\t').to_numpy()[:, 0]
            for i in range(1, 7):
                data['Temperature Sensor {} (C)'.format(i)] = temps[i - 1]
            formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same length
            df = pd.DataFrame(formatData)
            df.to_excel(fileName, index=False)

    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)"))
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorUsed)


    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.suptitle('Temperature')
    for i, temp in enumerate(temps, 1):
        fig.add_subplot(3, 2, i)
        plt.plot(temp)
        plt.title('Temperature Sensor {}'.format(i))
        plt.xlabel('Time (s)')
        plt.ylabel(r'Temperature ($^\circ$C)')
        if includeCycles:
            dataReader.colorPlotCycles(dataDir, sensorUsed)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


# Plots the load and supply currents from the given sensor and includes shading to indicate when charging or discharging
# Mainly used to double check that the cycle analysis was performed correctly and cycles line up properly
def makeCurrentPlot():
    def plotCurrents(sensorNum):
        supplyCurrent, loadCurrent = getters.getSupplyCurrentData(dataDir, sensorNum), getters.getLoadCurrentData(dataDir, sensorNum)
        chargeStarts, dischargeStarts = dataReader.determineCycles(supplyCurrent, loadCurrent, printing=False)
        if not np.any(chargeStarts) or not np.any(dischargeStarts):
            print('No cycles detected for load/supply {}'.format(sensorNum))
            return makeCurrentPlot()   # Just try again if the user input was bad
        dataReader.plotCycles(supplyCurrent, loadCurrent, chargeStarts, dischargeStarts)

    while True:
        sensorNum = input('Enter Load/Supply Number\n')
        if sensorNum == 'q':
            return
        try:
            sensorNum = int(sensorNum)  # ValueError will be thrown if the input given is not an integer
            if sensorNum not in calc.usedStacks:   # Throw an AttributeError if the integer given doesn't correspond to a sensor
                raise AttributeError
            break
        except ValueError:
            print('{} is not in a valid format.  Please enter an integer'.format(sensorNum))
        except AttributeError:
            print('{} is not a valid load/supply number.  Please enter a value in 1, 2, 4, 5'.format(sensorNum))
    plotCurrents(sensorNum)


# Plots the concentrations using a polynomial fit from the raw conductivity and temperature data
# Details on how the fit is constructed is given in calculations.makeConcentrationModel()
def makeConcentrationPlot():
    filterData = input('Filter Data (y/n)\n') in 'yY'
    conductivities = [getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)]    
    

    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*20) for i in range(len(conductivities))]    

    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed = int(input("Which stack is used? (give a number)\n"))
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorUsed)
        
        
        
        
        
        
    fig = plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.suptitle('Concentrations')
    for i, conc in enumerate(concentrations, 1):
        if i==6 or i==3:
            fig.add_subplot(3, 2,4)
            plt.title('Conductivity Sensor %s & %s' % (str(3),str(6)))

        if i==4 or i==5:
            fig.add_subplot(3, 2,6)
            plt.title('Conductivity Sensor %s & %s' % (str(4),str(5)))
        if i==1 or i==2:
            fig.add_subplot(3, 2,2)
            plt.title('Conductivity Sensor %s & %s' % (str(1),str(2)))

        

        plt.plot(conc, ls=":")
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (mol/L)')
        if includeCycles and i%2==0:
            dataReader.colorPlotCycles(dataDir, sensorUsed)
            
            
    for i, cond in enumerate(conductivities, 1):
        if i==6 or i==3:
            fig.add_subplot(3, 2, 3)
            plt.title('Conductivity Sensor %s & %s' % (str(3),str(6)))

            
        if i==4 or i==5:
            fig.add_subplot(3, 2, 5)
            plt.title('Conductivity Sensor %s & %s' % (str(4),str(5)))
        if i==1 or i==2:
            fig.add_subplot(3, 2, 1)
            plt.title('Conductivity Sensor %s & %s' % (str(1),str(2)))

        plt.plot(cond,ls="-.",alpha=alpha2)
        plt.xlabel('Time (s)')
        plt.ylabel('Conductivity ($\\frac{mS}{cm}$)')
        if includeCycles and i%2==0:
            dataReader.colorPlotCycles(dataDir, sensorUsed)  # Color the background
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


# Gives the user the ability to view and edit the values of membrane area and volume
def viewParams():
    # Display the existing vlalues
    print('Membrane Area = {} cm^2\nTotal Volume = {} m^3\nLoads/Supplies used = {}'.format(calc.memArea * 1e4, calc.totV, calc.usedStacks))
    # ALlow modifying
    change = input('Modify Existing Values? (y/n)\n') in 'yY'
    if change:
        while True:
            try:
                newArea = float(input('Enter a new value for the area (in cm^2):\n'))
                break
            except ValueError:  # Thrown if the input cannot be cast to float
                print('{} is an invalid format. Please enter a decimal number')
        while True:
            try:
                newVolume = float(input('Enter a new value for the volume (in m^3):\n'))
                break
            except ValueError:  # Thrown if the input cannot be cast to float
                print('{} is an invalid format. Please enter a decimal number')
        while True:
            try:
                usedStackstring = input('Enter the numbers for the loads and supplies to be included in analysis, separated by commas (e.g. \'1, 2, 3\')')
                usedLoadListStr = usedStackstring.split(',')
                usedLoadListInts = []
                for str in usedLoadListStr:
                    loadNum = int(str)
                    if loadNum not in range(1, 6):
                        raise IndexError
                    usedLoadListInts.append(int(str))
                break
            except ValueError:
                print('{} is an invalid format.  Please enter integeters in', [...])
            except IndexError:
                print('{} is an invalid format.  Make sure the inputs are in the range 1-5')

        calc.setParams(newArea, newVolume, usedLoadListInts, writeCSV=True)

def power_plot_cycles():
    
    
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
            
            
    supplyPowerC, supplyPowerD = dataReader.getSegmentedSupplyPower(dataDir, sensorNum)
    loadPowerC, loadPowerD = dataReader.getSegmentedLoadPower(dataDir, sensorNum)
    chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)

    totalCycles = np.min([chargeCycles, dischargeCycles])
    
    
    
    if (totalCycles == 0):     
        print('Detected no cycles for stack {}. Please select an active stack'.format(sensorNum))
        exit()
    Nplots=np.ceil(totalCycles/5) #number of plots
    
    
    fig=plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    
    for i in range(int(Nplots)):
        subfigure=fig.add_subplot(3,2,i+1)
        
        if i+1==int(np.max(Nplots)):
            cycleList=range(i*5,totalCycles)
        else:
            cycleList=range(i*5,(i+1)*5)
        print(cycleList)
 
        power_cycle=np.array([])     
        
        for j in cycleList:
            power_cycle=np.append(power_cycle,supplyPowerC[j])
            power_cycle=np.append(power_cycle,-loadPowerD[j])
            
        subfigure.plot(power_cycle,label="cycle {} - {}".format(str(np.min(cycleList)),str(np.max(cycleList))))
        plt.fill_between(np.arange(len(power_cycle)),power_cycle,"b")
        plt.title("cycle {} - {}".format(str(np.min(cycleList)+1),str(np.max(cycleList)+1)))
#        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.ylabel("power (W)")
        plt.xlabel("time(s)")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    

def Gibbs(volume,concentration,T=293,v=2):
    #https://science.sciencemag.org/content/sci/161/3844/884.full.pdf
    #^^ activity coefficient
    m=concentration #about the same as the concentration is low
    gamma=calc.concentration_to_gamma(concentration)
    R=8.314 #j/(k*mol)
    kg_water=volume*0.997*1000
    G=kg_water*v*m*R*T*np.log(gamma*m)
    G=G/(60*60*1000)
    return(G)
    
def free_energy_Gibbs():
    
#    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    includeCycles=True
    window=int(input("give the window to average. (only integers as inputs) \n"))

    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)
    
    
    def add_different_length(a,b):
        C=np.add(a[0:min(len(a),len(b))],b[0:min(len(a),len(b))])
        return(C)
    
    
    filterData=True
    ## waterlevels only 1 - 3 are important
    levels = [getters.getLevelData(dataDir, i) for i in range(1, 5)]
    ## conductivity
    
    conductivities = np.array([getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)])
#    temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
    Temp=np.array([getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)])
    concentrations = [calc.getConcentration(conductivities[i],Temp[i]) for i in range(len(conductivities))] 
    
    ## concentration
    
    G1=Gibbs(levels[0][0:min(len(levels[0]),len(concentrations[0]))],concentrations[0][0:min(len(levels[0]),len(concentrations[0]))])     
    print("1")
    G2=Gibbs(levels[0][0:min(len(levels[0]),len(concentrations[1]))],concentrations[1][0:min(len(levels[0]),len(concentrations[1]))])
    print("2")

    G3=Gibbs(levels[1][0:min(len(levels[1]),len(concentrations[4]))],concentrations[4][0:min(len(levels[1]),len(concentrations[4]))]) 
    print("3")
    G4=Gibbs(levels[1][0:min(len(levels[1]),len(concentrations[3]))],concentrations[3][0:min(len(levels[1]),len(concentrations[3]))])  
    print("4")
    G5=Gibbs(levels[2][0:min(len(levels[2]),len(concentrations[2]))],concentrations[2][0:min(len(levels[2]),len(concentrations[2]))]) 
    print("5")
    G6=Gibbs(levels[2][0:min(len(levels[2]),len(concentrations[5]))],concentrations[5][0:min(len(levels[2]),len(concentrations[5]))])  
    print("done")       
    
    Gibbs_total=add_different_length(G1,G2)
    Gibbs_total=add_different_length(Gibbs_total,G3)
    Gibbs_total=add_different_length(Gibbs_total,G4)
    Gibbs_total=add_different_length(Gibbs_total,G5)
    Gibbs_total=add_different_length(Gibbs_total,G6)
    
    
    #### loading data used to obtain Energy from load and supply
    sensorNum=sensorUsed
    LP=getters.getLoadPowerData(dataDir, sensorNum)
    SP=getters.getSupplyPowerData(dataDir, sensorNum)
    
       
    #### now we use the boundaries cb and db:
    
    
    def delta_gibbs(Gibbs_total,bounds,window=100):
        
        delta_g=np.mean(Gibbs_total[np.arange(bounds[1]-window,bounds[1]+window)])-np.mean(Gibbs_total[np.arange(bounds[0]-window,bounds[0]+window)])
        delta_g=np.abs(delta_g)
        return(delta_g)
        
    arraydatacharge=np.array([])
    arraydatadischarge=np.array([])
    
    CPsum=np.array([])
    DPsum=np.array([])


    for i in range(int(cb.shape[0])):
        try:
            CPsum=np.append(CPsum,np.sum(SP[range(cb[i,0],cb[i,1])]))
            DPsum=np.append(DPsum,np.sum(LP[range(db[i,0],db[i,1])]))
        except:
            print("error in power integral")
            
        try:
            arraydatacharge=np.append(arraydatacharge,delta_gibbs(Gibbs_total,cb[i,:],window))
            arraydatadischarge=np.append(arraydatadischarge,delta_gibbs(Gibbs_total,db[i,:],window))
            

        except:
            print("append didn't work")
    
    
    CPsum=CPsum/(60*60*1000)  
    DPsum=DPsum/(60*60*1000)

    
    ## some illustrating which average he takes for the gibbs on the boundaries
    
    def gibbs_average_bound(total_gibbs,bounds_1D,window):
        mean_energy=np.array([])
        new_bound=np.array([])
        
        for i in bounds_1D:
            try:
                i=int(i)
                mean_energy=np.append(mean_energy,np.mean(Gibbs_total[np.arange(i-window,i+window)]))
                new_bound=np.append(new_bound,i)
            except:
                print("one of the arrays already stops before the end of a transition")
        return(new_bound,mean_energy)
            
    bound=np.vstack((cb,db))
    flat_bound=bound.flatten()
    
    x,y=gibbs_average_bound(Gibbs_total,flat_bound,window)
    
    
    
    if len(DPsum)>len(CPsum):
        CPsum=np.append(CPsum,np.nan)
    if len(DPsum)<len(CPsum):
        DPsum=np.append(DPsum,np.nan)    
    
    ratio_c=arraydatacharge[0:min(len(arraydatacharge),len(CPsum))]/CPsum[0:min(len(arraydatacharge),len(CPsum))]
    ratio_d=(DPsum[0:min(len(arraydatadischarge),len(DPsum))]/arraydatadischarge[0:min(len(arraydatadischarge),len(DPsum))])
    ratio_cd=ratio_c[0:min(len(ratio_c),len(ratio_d))]*ratio_d[0:min(len(ratio_c),len(ratio_d))]
    ### storing the data in an dictionary and storing it in a DataFrame:

    ## test dict
    d={"charging gibbs":arraydatacharge,
          "charging energy SP":CPsum,
          "ratio charging":ratio_c,
          "discharging gibbs":arraydatadischarge,
          "discharging energy LD": DPsum,
          "ratio discharging":ratio_d,
          "RTE multiplication":np.abs(ratio_cd),
          "ratio gibbs c/d":np.abs(arraydatacharge[0:min([len(arraydatacharge),len(arraydatadischarge)])]/arraydatadischarge[0:min([len(arraydatacharge),len(arraydatadischarge)])]),
          "RTE discharge/charge":np.abs(DPsum/CPsum)
              }  
            
    df= pd.DataFrame.from_dict(d, orient='index')

    df=df.transpose()
    print(df)
    
    ## save df to excel
    name="gibbs free energy"
    save_to_excel(df,name,dataDir,sensorNum)
    
    #### plotting all the data

    fig=plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(3,2,1)
    plt.plot(G1,label="G1")
    plt.plot(G2,label="G2")
    plt.plot(G3,label="G3")
    plt.plot(G4,label="G4")
    plt.plot(G5,label="G5")
    plt.plot(G6,label="G6")
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)
    plt.ylabel("KWH")
    plt.xlabel("time(s)")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    plt.subplot(3,2,3)
    plt.plot(Gibbs_total)
    plt.scatter(x,y,c="r",marker="x",s=int(10**2))
    plt.ylabel("KWH")
    plt.xlabel("time(s)")
    
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)

    plt.subplot(3,2,5)
    size=int(8**2)
    plt.scatter(np.arange(len(ratio_c))+1.1,ratio_c,marker="x",c="g",s=size,label=r"$\mu_{charging}$")
    plt.scatter(np.arange(len(ratio_d))-0.1+1,ratio_d,marker="o",c="r",s=size,label=r"$\mu_{discharging}$")
    plt.scatter(np.arange(len(ratio_cd))+1,ratio_cd,marker="p",c="b",s=size,label=r"$\mu_{RTE}$")
    plt.ylabel("efficiency[-]")
    plt.xlabel("Cycle number [-]")
    plt.ylim(0,1.2)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xticks(ticks=np.arange(len(ratio_d)))
    
    
    plt.subplot(3,2,6)
    calc.double_integral(ratio_c,ratio_d)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    
def total_PCV_plot():
    
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
    
    includeCycles=input("include background charging/discharging? y/Y\n") in "yY"


    ## load data
            
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    LP=getters.getLoadPowerData(dataDir, sensorNum)
    SP=getters.getSupplyPowerData(dataDir, sensorNum)
    

    

    ## plot everything including background
    
    
    fig=plt.figure()
    plt.subplot(3,2,1)

    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum)    
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.fill_between(np.arange(len(LV)),-LV,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SV)),SV,label="supply",color="red", ls="-")
    plt.legend() 
    
    plt.subplot(3,2,3)
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum) 
    plt.fill_between(np.arange(len(LC)),-LC,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SC)),SC,label="supply",color="red", ls="-")
    plt.xlabel("time")
    plt.ylabel("current")
    
    
    plt.subplot(3,2,5)
#    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum) 
    
    plt.fill_between(np.arange(len(LP)),-LP,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SP)),SP,label="supply",color="red", ls="-")
    
    plt.xlabel("time")
    plt.ylabel("power")
    
    ## test for open voltage plot
#    plt.subplot(3,2,2)
#    plt.scatter(-1*np.ones(shape=len(SV))[SC!=0],"gx")
#    plt.scatter(np.ones(shape=len(LV))[LC!=0],"rx")
#    plt.xlabel("time")
#    plt.ylabel("voltage with no current")
    plt.show()  
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    
    
def theoretical_resistance_and_open_voltage():
    
    ## state which stack and the parameters for the analysis
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
    
    ## length of window
    windows=int(input("what is the window to determine the resistance? (only integers)\n"))
    #which brand for the theoretical resistance
    while True:
        a=str(input("(F)ujifilm of (E)voque?\n") )
        if a in "FfEe":
            break
    number_of_cells,surface_area,L,resistance_m2_A,resistance_m2_C,width_flow_canal,perm_selectivity=calc.brand_specifications(a)

    filterData = input('Filter Data (y/n)\n') in 'yY'
    
    test=input("want to check the resistance plots? (Yy)\n ")in "Yy"
    if test:
        try:
            test_cycles=input("what are the cycles you want to check? \n [seperated by a comma (,)] \n")
            test_cycles=np.array(test_cycles.split(sep=","),dtype=int)
            test_cycles1=False

        except:
            test_cycles1=True
    
    ## data to load in
    flows = np.array([getters.getFlowData(dataDir, i) for i in range(1, 3)])

    conductivities = [getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)]    
    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*20) for i in range(len(conductivities))]  
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    ## theoretical resistance
    R_c,R_membranes= calc.theoretical_resistance(flows[0],flows[1],concentrations[0],concentrations[1],LC,a)

    
    ## calculations for open voltage using the nernst equation
    V_open_5=calc.open_voltage_function(concentrations[1],concentrations[5],perm_selectivity)
    V_open_3=calc.open_voltage_function(concentrations[1],concentrations[3],perm_selectivity)
    
    ## checking which tank is providing water to the stacks (dilute 1 or 2)
    level3 = getters.getLevelData(dataDir, 3) 

    tank3_diff=np.diff(level3)
    [V_open_3,V_open_5,tank3_diff]=make_same_len([V_open_3,V_open_5,tank3_diff])
    
    ## tank3_diff<0 means the level is dropping -> providing water to the stacks
    Nernst_voltage=np.where(tank3_diff<=0,V_open_3,V_open_5)

    ## using correction from the measurements of redstack. 0.79 percentage of the theoretical open voltage
    Nernst_voltage=Nernst_voltage*0.78
    
    ########### using a window of N length after starting the load or supply  ################
    
    chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorNum)
    if test_cycles1:
        test_cycles=np.arange(len(cb))
        
    #calculating and plotting if testing
    if True:
        SC,SV,ST,LC,LV,LT=getters.getPCVT_raw(dataDir,sensorNum)

        R_ohmic_c=[]
        R_ohmic_d=[]
        R_non_ohmic_c=[]
        R_non_ohmic_d=[]
        V_theoretical_c=[]
        V_theoretical_d=[]
        for a,i in enumerate(test_cycles):        
            if a%3==0:
                fig=plt.figure(a)
                fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
            plt.subplot(3,2,(a%3)*2+1)
            R_ohmic,R_non_ohmic,V_theoretical=calc.concentration_polarization_resistance(SC,SV,ST,cb[a,0],windows,Nernst_voltage,charging=True,title="charging cycle {0}".format(a+1))
            R_ohmic_c.append(R_ohmic)
            R_non_ohmic_c.append(R_non_ohmic)
            V_theoretical_c.append(V_theoretical)
            
            plt.subplot(3,2,(a%3)*2+2)
            R_ohmic,R_non_ohmic,V_theoretical=calc.concentration_polarization_resistance(LC,LV,LT,db[a,0],windows,Nernst_voltage,charging=False,title="discharging cycle {0}".format(a+1))
            R_ohmic_d.append(R_ohmic)
            R_non_ohmic_d.append(R_non_ohmic)
            V_theoretical_d.append(V_theoretical)
            
            
    ## and datafram dict
    d={"R ohmic c":R_ohmic_c,
       "R non ohmic c":R_non_ohmic_c,
       "V open c":V_theoretical_c,
       "R ohmic d":R_ohmic_d,
       "R non ohmic d":R_non_ohmic_d,
       "V open d":V_theoretical_d
              }  
    
    resistance_df={"R_c":R_c,
                   "R_membranes":R_membranes
                   }
            
    df= pd.DataFrame.from_dict(d, orient='index')
    df2=pd.DataFrame(resistance_df)
        
#    df=pd.DataFrame(data)
    df=df.transpose()
    df2=df2.transpose()
    
    print(df)
    print(df2)
    save_to_excel(df,"resistance",dataDir,sensorNum)
    save_to_excel(df,"resistance_theoretical",dataDir,sensorNum)
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    
        
        
def conservation_of_mass():
    #test function
    #
    while True:
        a=str(input("(F)ujifilm of (E)voque?\n") )
        if a in "FfEe":
            break
    number_of_cells,surface_area,L,resistance_m2_A,resistance_m2_C,width_flow_canal,perm_selectivity=calc.brand_specifications(a)
    
    filterData=True
    ## waterlevels only 1 - 3 are important
    levels = [getters.getLevelData(dataDir, i)*1000 for i in range(1, 5)]
    ## conductivity
    conductivities = np.array([getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)])
    
    Temp=np.array([getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)])
    concentrations = [calc.getConcentration(conductivities[i],Temp[i]) for i in range(len(conductivities))]     
    def same_length(x):
        a=[]
        for i in x:
            a.append(len(i))
        b=[]
        for i in x:
            b.append(i[0:min(a)])
        
        return(b)
        
    [levels[1],levels[2],levels[3],concentrations[0],concentrations[1],concentrations[2],concentrations[3],concentrations[4],concentrations[5]]=same_length([levels[1],levels[2],levels[3],concentrations[0],concentrations[1],concentrations[2],concentrations[3],concentrations[4],concentrations[5]])
    mass_s=levels[1]*(concentrations[0]+concentrations[1])
    mass_d1=levels[2]*(concentrations[2]+concentrations[5])
    mass_d2=levels[3]*(concentrations[3]+concentrations[4])
    mass_total=(mass_d1+mass_d2+mass_s)*58.44/1000
    
    
    ## theoretical open voltage
    
    ## calculations for open voltage using the nernst equation
    V_open_5=calc.open_voltage_function(concentrations[1],concentrations[5],perm_selectivity)
    V_open_3=calc.open_voltage_function(concentrations[1],concentrations[3],perm_selectivity)
    
    ## checking which tank is providing water to the stacks (dilute 1 or 2)
    level3 = getters.getLevelData(dataDir, 3) 

    tank3_diff=np.diff(level3)
    [V_open_3,V_open_5,tank3_diff]=make_same_len([V_open_3,V_open_5,tank3_diff])
    
    ## tank3_diff<0 means the level is dropping -> providing water to the stacks
    Nernst_voltage=np.where(tank3_diff<=0,V_open_3,V_open_5)
    
    ## correction of the theoretical open voltage versus the measurements of redstack
    Nernst_voltage=0.78*Nernst_voltage
    
    ##plotting
    fig=plt.plot()
    ax1=plt.subplot(3,2,1)
#    plt.plot(mass_s,label="salt")
#    plt.plot(mass_d1,label="fresh 1")
#    plt.plot(mass_d2,label="fresh 2")
    plt.plot(mass_total,label="total mass NaCl")
    plt.ylabel("kg")
    plt.xlabel("time(s)")
    plt.legend()
    
    ax2=plt.subplot(3,2,3)
    plt.plot(V_open_3,label="V 3")
    plt.plot(V_open_5,label="V 5")
    plt.plot(-Nernst_voltage, label="Nernst")
  
    
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


    plt.show()
    
def monte_carlo_best_stack_long_title():
    
    N=int(input("give the order of samples you want to run. (1ex) give x \n"))
    N=int(10**N)
    #loading the data from the file which is made by hand
    df=pd.read_excel("efficiencies.xlsx")
    
    columns=df.shape[1]
    
    #creating round trip efficiencies
    RTE=[]
    labels1=[]
    print("prepping data")
    for i in range(int(columns/2)):
#        print(i)
        labels1.append(df.columns[int(i*2)][0:2])
#        print(df.columns[int(i*2)])
        
        a1,a2=np.meshgrid(df.iloc[:,2*i+1],df.iloc[:,2*i])
        efficiencies=a1*a2
        efficiencies=efficiencies.flatten()
        efficiencies=efficiencies[~np.isnan(efficiencies)]
        RTE.append(efficiencies)
    
    #using the monte carlo simulation
    final_results_table=calc.simple_monte_carlo(RTE,N)
    final_results_table.transpose()
    #creating a dataframe
    dict1=dict()
    for n,i in enumerate(labels1):
        dict1[i]=final_results_table[:,n]
        
    df1= pd.DataFrame.from_dict(dict1, orient='index')
    df1=df1.transpose()
    print(df1)
    
    df1.to_excel("monte_carlo_simulation_results.xlsx")    

# Loads the stored values for membrane area and volume from the CSV
def loadParams():
    df = pd.read_csv(getters.paramPath)
    memArea = df['Membrane Area (cm^2)'][0]
    totV = df['Total Volume (m^3)'][0]
    usedStacks = df['Loads / Supplies Used']
    calc.setParams(memArea, totV, usedStacks, writeCSV=False)   # Set the values in calc's global variables
# ===============================================================================================


# The main menu where the user wil be taken after each step
def menu():
    optionDisplay()
    # This is the dictionary containing all possible actions to take from the main menu
    figures = {
        '1': makeCombinedPlots,
        '2': makeLevelPlots,
        '3': makeConductivityPlots,
        '4': makeConcentrationPlot,
        '5': makeFlowPlots,
        '6': makePressurePlots,
        '7': makeTempPlots,
        '8': makeEfficiencies,
        '9': makeIVCurves,
        '10': makeCurrentPlot,
        "11": power_plot_cycles,
        "12":free_energy_Gibbs,
        "13":total_PCV_plot,
        "14":theoretical_resistance_and_open_voltage,
        "15":conservation_of_mass,
        "16":monte_carlo_best_stack_long_title,
        'd': importFiles,
        'p': viewParams,
        'q': sys.exit
    }

    while True:
        num = input('Select Action\n')
        if not num or num not in figures.keys():
            print('Inproper value {}'.format(num))
            continue
        break
    figures.get(num)()  # Call the appropriate plotting function
    menu()              # After we leave the function, repeat


# Prints out all of the options that the user can take from the main menu
def optionDisplay():
    print('===================================')
    print('1 -> Current, Voltage, and Power')
    print('2 -> Water Levels')
    print('3 -> Conductivities')
    print('4 -> Concentrations')
    print('5 -> Flow Rates')
    print('6 -> Pressure Levels')
    print('7 -> Temperature')
    print('8 -> Show Efficiencies')
    print('9 -> Make IV-Curves')
    print('10 -> View Supply and Load Currents for Cycle Detection')
    print('11 -> View power cycles in sets of 5')
    print("12 -> free energy efficiencies")
    print("13 -> total_PCV_plot")
    print("14 -> Resistance start cycles")
    print("15 -> conservation of mass")
    print("16 -> Monte carlo simulation")
    print('-----------------------------------')
    print('d -> Import New Data')
    print('p -> View Current Parameters')
    print('q -> Quit')
    print('===================================')


def importFiles():
    global dataDir
    print('Select the directory containing raw data:')
    while True:
        path = directoryGetter.getDir()
        if path == '':  # Means the window was manually closed
            print('Closing app')
            sys.exit(0)
            break
        try:
            files = os.listdir(path)
            dataDir = path + '/'                        # Overwrite the global variable dataDir
            for file in getters.getAllFileNames():
                if file not in files:
                    raise NotADirectoryError
            break   # If no exceptions were thrown then we're good to break out
        except FileNotFoundError:
            print('Path does not exist.  Try again')
        except NotADirectoryError:
            print('Path given does not contain proper files.  Missing \'{}\''.format(file))
    return dataDir