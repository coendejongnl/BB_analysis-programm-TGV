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

#    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)    
#    window1 =fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#    
##    window1=window
#    window1.x1=window1.x1/2
#    window1.y0=window1.y1*2/3
#    fig.savefig(str(dataDir+"data_analysis/waterlevel1.png"),bbox_inches=window1.expanded(1.05,1.05))
#    
#    window2=window
#    window2.x0=window2.x1/2
#    window2.y1=window2.y1/3
#    fig.savefig(str(dataDir+"figures/waterlevel2.png"),bbox_inches=window2.expanded(1.05,1.05))
#    
#    window3=window
#    window3.x0=window.x0
#    window3.x1=window.x1/2
#    window3.y0=window.y1/3
#    window3.y1=window3.y1*2/3
#    fig.savefig(str(dataDir+"figures/waterlevel3.png"),bbox_inches=window3.expanded(1.05,1.05))
#    
#    window4=window
#    window4.x0=window.x1/2
#    window4.y0=window.y1/3
#    window4.y1=window4.y0*2
#    fig.savefig(str(dataDir+"figures/waterlevel4.png"),bbox_inches=window4.expanded(1.05,1.05))
#    
#    window5=window
#    window5.x0=window.x1/2
#    window5.y0=window.y1*2/3
#    window5.y1=window.y1
#    fig.savefig(str(dataDir+"figures/waterlevel5.png"),bbox_inches=window5.expanded(1.05,1.05))

# Displays six plots of the conductivities, one for each sensor
def makeConductivityPlots():
    # Give the user the option to save the cleaned data using the savgol filter from getters
    filterData = input('Filter Data? (y/n)\n') in 'yY'
    conductivities = np.array([getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)])

    if False:#filterData:
        output = input('Output Cleaned Data to Excel? (y/n)\n')
        if output == 'y' or output == 'Y':
            fileName = input('Enter a name for the file:\n')
            if '.xlsx' not in fileName:
                fileName += '.xlsx'

            data = {}
            data['Time (s)'] = pd.read_csv(os.path.join(dataDir, 'CT01_CONDUCTIVITY.bcp'), delimiter='\t').to_numpy()[:, 0]
            for i in range(1, 7):
                data['Sensor {}'.format(i)] = conductivities[i - 1]
            formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same length
            df = pd.DataFrame(formatData)
            df.to_excel(fileName, index=False)


#    if chargeCycles and dischargeCycles:
#        sensorUsed = sensorNum
        
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    if includeCycles:
        sensorUsed =int(input("Which stack is used? (give a number)\n"))
    
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
#            while True:
#                try:
#                    df.to_excel(str(dataDir)+"data_analysis/Load + Supply {}.xlsx".format(sensorNum), sheet_name='Load + Supply {}'.format(sensorNum), index=False)
#                    break
#                except PermissionError:
#                    print('Attempting to Overwrite an Open Excel Workbook.\nPlease Close the Workbook or Change the Name of the Destination File')
#                    fileName = input('Input new file name\n')
#                    if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
#                        fileName += '.xlsx'

        fig = plt.figure()
        fig.suptitle('Load / Supply {} Efficiencies'.format(sensorNum))
        fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

        fig1=fig.add_subplot(3, 2, 1)
#        plt.title('Efficiencies')
        
        
        
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
    
    ## test 
#    plt.figure()
#    plt.plot(conductivities[4])
#    plt.plot(conductivities[3])
    # Conductivity in mS / cm
#    temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
#    concentrations = [calc.getConcentration(conductivities[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))], temps[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))]) for i in range(len(conductivities))]               # Concentration in mols / L
    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*293.15) for i in range(len(conductivities))]    
    if filterData:
        pass
#        output = input('Output Cleaned Concentration Data to Excel? (y/n)\n')
#        if output == 'y' or output == 'Y':
#            fileName = input('Enter a name for the file:\n')
#            if '.xlsx' not in fileName:
#                fileName += '.xlsx'
#            data = {}
#            data['Time (s)'] = pd.read_csv(os.path.join(dataDir, 'CT01_CONDUCTIVITY.bcp'), delimiter='\t').to_numpy()[:, 0]
#            for i in range(1, 7):
#                data['Sensor {} (M)'.format(i)] = concentrations[i - 1]
#            formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same length
#            df = pd.DataFrame(formatData)
#            df.to_excel(fileName, index=False)

   

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
    m=concentration #about the same
    gamma=0.68
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
    
    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*293.15) for i in range(len(conductivities))] 
    
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
    ratio_d=(DPsum[0:min(len(arraydatadischarge),len(DPsum))]/arraydatadischarge[0:min(len(arraydatadischarge),len(DPsum))])**-1
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
          "ratio gibbs c/d":np.abs(arraydatacharge/arraydatadischarge),
          "RTE discharge/charge":np.abs(DPsum/CPsum)
              }  
            
    df= pd.DataFrame.from_dict(d, orient='index')
#    data={"charging gibbs":arraydatacharge,
#          "charging energy SP":CPsum,
##          "ratio charging":arraydatacharge/CPsum,
#          "discharging gibbs":arraydatadischarge,
#          "disharging energy LD": DPsum,
##          "ratio discharging":DPsum/arraydatadischarge        
#              }
        
#    df=pd.DataFrame(data)
    df=df.transpose()
    print(df)
    
    ## save df to excel
    name="gibbs free energy"
    save_to_excel(df,name,dataDir,sensorNum)
    
    #### plotting all the data

    fig=plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(3,1,1)
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

    plt.subplot(3,1,2)
    plt.plot(Gibbs_total)
    plt.scatter(x,y,c="r",marker="x",s=int(10**2))
    plt.ylabel("KWH")
    plt.xlabel("time(s)")
    
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)

    plt.subplot(3,1,3)

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
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
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
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


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
        
    sensorUsed=sensorNum
    includeCycles=True
    
    if includeCycles:
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)

    if test_cycles1:
        test_cycles=np.arange(cb.shape[0])
        
        
    ##function which calculates the open voltage by taking all points which have a current below the mean ( the BMS is going to constant current so a lot of the data is centered. So all points below this are at the start of the phase.)
    def open_voltage_and_resistance_evolution(I,V,boundaries,window,title="test"):
        # modify data with zero voltage elements as this means the load or supply is not turned on yet.
        
        # first correct window
        I=I[boundaries:boundaries+window]
        V=V[boundaries:boundaries+window]
        
        # second points of interest
        I=I[V!=0]
        V=V[V!=0]
        
        #data for lin fit
        I_lin=I[0:4]
        V_lin=V[0:4]

        try:
            ohm, V_open_start, r, p, err = scipy.stats.linregress(I_lin,V_lin)
        except:
            V_open_start=np.nan
            print(I_lin.shape)
        #use V_open_start to calculate the resistance through time
        resistance=np.abs((V-V_open_start)/I)
        plt.plot(np.arange(len(resistance)),resistance,label="resistance",c="b")
        plt.ylabel(r"resistance $(\Omega)$")
        plt.xlabel("time (s)")
        plt.legend()

        plt.title(title+": open voltage = {0:.1f} V".format(V_open_start))
                    
    
    for a,i in enumerate(test_cycles):
        if a%3==0:
            fig=plt.figure(a)
            fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

        plt.subplot(3,2,(a%3)*2+1)
        open_voltage_and_resistance_evolution(SC,SV,cb[i,0],windows,title="charge cycle {0}".format(str(i+1)))
        

        plt.subplot(3,2,(a%3)*2+2)
        open_voltage_and_resistance_evolution(LC,LV,db[i,0],windows,title="discharge cycle {0}".format(str(i+1)))
        
        
    #this is a test function to check which points are taken to determine the open voltage, which is used for all the calculations regarding resistance. (previous testing showed that the open voltage was negative for a lot of cases. So this is the debugging)
    
    def open_voltage_and_resistance_evolution_test(I,V,boundaries,window,title="test"):
        # first correct window
        I=I[boundaries:boundaries+window]
        V=V[boundaries:boundaries+window]
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
            ohm, V_open_start, r, p, err = scipy.stats.linregress(I_lin,V_lin)
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
            open_voltage_and_resistance_evolution_test(SC,SV,cb[i,0],windows,title="charge cycle {0}: test lin fit".format(i))

            plt.subplot(3,2,(a%3)*2+2)
            open_voltage_and_resistance_evolution_test(LC,LV,db[i,0],windows,title="discharge cycle {0}: test lin fit".format(i))


    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    

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
        "14":resistance,
        "15":resistance_test_linear_plot,
        "16":resistance_evolution,
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
    print("15 -> Resistance test linear plot and bootstrap")
    print("16 -> Resistance time evolution")
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