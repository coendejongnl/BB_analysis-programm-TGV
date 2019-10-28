# ======================================= Imports =================================================
import directoryGetter
import calculations as calc
import dataGetters as getters
import dataReader
import pathlib
from datetime import datetime


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
    
    
    
        while True:
            try:
                df.to_excel(str(dataDir)+"data_analysis/"+str(name)+"{}.xlsx".format(sensorNum), sheet_name=str(name)+'_sensor_{}'.format(sensorNum), index=False)
                break
            except PermissionError:
                print('Attempting to Overwrite an Open Excel Workbook.\nPlease Close the Workbook or Change the Name of the Destination File')
                fileName = input('Input new file name\n')
                if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
                    fileName += '.xlsx'
 


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
            color = colors[i % len(colors)]
            currentChargingPlot.plot(chargeTime, supplyCurrentC[i - 1], label='Cycle {}'.format(i), alpha=alpha1)
            voltageChargingPlot.plot(chargeTime, supplyVoltageC[i - 1], label='Cycle {}'.format(i), alpha=alpha1)
            powerChargingPlot.plot(chargeTime, supplyPowerC[i - 1], label='Cycle {}'.format(i),  alpha=alpha1)
        for j in cycles:
            try:
                dischargeTime = np.arange(0, len(loadCurrentD[j - 1]))
            except IndexError:
                print('Cycle {} not found for discharging'.format(j))
                continue
            color = colors[j % len(colors)]
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

    if filterData:
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

        plt.plot(cond,ls="-.",alpha=alpha2)
        plt.xlabel('Time (s)')
        plt.ylabel('Conductivity ($\\frac{mS}{cm}$)')
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
    conductivities = [getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)]     # Conductivity in mS / cm
#    temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
#    concentrations = [calc.getConcentration(conductivities[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))], temps[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))]) for i in range(len(conductivities))]               # Concentration in mols / L
    concentrations = [calc.getConcentration(conductivities[i], np.ones(shape=conductivities[i].shape)*293.15) for i in range(len(conductivities))]    
    if filterData:
        output = input('Output Cleaned Concentration Data to Excel? (y/n)\n')
        if output == 'y' or output == 'Y':
            fileName = input('Enter a name for the file:\n')
            if '.xlsx' not in fileName:
                fileName += '.xlsx'
            data = {}
            data['Time (s)'] = pd.read_csv(os.path.join(dataDir, 'CT01_CONDUCTIVITY.bcp'), delimiter='\t').to_numpy()[:, 0]
            for i in range(1, 7):
                data['Sensor {} (M)'.format(i)] = concentrations[i - 1]
            formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same length
            df = pd.DataFrame(formatData)
            df.to_excel(fileName, index=False)

   

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
            plt.title('Conducitivity Sensor %s & %s' % (str(3),str(6)))

            
        if i==4 or i==5:
            fig.add_subplot(3, 2, 5)
            plt.title('Conducitivity Sensor %s & %s' % (str(4),str(5)))
        if i==1 or i==2:
            fig.add_subplot(3, 2, 1)
            plt.title('Conducitivity Sensor %s & %s' % (str(1),str(2)))

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
    m=concentration #about the same
    R=8.314 #j/(k*mol)
    kg_water=volume*0.997
    G=kg_water*v*m*R*T*np.log(m)
    G=G/(60*60*1000)
    return(G)
    
def free_energy_Gibbs():
    
#    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'
    includeCycles=True
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
    
    G1=Gibbs(levels[0][0:min(len(levels[0]),len(conductivities[0]))],conductivities[0][0:min(len(levels[0]),len(conductivities[0]))])     
    print("1")
    G2=Gibbs(levels[0][0:min(len(levels[0]),len(conductivities[1]))],conductivities[1][0:min(len(levels[0]),len(conductivities[1]))])
    print("2")

    G3=Gibbs(levels[1][0:min(len(levels[1]),len(conductivities[4]))],conductivities[4][0:min(len(levels[1]),len(conductivities[4]))]) 
    print("3")
    G4=Gibbs(levels[1][0:min(len(levels[1]),len(conductivities[3]))],conductivities[3][0:min(len(levels[1]),len(conductivities[3]))])  
    print("4")
    G5=Gibbs(levels[2][0:min(len(levels[2]),len(conductivities[2]))],conductivities[2][0:min(len(levels[2]),len(conductivities[2]))]) 
    print("5")
    G6=Gibbs(levels[2][0:min(len(levels[2]),len(conductivities[5]))],conductivities[5][0:min(len(levels[2]),len(conductivities[5]))])  
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
    
    def delta_gibbs(Gibbs_total,bounds,window=5):
        
        delta_g=np.mean(Gibbs_total[np.arange(bounds[1]-window,bounds[1])])-np.mean(Gibbs_total[np.arange(bounds[0]-window,bounds[0])])
        return(delta_g)
        
    arraydatacharge=np.array([])
    arraydatadischarge=np.array([])
    
    CPsum=np.array([])
    DPsum=np.array([])


    for i in range(int(cb.shape[0])):
        try:
            CPsum=np.append(CPsum,np.sum(SP[range(cb[i,0],cb[i,1])]))
            DPsum=np.append(DPsum,np.sum(LP[range(db[i,0],db[i,1])]))
            
            arraydatacharge=np.append(arraydatacharge,delta_gibbs(Gibbs_total,cb[i,:],window=5))
            arraydatadischarge=np.append(arraydatadischarge,delta_gibbs(Gibbs_total,db[i,:],window=5))
            

        except:
            print("one of summations didnt work")
    
    
    CPsum=CPsum/(60*60*1000)  
    DPsum=DPsum/(60*60*1000)
    
    
    
    
    
    
    ### storing the data in an dictionary and storing it in a DataFrame:
    
#    data={"charging gibbs":arraydatacharge,
#          "charging energy SP":CPsum,
##          "ratio charging":arraydatacharge/CPsum,
#          "discharging gibbs":arraydatadischarge,
#          "disharging energy LD": DPsum,
##          "ratio discharging":DPsum/arraydatadischarge        
#              }
#        
#    df=pd.DataFrame(data)
#        
#    print(df)
    
    
    #### plotting all the data

    fig=plt.figure()
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.subplot(2,1,1)
    plt.plot(G1)
    plt.plot(G2)
    plt.plot(G3)
    plt.plot(G4)
    plt.plot(G5)
    plt.plot(G6)
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)
    plt.ylabel("KWH")
    plt.xlabel("time(s)")
    
    plt.subplot(2,1,2)
    plt.plot(Gibbs_total)
    plt.ylabel("KWH")
    plt.xlabel("time(s)")
    
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)

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
#            
#            
#    SVC, SVD = dataReader.getSegmentedSupplyVoltage(dataDir, sensorNum)
#    LVC, LVD = dataReader.getSegmentedLoadVoltage(dataDir, sensorNum)
#    SIC, SID = dataReader.getSegmentedSupplyCurrent(dataDir, sensorNum)
#    LIC, LID = dataReader.getSegmentedLoadCurrent(dataDir, sensorNum)
#    chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
#
#    totalCycles = np.min([chargeCycles, dischargeCycles])
#    
#    
#    
#    if (totalCycles == 0):     
#        print('Detected no cycles for stack {}. Please select an active stack'.format(sensorNum))
#        exit()
#        
#    cyclesV=np.array([]) 
#    cyclesI=np.array([])     
#    
#        
#    for j in range(int(totalCycles)): 
#        
#        cyclesV=np.append(cyclesV,SVC[j])
#        cyclesV=np.append(cyclesV,-LVD[j])
#        
#        cyclesI=np.append(cyclesI,SIC[j])
#        cyclesI=np.append(cyclesI,LID[j])
#        
#    Resistance=
            
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
#    global LC,SC
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    LP=getters.getLoadPowerData(dataDir, sensorNum)
    SP=getters.getSupplyPowerData(dataDir, sensorNum)
    

    

    
    
    fig=plt.figure()
    plt.subplot(3,2,1)

    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum)    
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    plt.fill_between(np.arange(len(LV)),-LV,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SV)),SV,label="supply",color="red", ls="-")
    plt.legend() 
    
    plt.subplot(3,2,3)
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum) 
    plt.fill_between(np.arange(len(LC)),-LC,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SC)),SC,label="supply",color="red", ls="-")
    
    plt.subplot(3,2,5)
    fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorNum) 
    
    plt.fill_between(np.arange(len(LP)),-LP,label= "load",color="blue",ls="-")
    plt.fill_between(np.arange(len(SP)),SP,label="supply",color="red", ls="-")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def resistance():
    
    #################
    #This function is created to determine the total resistance
    #################
    
    
    ## give inputs of all the important data
    
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
    
    
    
    ## loading data
    
    LV=getters.getLoadVoltageData(dataDir, sensorNum)
    SV=getters.getSupplyVoltageData(dataDir, sensorNum)
    
    LC=getters.getLoadCurrentData(dataDir, sensorNum)
    SC=getters.getSupplyCurrentData(dataDir, sensorNum)
    
    ## determening boundaries
    
    sensorUsed=sensorNum
    includeCycles=True
    if includeCycles:
        chargeCycles, dischargeCycles,cb,db = dataReader.getCycles1(dataDir, sensorUsed)
    
    ## obtaining V_open by taking voltage before cycle begins. taking np.mean(window=100sec)
    
    def V_open(voltage,bound,window=100):
        V=voltage[np.arange(bound-window,bound)]
        V=V[V!=0]
        V_mean=np.mean(V)
        return(V_mean)
    
    V_open_c=np.array([])
    V_open_d=np.array([])
    
    for i in range(int(cb.shape[0])):
        V_open_c=np.append(V_open_c,V_open(SV,cb[i,0],window=100))
        V_open_d=np.append(V_open_d,V_open(LV,db[i,0],window=100))
        
    ## determine voltage load/supply 100 seconds after charging/discharging cycle has started. with a mean function
    def x_system(x1,y1,bound,window=100):
        x=x1[np.arange(bound,bound+window)]
        y=y1[np.arange(bound,bound+window)]
    
#        x=x[y!=0]
        x=np.ma.masked_array(x,y==0)
        y=np.ma.masked_array(y,y==0)
        
        
        return(x)


    V_system_c=np.array([])
    V_system_d=np.array([])
    I_system_c=np.array([])
    I_system_d=np.array([])
        
    for i in range(int(cb.shape[0])):
        if i==0:
            V_system_c=x_system(SV,SC,cb[i,0],window=100)
            V_system_d=x_system(LV,LC,db[i,0],window=100)
            
            I_system_c=x_system(SC,SC,cb[i,0],window=100)
            I_system_d=x_system(LC,LC,db[i,0],window=100)
        else:
            try:
                I_system_c=np.vstack((I_system_c,x_system(SC,SC,cb[i,0],window=100)))
                I_system_d=np.vstack((I_system_d,x_system(LC,LC,db[i,0],window=100)) )                 
                V_system_c=np.vstack((V_system_c,x_system(SV,SC,cb[i,0],window=100)))
                V_system_d=np.vstack((V_system_d,x_system(LV,LC,db[i,0],window=100)) ) 
                

            except:
                
                print(i)
    ## calculating the resistance using emperical data
    print(V_system_c.shape)
    print(V_system_d.shape)
    print(I_system_c.shape)
    print(I_system_d.shape)
    print(V_open_c.shape)
    print(V_open_d.shape)
    
    
    
    Resistance_c=np.mean(np.abs((V_system_c.T-V_open_c)/I_system_c.T),axis=0)
    Resistance_d=np.mean(np.abs((V_system_d.T-V_open_d)/I_system_d.T),axis=0)
    Voltage_c=np.mean(V_system_c,axis=1)
    Voltage_d=np.mean(V_system_d,axis=1)
    Current_c=np.mean(I_system_c,axis=1)
    Current_d=np.mean(I_system_d,axis=1)
    open_voltage_c=V_open_c
    open_voltage_d=V_open_d

    
    ## create an DataFrame
    
    data={"resistance C":Resistance_c,
          "resistance D":Resistance_d  ,
          "voltage C":Voltage_c,
          "voltage d":Voltage_d,
          "open V C": open_voltage_c,
          "open V d": open_voltage_d, 
          "current c":Current_c,
          "current d":Current_d
              }
    
    df=pd.DataFrame(data)
    print(df)
    
    ## save to excel
    save_to_excel(df,"resistance",dataDir,sensorNum)
    

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