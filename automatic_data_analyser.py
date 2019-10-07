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

#from function_file import  *

# ================================================================================================
dataDir = ''    # This global variable will store the path to the folder 



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

#

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
      'tab:pink', 'tab:gray', 'tab:olive', 'tab:olive', 'indigo', 'navy', 'tan', 'black',
      'lightgreen', 'lightcoral', 'cadetblue']
cc = cycler(linestyle=[ '--', '-.',':','-']) * (cycler(color=colors))     

plt.rc('axes', prop_cycle=cc)

# grid on for all plots
plt.rcParams['axes.grid'] = True
#transparancy plots 
alpha1=0.8  
alpha2=1

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
        fig.subplots_adjust(hspace=0.4)
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
            currentChargingPlot.plot(chargeTime, supplyCurrentC[i - 1], label='Cycle {}'.format(i), color=color, alpha=alpha1)
            voltageChargingPlot.plot(chargeTime, supplyVoltageC[i - 1], label='Cycle {}'.format(i), color=color, alpha=alpha1)
            powerChargingPlot.plot(chargeTime, supplyPowerC[i - 1], label='Cycle {}'.format(i), color=color, alpha=alpha1)
        for j in cycles:
            try:
                dischargeTime = np.arange(0, len(loadCurrentD[j - 1]))
            except IndexError:
                print('Cycle {} not found for discharging'.format(j))
                continue
            color = colors[j % len(colors)]
            currentDischargingPlot.plot(dischargeTime, loadCurrentD[j - 1], label='Cycle {}'.format(j), color=color, alpha=alpha1)
            voltageDischargingPlot.plot(dischargeTime, loadVoltageD[j - 1], label='Cycle {}'.format(j), color=color, alpha=alpha1)
            powerDischargingPlot.plot(dischargeTime, loadPowerD[j - 1], label='Cycle {}'.format(j), color=color, alpha=alpha1)
        for plot in [currentChargingPlot, voltageChargingPlot, powerChargingPlot, currentDischargingPlot,
                     voltageDischargingPlot, powerDischargingPlot]:
            plot.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        fig.subplots_adjust(left=0.065, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

        plt.show()
    ccp =fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ccp.x1=ccp.x1/2
    fig.savefig("ccp.png",bbox_inches=ccp.expanded(1.1,1.1))



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
    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break

    # Option to shade the background with charging / discharging regions
    includeCycles = input('Include Cycles in Background? (y/n)') in 'yY'
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
    
    
    fig.suptitle('Water Levels')
    for i, level in enumerate(levels, 1):
        title = 'Level Sensor {}'.format(i)
        if i == len(levels):    # This is the combined level plot
            title = 'Total Volume'
            fig.add_subplot(3, 1, 3)    # Let the total plot take up twice the normal width
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
    plt.legend()


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

    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Conductivity Data')
    for i, cond in enumerate(conductivities, 1):
        fig.add_subplot(3, 1, int(np.around(i/2+0.5,decimals=1)))
        plt.plot(cond,ls=":",alpha=alpha2)
        if i%2==1:
            plt.title('Conducitivity Sensor %s & %s' % (str(i),str(i+1)))
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
    if save:
        fileName = input('Enter Desired File Name\n')
        if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
            fileName += '.xlsx'

    # Calculate efficiencies for each load / supply that is being used
    for sensorNum in calc.usedStacks:  # Don't include a point for 3, since unused
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
            roundtrip, coulombic, VE, chargePD, dischargePD, energyDensity = calc.getEfficiencies(supplyCurrentC[cycle],
                                                                                                  supplyVoltageC[cycle],
                                                                                                  supplyPowerC[cycle],
                                                                                                  loadCurrentD[cycle],
                                                                                                  loadVoltageD[cycle],
                                                                                                  loadPowerD[cycle])
            PDratio = dischargePD / chargePD

            df.loc[cycle] = [str(int(cycle + 1)), round(roundtrip, 3), round(coulombic, 3),
                             round(VE, 3), round(chargePD, 3), round(dischargePD, 3),
                             round(PDratio, 3), round(energyDensity, 3), round(timeC[cycle], 3),
                             round(timeD[cycle], 3)
                             ]    # Add the efficiency info for current cycle to the bottom of the dataframe

        print('--------------------------------------------------------')
        print('Data for Load/Supply {}'.format(sensorNum))
        for col in df.columns:  # Print out all of our data nicely
            print('{}\n{}\n'.format(col, df[col].to_string(index=False)))

        if save:
            while True:
                try:
                    df.to_excel(fileName, sheet_name='Load + Supply {}'.format(sensorNum), index=False)
                    break
                except PermissionError:
                    print('Attempting to Overwrite an Open Excel Workbook.\nPlease Close the Workbook or Change the Name of the Destination File')
                    fileName = input('Input new file name\n')
                    if '.xlsx' not in fileName:     # Forgive our poor user for not including the proper file extension
                        fileName += '.xlsx'

        fig = plt.figure()
        fig.suptitle('Load / Supply {} Efficiencies'.format(sensorNum))
        fig.subplots_adjust(hspace=0.4)

        fig.add_subplot(2, 1, 1)
        plt.title('Roundtrip Efficiency')
        plt.plot(df['Cycle Number'].values, df['Roundtrip Efficiency'].values, 'o')
        plt.xlabel('Cycle')
        plt.ylabel('Efficiency')

        fig.add_subplot(2, 1, 2)
        plt.title('Coulombic Efficiency')
        plt.plot(df['Cycle Number'].values, df['Coulombic Efficiency'].values, 'o')
        plt.xlabel('Cycle')
        plt.ylabel('Efficiency')

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
        plt.legend()
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
    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'

    flows = np.array([getters.getFlowData(dataDir, i) for i in range(1, 3)])
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Flow Rate')
    for i, flow in enumerate(flows, 1):
        title = 'Pump {}'.format(i)
        fig.add_subplot(2, 1, i)
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

    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break

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

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Pressure Levels')
    for i, pressure in enumerate(pressures, 1):
        fig.add_subplot(3, 1, i)
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
    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break

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

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
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
    temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
    concentrations = [calc.getConcentration(conductivities[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))], temps[i][0:int(np.minimum(len(conductivities[i]),len(temps[i])))]) for i in range(len(conductivities))]               # Concentration in mols / L

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

    sensorUsed = 0
    for sensorNum in calc.usedStacks:
        chargeCycles, dischargeCycles = dataReader.getCycles(dataDir, sensorNum)
        if chargeCycles and dischargeCycles:
            sensorUsed = sensorNum
            break
    includeCycles = input('Include Cycles in Background? (y/n)\n') in 'yY'

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Concentrations')
    for i, conc in enumerate(concentrations, 1):
        fig.add_subplot(3, 1,int(np.round(i/2+0.5,decimals=1)))
        if i%2==1: 
            plt.title('Conductivity Sensor %s & %s' % (str(i),str(i+1)))
        plt.plot(conc, ls=":")
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (mol/L)')
        if includeCycles and i%2==0:
            dataReader.colorPlotCycles(dataDir, sensorUsed)
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


# Loads the stored values for membrane area and volume from the CSV
def loadParams():
    df = pd.read_csv(getters.paramPath)
    memArea = df['Membrane Area (cm^2)'][0]
    totV = df['Total Volume (m^3)'][0]
    usedStacks = df['Loads / Supplies Used']
    calc.setParams(memArea, totV, usedStacks, writeCSV=False)   # Set the values in calc's global variables
# ===============================================================================================


# The main menu where the user wil be taken after each step



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



if __name__ == '__main__':
    loadParams()
    importFiles()
#    stack_number=input()
    makeCombinedPlots()