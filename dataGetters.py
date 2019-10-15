import pandas as pd
import calculations as calc
import numpy as np
from scipy.signal import savgol_filter


paramPath = 'Parameters.csv'    # Must be in same working directory


def getAllFileNames():
    condPaths = ['CT{:02d}_CONDUCTIVITY.bcp'.format(i) for i in range(1, 7)]
    tempPaths = ['CT{:02d}_TEMPERATURE.bcp'.format(i) for i in range(1, 7)]
    flowPaths = ['FT{:02d}_FLOW.bcp'.format(i) for i in range(1, 3)]
    presPaths = ['PT{:02d}_PRESSURE.bcp'.format(i) for i in range(1, 4)]
    levelPaths = ['LT{:02d}_LEVEL.bcp'.format(i) for i in range(1, 5)]
    supplyVPaths = ['SP{:02d}_ACTUAL_VOLTAGE.bcp'.format(i) for i in calc.usedStacks]
    supplyIPaths = ['SP{:02d}_ACTUAL_CURRENT.bcp'.format(i) for i in calc.usedStacks]
    supplyPPaths = ['SP{:02d}_ACTUAL_POWER.bcp'.format(i) for i in calc.usedStacks]
    loadVPaths = ['SP{:02d}_ACTUAL_VOLTAGE.bcp'.format(i) for i in calc.usedStacks]
    loadIPaths = ['SP{:02d}_ACTUAL_CURRENT.bcp'.format(i) for i in calc.usedStacks]
    loadPPaths = ['SP{:02d}_ACTUAL_POWER.bcp'.format(i) for i in calc.usedStacks]
    allFiles = condPaths + tempPaths + presPaths + levelPaths + supplyVPaths + supplyIPaths + flowPaths + supplyPPaths + loadVPaths + loadIPaths + loadPPaths
    return allFiles


def pathToArray(path):
    df = pd.read_csv(path, header=None, delimiter='\t')
    return df.values[:, 4]


def getSupplyCurrentData(dataDir, sensorNum):
    chargePath = dataDir + 'SP{:02d}_ACTUAL_CURRENT.bcp'.format(sensorNum)
    chargeData = pathToArray(chargePath)
    return chargeData


def getLoadCurrentData(dataDir, sensorNum):
    dischargePath = dataDir + 'LD{:02d}_ACTUAL_CURRENT.bcp'.format(sensorNum)
    dischargeData = pathToArray(dischargePath)
    return dischargeData


def getSupplyPowerData(dataDir, sensorNum):
    chargePath = dataDir + 'SP{:02d}_ACTUAL_POWER.bcp'.format(sensorNum)
    chargeData = pathToArray(chargePath)
    return chargeData


def getLoadPowerData(dataDir, sensorNum):
    dischargePath = dataDir + 'LD{:02d}_ACTUAL_POWER.bcp'.format(sensorNum)
    dischargeData = pathToArray(dischargePath)
    return dischargeData


def getSupplyVoltageData(dataDir, sensorNum):
    chargePath = dataDir + 'SP{:02d}_ACTUAL_VOLTAGE.bcp'.format(sensorNum)
    chargeData = pathToArray(chargePath)
    return chargeData


def getLoadVoltageData(dataDir, sensorNum):
    dischargePath = dataDir + 'LD{:02d}_ACTUAL_VOLTAGE.bcp'.format(sensorNum)
    dischargeData = pathToArray(dischargePath)
    return dischargeData


# The conductivity sensors are known to give a value of 9.996 when there's a communication error
# Give this getter the option to filter out these data points (or would it be better to replace them?)
def getConductivityData(dataDir, sensorNum, filtering=True):
    condPath = dataDir + 'CT{:02d}_CONDUCTIVITY.bcp'.format(sensorNum)
    ERROR_VALUE = 9.99  # The sensors output this nonsense value when there's an issue
    condData = pathToArray(condPath)

    if filtering:       # Gets rid of error values and values outside of possible range

        errorCondition = abs(condData - ERROR_VALUE) > 0.01
        valueCondition = np.ones(shape=len(condData))
        if sensorNum in [3, 4, 5, 6]:   # Fresh sensors should never read above 50
            valueCondition = np.logical_and(np.logical_and(condData < 60,np.abs(condData-10) > 0.3), np.abs(condData-8) > 0.2)
        else:      # Salt sensors should never read below 30
            valueCondition =np.logical_and(condData > 30 , np.abs(condData-10) > 0.3)

#        # Clean the data as long as either condition isn't satisfied
#        while not np.all(errorCondition) or not np.all(valueCondition):
#
#            condData[:-1] = np.where(errorCondition, condData[:-1], condData[1:])
#            condData[:-1] = np.where(valueCondition, condData[:-1], condData[1:])
#
#            errorCondition = abs(condData[:-1] - ERROR_VALUE) > 0.01
#            valueCondition = [True for i in range(1, len(condData))]
#            if sensorNum in [3, 4, 5, 6]:   # Fresh sensors should never read above 50
#                valueCondition &= (condData[:-1] < 70)
#            else:      # Salt sensors should never read below 30
#                valueCondition &= (condData[:-1] > 30)
            
            
        #trying to do Coens method of masking all values
        condData=np.ma.masked_array(condData,np.logical_or(~errorCondition, ~valueCondition))
        print("step made")
            
#        condData = savgol_filter(condData, 21, 5)
    return condData


def getTemperatureData(dataDir, sensorNum, filtering=True):
    tempPath = dataDir + 'CT{:02d}_TEMPERATURE.bcp'.format(sensorNum)
    temps = pathToArray(tempPath)
    ERROR_VALUE = 9997.0    # The value that gets output fairly regularly that is obviously wrong
    if filtering:
        while np.any(abs(temps - ERROR_VALUE) < 0.01):  # As long as there are any error values in the dataset
            condition = abs(temps[:-1] - ERROR_VALUE) > 0.01    # Replacement condition (Replace when true)
            temps[1:] = np.where(condition, temps[:-1], temps[1:])  # Replace any error values with the neighboring value
        temps = savgol_filter(temps, 1001, 3)
    return temps


def getFlowData(dataDir, sensorNum):
    flowPath = dataDir + 'FT{:02d}_FLOW.bcp'.format(sensorNum)
    return pathToArray(flowPath)


def getPressureData(dataDir, sensorNum, filtering=True):
    pressurePath = dataDir + 'PT{:02d}_PRESSURE.bcp'.format(sensorNum)
    pressureData = pathToArray(pressurePath)
    if filtering:
        pressureData = savgol_filter(pressureData, 1001, 6)
    return pressureData


def getLevelData(dataDir, sensorNum):
    levelPath = dataDir + 'LT{:02d}_LEVEL.bcp'.format(sensorNum)
    return pathToArray(levelPath)


# Imports the standard data used to make a fit from conductivity to concentration
# Data must be in the same format as the example file
def getConcentrationCalcData(calcPath):
    data = pd.read_excel(calcPath).to_numpy()
    cond = data[:, 1] * 10
    temp = data[:, 2]
    conc = data[:, 3]   # Convert mS/cm -> S/m
    return conc, temp, cond
