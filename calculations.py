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
    roundtrip = abs(dischargingPower / chargingPower)

    totalCurrentC = np.sum(currentC)    # Adding up all the currents gives a measure of total charge
    totalCurrentD = np.sum(currentD)    # Just like integrating
    # Coulombic effiency = Discharged charge / charged charge
    coulombic = abs(totalCurrentD / totalCurrentC)
    if printing:
        print('Roundtrip efficiency = {}'.format(roundtrip))
        print('Coulombic efficiency = {}'.format(coulombic))

    # Voltage efficiency == roundtrip / coulombic
    VE = roundtrip / coulombic

    # Power Density and power density efficiency
    # Power density == power / memArea

    chargePD = np.mean(powerC / memArea)
    dischargePD = np.mean(powerD / memArea)

    # Energy density
    # energy of discharge / tot volume
    dischargeEnergy = np.zeros(len(powerD))

    for i in range(len(powerD)):  # Integrate
        dischargeEnergy[i] = np.sum(powerD[:i])

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
