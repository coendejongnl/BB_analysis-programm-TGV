import numpy as np
import matplotlib.pyplot as plt
# Queue module was renamed to queue in Python 3.  In case using Python 2 include this catch
try:
    import queue
except ImportError or ModuleNotFoundError:
    import Queue as queue
import dataGetters as getters

# Conductivity sensors give error value of 9.96.
EPSILON = 0.01  # Checks for values near this error value to catch errors


# Function that takes current data from a supply and a load and returns the indices at Which
# the system transitions between charging states.
# Determines charging state by looking for where either device outputs 0's for extended periods
def determineCycles(chargeData, dischargeData, printing=False):

    chargeStarts = np.empty(0, dtype=np.uint16)
    chargeEnds = np.empty(0, dtype=np.uint16)
    numPoints = 100     # Consider numPoints measurements when deciding if charging or discharging
    recentQueue = queue.Queue()

    # Don't begin counting cycles until something actually starts happening
    startIdx = 0
    try:
        while (chargeData[startIdx] < EPSILON and dischargeData[startIdx] < EPSILON):
            startIdx += 1
    except IndexError:  # If we iterate through the entire dataset without detecting a charge, then there are 0 of both cycles
        return np.empty(0), np.empty(0)
    # This variable name 'charging' is misleading for the first data point, since we're intentionally setting it to the opposite
    # value in order to have the loop below register it properly
    charging = not chargeData[startIdx] >= EPSILON

    for i in range(startIdx, len(chargeData)):
        # Store all recent current measurements in a FIFO queue to maintain recent data points accessible
        recentQueue.put(chargeData[i])                  # Add the new datapoint to the queue
        if (recentQueue.qsize() > numPoints):           # Cap the size of the queue to numPoints
            recentQueue.get()                           # Remove the oldest point once full

        if charging:
            if not np.any(list(recentQueue.queue)):     # Consider discharging if all recent data points are 0
                charging = False
                chargeEnds = np.append(chargeEnds, i - numPoints)   # Add the point where the transition BEGAN (numPoints before detected) to the list of marked points
        else:
            if np.all(list(recentQueue.queue)):
                charging = True
                chargeStarts = np.append(chargeStarts, i - numPoints)
    # Once we reach the end of the dataset, append a final (start) point to close the cycles
    if charging:    # If we ended in charging state
        chargeEnds = np.append(chargeEnds, np.max(np.where(dischargeData)))  # Append the last non-zero data point
    else:           # Ended in discharging state
        chargeStarts = np.append(chargeStarts, np.max(np.where(dischargeData)))

    if printing:
        dischargeCycles = len(chargeEnds)
        chargeCycles = len(chargeStarts)

        if charging:                                     # If everything ended as charging, meaning discharging was the last full step
            dischargeCycles -= 1
        else:
            chargeCycles -= 1

        print('Number of discharge cycles = {}'.format(dischargeCycles))
        print('Number of charge cycles = {}'.format(chargeCycles))

    return chargeStarts, chargeEnds


# Converts arrays containing the start points for charging and discharging cycles to
# 2D array countaining the boundaries for each curve
def startsToBounds(chargeStarts, chargeEnds):
    # This means the sensor looked at wasn't used -> no relevant charging or discharge data
    if len(chargeStarts) == 0 or len(chargeEnds) == 0:
        return np.empty((0,) * 2), np.empty((0,) * 2)   # Return empty 2D arrays to allow later functions to not fail

    chargeFirst = chargeStarts[0] < chargeEnds[0]       # We started the measurements by charging

    # Store the bounds in 2D arrays where each column corresponds to (start, end)
    if chargeFirst:
        if len(chargeStarts) > len(chargeEnds):     # This means we started charging and ended with a discharge cycle
            # discount the last point in chargeStarts when determining charging cycles (but do need it for discharge)
            chargeBounds = np.stack((chargeStarts[:-1], chargeEnds[:]), axis=1)
            dischargeBounds = np.stack((chargeEnds[:], chargeStarts[1:]), axis=1)
        else:   # The two lengths are equal, meaning we started charging and ended on a charge cycle (less common)
            chargeBounds = np.stack((chargeStarts[:], chargeEnds[:]), axis=1)
            dischargeBounds = np.stack((chargeEnds[:-1], chargeStarts[1:]), axis=1)
    else:   # If we started discharging
        if len(chargeEnds) > len(chargeStarts):     # This means we started charging and ended with a discharge cycle
            # discount the last point in chargeStarts when determining charging cycles (but do need it for discharge)
            chargeBounds = np.stack((chargeStarts[:], chargeEnds[:-1]), axis=1)
            dischargeBounds = np.stack((chargeEnds[1:], chargeStarts[:]), axis=1)
        else:   # This means we started charging and ended on a charge cycle (less common)
            chargeBounds = np.stack((chargeStarts[:-1], chargeEnds[1:]), axis=1)
            dischargeBounds = np.stack((chargeEnds[:], chargeEnds[:]), axis=1)
    return chargeBounds, dischargeBounds


# Converts continuous charging/discharging curves to only segments where it's either fully
# charging or discharging.  Can be used for voltage, current, and power measurements
def segmentData(chargeBounds, dischargeBounds, data):
    dischargeData = []
    chargeData = []
    # Go row by row through
    for (start, end) in chargeBounds:
        chargeData.append(data[start:end])
    for (start, end) in dischargeBounds:
        dischargeData.append(data[start:end])
    return chargeData, dischargeData


# Generates plots containing charge and discharge curves.  Used for load and supply currents
# Useful for troubleshooting the segmentation into charging and discharging sections
def plotCycles(chargeData, dischargeData, chargeStarts, dischargeStarts):
    if not np.any(chargeStarts) or not np.any(dischargeStarts):     # If no cycles were detected, then quit
        return
    # Plot the raw data from each current supply/load
    plt.plot(chargeData, label='Supply Current')
    plt.plot(dischargeData, label='Load Current')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.title('Supply and Load Curents with Detected Cycles')

    # Did charging or discharging occur first?
    startedCharging = chargeStarts[0] < dischargeStarts[0]
    for i in range(np.min([len(dischargeStarts), len(chargeStarts)])):
        if (startedCharging):
            plt.axvspan(chargeStarts[i], dischargeStarts[i], alpha=0.2, color='green')
            try:
                plt.axvspan(dischargeStarts[i], chargeStarts[i + 1], alpha=0.2, color='red')
            except IndexError:
                pass
        else:
            plt.axvspan(dischargeStarts[i], chargeStarts[i], alpha=0.2, color='red')
            try:
                plt.axvspan(chargeStarts[i], dischargeStarts[i + 1], alpha=0.2, color='green')
            except IndexError:
                pass
    # Add an invisible section of each coloring in order to add it to the legend
    plt.axvspan(chargeStarts[0], chargeStarts[0], alpha=0.2, color='green', label='Charging')
    plt.axvspan(dischargeStarts[0], dischargeStarts[0], alpha=0.2, color='red', label='Discharging')

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.legend()
    plt.show()


# ======================================================================================
# Sophisticated getter functions that return the parameter of interest segmented into charging & discharging chunks
chargeBounds = np.empty((0,) * 2)   # Creates a 2D array of size 0
dischargeBounds = np.empty((0,) * 2)
currentBoundNumber = 0     # Keeps track of if the current boundaries saved are for the sensors that we want.  If not, recalculate


# Public-facing getter that returns the indices in time for each charging/discharging cycle
# Improves efficiency by not requiring recalculation of bounds every single time
def getBounds(dataDir, sensorNum):
    if currentBoundNumber != sensorNum:
        setBounds(dataDir, sensorNum)
    return chargeBounds, dischargeBounds


# This should only be called from within getBounds in order to eliminate confusion
def setBounds(dataDir, sensorNum):
    global chargeBounds, dischargeBounds, currentBoundNumber            # Load these parameters in and make editable
    currentBoundNumber = sensorNum
    supplyCurrent = getters.getSupplyCurrentData(dataDir, sensorNum)
    loadCurrent = getters.getLoadCurrentData(dataDir, sensorNum)
    chargeStarts, dischargeStarts = determineCycles(supplyCurrent, loadCurrent, printing=False)     # Gets the index at which each cycle begins
    chargeBounds, dischargeBounds = startsToBounds(chargeStarts, dischargeStarts)                   # Converts starting indices to 2D arrays
    return np.array(chargeBounds), np.array(dischargeBounds)


# Returns the number of charging and discharging cycles, in case they're not the same
def getCycles(dataDir, sensorNum):
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    return chargeBounds.shape[0], dischargeBounds.shape[0]

def getCycles1(dataDir, sensorNum):
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    return chargeBounds.shape[0], dischargeBounds.shape[0],chargeBounds,dischargeBounds


# Call this function when plotting any of the continuous sensors in order to include shadings in the
# plot background indicating charging state
def colorPlotCycles(dataDir, sensorNum):
    # Even if an unused sensor is chosen, this should not crash
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    for i, bound in enumerate(chargeBounds):
        plt.axvspan(bound[0], bound[1], alpha=0.2, color='green')
    for bound in dischargeBounds:
        plt.axvspan(bound[0], bound[1], alpha=0.2, color='red')
    if len(chargeBounds) > 0 and len(dischargeBounds) > 0:    # Only include legend if there are actually cycles to plot
        plt.axvspan(chargeBounds[0, 1], chargeBounds[0, 1], alpha=0.2, color='green', label='Charging')
        plt.axvspan(dischargeBounds[0, 1], dischargeBounds[0, 1], alpha=0.2, color='red', label='Discharging')
        plt.legend()


def getSegmentedTime(dataDir, sensorNum, dt):
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    timeC = np.array([])
    timeD = np.array([])
    for bound in chargeBounds:
        elapsedT = (bound[1] - bound[0]) * dt / (60 * 60)
        timeC = np.append(timeC, elapsedT)
    for bound in dischargeBounds:
        elapsedT = (bound[1] - bound[0]) * dt / (60 * 60)
        timeD = np.append(timeD, elapsedT)
    return timeC, timeD


def getSegmentedSupplyCurrent(dataDir, sensorNum):
    supplyCurrent = getters.getSupplyCurrentData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    supplyCurrentC, supplyCurrentD = segmentData(chargeBounds, dischargeBounds, supplyCurrent)
    return supplyCurrentC, supplyCurrentD


def getSegmentedLoadCurrent(dataDir, sensorNum):
    loadCurrent = getters.getLoadCurrentData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    loadCurrentC, loadCurrentD = segmentData(chargeBounds, dischargeBounds, loadCurrent)
    return loadCurrentC, loadCurrentD


def getSegmentedSupplyPower(dataDir, sensorNum):
    supplyPower = getters.getSupplyPowerData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    supplyPowerC, supplyPowerD = segmentData(chargeBounds, dischargeBounds, supplyPower)
    return supplyPowerC, supplyPowerD


def getSegmentedLoadPower(dataDir, sensorNum):
    loadPower = getters.getLoadPowerData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    loadPowerC, loadPowerD = segmentData(chargeBounds, dischargeBounds, loadPower)
    return loadPowerC, loadPowerD


def getSegmentedSupplyVoltage(dataDir, sensorNum):
    supplyVoltage = getters.getSupplyVoltageData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    supplyVoltageC, supplyVoltageD = segmentData(chargeBounds, dischargeBounds, supplyVoltage)
    return supplyVoltageC, supplyVoltageD


def getSegmentedLoadVoltage(dataDir, sensorNum):
    loadVoltage = getters.getLoadVoltageData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    supplyVoltageC, supplyVoltageD = segmentData(chargeBounds, dischargeBounds, loadVoltage)
    return supplyVoltageC, supplyVoltageD


def getSegmentedConductivity(dataDir, sensorNum):
    conductivity = getters.getConductivityData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    conductivityC, conductivityD = segmentData(chargeBounds, dischargeBounds, conductivity)
    return conductivityC, conductivityD


def getSegmentedTemperature(dataDir, sensorNum):
    temperature = getters.getTemperatureData(dataDir, sensorNum)
    chargeBounds, dischargeBounds = getBounds(dataDir, sensorNum)
    tempC, tempD = segmentData(chargeBounds, dischargeBounds, temperature)
    return tempC, tempD


# =======================================================================================


# If the program is launched from this app, plot the charge/discharge cycles for testing & debugging
if __name__ == '__main__':
    import plotInterface as pi
    path = pi.importFiles()
    supplyCurrent, loadCurrent = getters.getSupplyCurrentData(path, 4), getters.getLoadCurrentData(path, 4)
    chargeStarts, dischargeStarts = determineCycles(supplyCurrent, loadCurrent, printing=True)
    plotCycles(supplyCurrent, loadCurrent, chargeStarts, dischargeStarts)
    chargeBounds, dischargeBounds = startsToBounds(chargeStarts, dischargeStarts)
    supplyCurrentC, supplyCurrentD = segmentData(chargeBounds, dischargeBounds, supplyCurrent)
    loadCurrentC, loadCurrentD = segmentData(chargeBounds, dischargeBounds, loadCurrent)
    for i in range(len(dischargeBounds)):
        plt.plot(np.arange(dischargeBounds[i][0], dischargeBounds[i][1]), loadCurrentD[i])
    plt.show()
