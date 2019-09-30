# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:26:03 2019

@author: Media Markt
"""

filterData = input('Filter Data (y/n)\n') in 'yY'
conductivities = [getters.getConductivityData(dataDir, i, filtering=filterData) for i in range(1, 7)]     # Conductivity in mS / cm
temps = [getters.getTemperatureData(dataDir, i, filtering=filterData) for i in range(1, 7)]
concentrations = [calc.getConcentration(conductivities[i], temps[i]) for i in range(len(conductivities))]               # Concentration in mols / L

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
        formatData = dict([(k, pd.Series(v)) for k, v in data.items()])     # Makes all columns the same lengtyh
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
    fig.add_subplot(3, 2, i)
    plt.plot(conc, label='Data')
    plt.title('Conductivity Sensor {}'.format(i))
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mol/L)')
    if includeCycles:
        dataReader.colorPlotCycles(dataDir, sensorUsed)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
