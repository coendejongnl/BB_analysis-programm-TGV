import numpy as np
import matplotlib.pyplot as plt
import dataGetters as getters
import calculations as calc
import tkinter as tk
from tkinter import filedialog
import dataReader


def theoretical_resistance(flowrate_s_raw, flowrate_r_raw, concentration_s, concentration_f, current, a="f"):

    #############fuji film resistance############
    if a in "Ff":
        number_of_cells = 512
        surface_area = 0.22**2
        L = .22
        resistance_m2_A = 1.7/100**2
        resistance_m2_C = 2/100**2
        width_flow_canal = 165e-6

    if a in "Ee":
        number_of_cells = 512
        surface_area = 0.22**2
        L = .22
        # 9.7 ohm/cm**2
        resistance_m2_A = 9.7/100**2
        resistance_m2_C = 0/100**2
        width_flow_canal = 165e-6

    # same length function
    def same_length(a):
        for i in a:
            #            print(len(i))
            try:
                min_len = np.min([len(i), min_len])

            except:
                min_len = len(i)

        new_list = []
        for i in a:
            new_list.append(np.array(i[0:min_len]))
#            print(len(i[0:min_len]))

        return(new_list)
    ################R membranes############
    resistance_A = number_of_cells*(resistance_m2_A/surface_area)
    resistance_C = number_of_cells*(resistance_m2_C/surface_area)

    R_membranes = resistance_A+resistance_C

    ####################R_c###################
    alpha = 0.97
    R = 8.314
    T = 293.15
    F = 96485
    J = current/surface_area  # current /density
    flowrate_r = flowrate_r_raw/(1000*60)
    flowrate_s = flowrate_s_raw/(1000*60)

    q_r = flowrate_r/width_flow_canal/512  # m**2/s     
    q_s = flowrate_s/width_flow_canal/512

    C_r = concentration_f*1000  # mol/m**3
    C_s = concentration_s*1000

    (J, C_r, C_s, q_r, q_s) = same_length((J, C_r, C_s, q_r, q_s))

    delta_ar = 1+np.divide(J*L, F*q_r * C_r)
    delta_as = 1-np.divide(J*L, F*q_s * C_s)

    with np.errstate(all='ignore'):
        R_c = np.where(J != 0, alpha*R*T/(F*J) *
                       np.log(delta_ar/delta_as)*512, np.nan)

    return(R_c, R_membranes)


if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()

    dataDir = filedialog.askdirectory()+"/"

    # first trial with the fujifilm
    while True:
        a = str(input("(F)ujifilm of (E)voque?\n"))
        if a in "FfEe":
            break

    filterData = input('Filter Data (y/n)\n') in 'yY'

    # data to load in
    flows = np.array([getters.getFlowData(dataDir, i) for i in range(1, 3)])

    conductivities = [getters.getConductivityData(
        dataDir, i, filtering=filterData) for i in range(1, 7)]
    concentrations = [calc.getConcentration(conductivities[i], np.ones(
        shape=conductivities[i].shape)*20) for i in range(len(conductivities))]
    LC = getters.getLoadCurrentData(dataDir, 3)
    SC = getters.getSupplyCurrentData(dataDir, 3)
    R_c, R_membranes = theoretical_resistance(
        flows[0], flows[1], concentrations[0], concentrations[1], LC, a)

    plt.stackplot(np.arange(len(R_c)), R_c, np.full(R_c.shape, R_membranes))
