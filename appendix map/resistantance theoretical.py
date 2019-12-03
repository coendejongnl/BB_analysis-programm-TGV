import numpy as np
import matplotlib.pyplot as plt

# first trial with the fujifilm
while True:
    a = str(input("(F)ujifilm of (E)voque?\n"))
    if a in "FfEe":
        break


# testing inputs
current = 1
flowrate_s_raw = 1  # L/min=60*0.1**3 m**3/s
flowrate_r_raw = 1

flowrate_s = flowrate_r_raw*60/1000
flowrate_r = flowrate_s_raw*60/1000

concentration_s = 1  # mol/L
concentration_f = 1

#############fuji film resistance############
if a in "Ff":
    number_of_cells = 512
    surface_area = 0.22**2
    L = .22
    resistance_m2_A = 1.7/100**2
    resistance_m2_C = 2/100**2

    resistance_A = number_of_cells*(resistance_m2_A/surface_area)
    resistance_C = number_of_cells*(resistance_m2_C/surface_area)
    width_flow_canal = 165e-6

####################R_c###################
alpha = 0.97
R = 8.314
T = 293.15
F = 96485
J = current/surface_area  # current /density

q_r = flowrate_r/width_flow_canal/512  # m**2/s
q_s = flowrate_s/width_flow_canal/512

C_r = concentration_f*1000  # mol/m**3
C_s = concentration_s*1000

delta_ar = 1+J*L/(F*q_r*C_r)
delta_as = 1-J*L/(F*q_s*C_s)

R_c = alpha*R*T/(F*J)*np.log(delta_ar/delta_as)
