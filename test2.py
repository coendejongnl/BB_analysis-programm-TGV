concentrations = [calc.getConcentration(conductivities[i], temps[i]) for i in range(len(conductivities))]               # Concentration in mols / L

concentrations=[]
for i in range(len(conductivities)):
    concentrations=np.append(concentrations,calc.getConcentration(conductivities[i], temps[i]))
    
    
    