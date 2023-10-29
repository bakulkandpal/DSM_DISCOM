from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

T = range(24)

df = pd.read_csv('load_data.csv')

PV_data = np.array(df['PV'], dtype=float)
load_data = np.array(df['Load'], dtype=float)

model = ConcreteModel()
model.T = Set(initialize=T)

model.PV = Param(model.T, within=NonNegativeReals, initialize=dict(enumerate(PV_data)))
model.demand = Param(model.T, within=NonNegativeReals, initialize=dict(enumerate(load_data)))

model.BESS = Var(model.T, within=Reals)  # Discharging is positive and charging is negative
model.SOC = Var(model.T, within=NonNegativeReals)  

def objective(model):
    return sum((model.demand[t] - model.BESS[t] - model.PV[t]) ** 2 for t in model.T)

model.obj = Objective(rule=objective)

model.initial_SOC = Param(initialize=100)
def SOC_constraint(model, t):
    if t == 0:
        return model.SOC[t] == model.initial_SOC - model.BESS[t]
    return model.SOC[t] == model.SOC[t-1] - model.BESS[t]

model.SOC_constraint = Constraint(model.T, rule=SOC_constraint)

def initial_final_SOC_constraint(model):
    return model.SOC[0] == model.SOC[23]

model.initial_final_SOC_constraint = Constraint(rule=initial_final_SOC_constraint)

solver = SolverFactory('bonmin')
solver.solve(model)

BESS_output = [model.BESS[t].value for t in model.T]

agg_load=load_data-PV_data-BESS_output

plt.figure(figsize=(12, 6))
plt.plot(T, PV_data, label='PV Generation', linestyle='--')
plt.plot(T, load_data, label='Initial Load', linestyle='-')
plt.plot(T, agg_load, label='Final Load with BESS', linestyle='-.')
plt.xlabel('Time [Hour]')
plt.ylabel('Power [kW]')
plt.title('Final Loads after BESS Scheduling')
plt.legend()
plt.show()


plt.plot(T, -np.array(BESS_output), label='BESS Output', linestyle='-')
plt.xlabel('Time [Hour]')
plt.ylabel('BESS Power [kW]')
plt.grid(axis='y', linewidth=0.4, alpha=0.5)