from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_flow_func import load_flow_function

T = range(24)

df = pd.read_csv('load_data.csv')

df['PV'] = df['PV'] * 3  # Increase in PV generation for case-study
df['Load'] = df['Load'] * 1.5  # Increase in load for case-study

PV_data = np.array(df['PV'], dtype=float)
load_data = np.array(df['Load'], dtype=float)
TOU_prices = np.array(df['Price'], dtype=float)

model = ConcreteModel()
model.T = Set(initialize=T)

model.PV = Param(model.T, within=NonNegativeReals, initialize=dict(enumerate(PV_data)))
model.demand = Param(model.T, within=NonNegativeReals, initialize=dict(enumerate(load_data)))
model.price = Param(model.T, within=NonNegativeReals, initialize=dict(enumerate(TOU_prices)))

model.P = Var(model.T, within=NonNegativeReals)

def objective(model):
    return sum(model.price[t] * model.P[t] for t in model.T)

model.obj = Objective(rule=objective, sense=minimize)

def load_constraint_lower(model, t):
    return model.P[t] >= model.demand[t] - model.demand[t]/5

def load_constraint_upper(model, t):
    return model.P[t] <= model.demand[t] + model.demand[t]/5

def sum_constraint(model):
    return sum(model.P[t] for t in model.T) == sum(model.demand[t] for t in model.T)

model.load_constraint_lower = Constraint(model.T, rule=load_constraint_lower)
model.load_constraint_upper = Constraint(model.T, rule=load_constraint_upper)
model.sum_constraint = Constraint(rule=sum_constraint)

solver = SolverFactory('bonmin')
solver.solve(model)

power_consumption = [model.P[t].value for t in model.T]
min_value = np.min(load_data-PV_data)
min_index = np.argmin(load_data-PV_data)

data_point=load_data

df_old=pd.read_csv('network_data.csv')
df_old.loc[5, 'P'] = min_value

voltage_magnitude_old, current_magnitude_old = load_flow_function(df_old)

df_new = df_old.copy()
df_new.loc[5, 'P'] = power_consumption[min_index]-PV_data[min_index]

voltage_magnitude_new, current_magnitude_new = load_flow_function(df_new)

plt.figure(figsize=(12, 6))
plt.plot(T, load_data, label='Initial Load', linestyle='-')
plt.plot(T, power_consumption, label='Power Consumption after TOU', linestyle='-.')
plt.xlabel('Time [Hour]')
plt.ylabel('Power [kW]')
plt.title('Impact of TOU Pricing on Power Consumption')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(T, TOU_prices, linestyle='--', label='TOU Prices', color='g')
ax1.set_xlabel('Time [Hour]')
ax1.set_ylabel('TOU Price [€/kWh]', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(T, PV_data, label='PV Generation (kW)', color='r')
ax2.set_ylabel('PV Generation [kW]', color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.legend(loc='upper right')
plt.show()


plt.plot(voltage_magnitude_new, label='After TOU voltages', color='g')
plt.plot(voltage_magnitude_old, label='Original voltages', color='b')
plt.xlabel('Bus No.')
plt.ylabel('Voltage [p.u.]')
plt.title('Impact of TOU Pricing on Voltages')
plt.legend()
plt.show()