import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('network_data.csv')  # Loads are in kW and kVAR.

n_buses = 7  # 6-bus & 1 slack (first)
slack_voltage = 1.0  # Voltage at slack bus (in per unit)
tolerance = 1e-6  # Convergence tolerance
max_iter = 500  # Maximum number of iterations

Sbase=100  # Sbase = 100 kVA;
Zbase=400*400/100000;  # Used Vbase=400V

voltages = np.ones(n_buses, dtype=complex)
bus_injections = np.zeros(n_buses, dtype=complex)
branch_currents = np.zeros(n_buses - 1, dtype=complex)

load_P = np.array(df['P'], dtype=float)
load_Q = np.array(df['Q'], dtype=float)
load_S = load_P + 1j * load_Q  
S_pu=load_S/Sbase

R = np.array(df['R'], dtype=float)
X = np.array(df['X'], dtype=float)
Z =( R + 1j * X ) /Zbase 

for iter_count in range(max_iter):
    
    prev_voltages = np.copy(voltages)
    
    bus_injections = np.zeros(n_buses, dtype=complex)
    branch_currents = np.zeros(n_buses - 1, dtype=complex)

    for i in reversed(range(n_buses - 1)):
        from_bus = df.loc[i, 'From Bus']
        to_bus = df.loc[i, 'To Bus']
    
        bus_injections[to_bus] = np.conj(S_pu[i] / voltages[to_bus])  # I = conj(S / V)
    
        
        branch_currents[i] = bus_injections[to_bus]
    
        
        for j in range(i+1, n_buses - 1):
            if df.loc[j, 'From Bus'] == to_bus:
                branch_currents[i] += branch_currents[j]
    
    voltages[0] = slack_voltage  
    for i in range(n_buses - 1):
        from_bus = df.loc[i, 'From Bus']
        to_bus = df.loc[i, 'To Bus']
        voltages[to_bus] = voltages[from_bus] - branch_currents[i] * Z[i]  
        
    if np.all(np.abs(voltages - prev_voltages) < tolerance):
        break

bus_results = pd.DataFrame({
    'Bus Number': np.arange(n_buses),
    'Voltage Magnitude (p.u.)': np.abs(voltages),
    'Voltage Angle (degrees)': np.angle(voltages, deg=True),
    'Bus Injection Magnitude (MVA)': np.abs(bus_injections),
    'Bus Injection Angle (degrees)': np.angle(bus_injections, deg=True)
})

branch_results = pd.DataFrame({
    'From Bus': df['From Bus'],
    'To Bus': df['To Bus'],
    'Branch Current Magnitude (MVA)': np.abs(branch_currents),
    'Branch Current Angle (degrees)': np.angle(branch_currents, deg=True)
})

bus_results, branch_results

voltage_magnitudes = np.abs(voltages)
current_magnitudes = np.abs(branch_currents)

power_losses = current_magnitudes**2 * np.abs(Z)  # This is in p.u.
total_losses_kva=np.sum(power_losses*100)  # Losses in kVA. Sbase = 100 kVA

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.bar(np.arange(len(voltage_magnitudes)), voltage_magnitudes, color='b')
plt.xlabel('Bus Number')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.subplot(1, 3, 2)
plt.bar(np.arange(len(current_magnitudes)), current_magnitudes, color='r')
plt.xlabel('Branch Number')
plt.ylabel('Current Magnitude (p.u.)')
plt.subplot(1, 3, 3)
plt.bar(np.arange(len(power_losses)), power_losses, color='g')
plt.xlabel('Branch Number')
plt.ylabel('Losses (MVA)')
plt.tight_layout()
plt.show()