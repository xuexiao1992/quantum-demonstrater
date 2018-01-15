# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:45:29 2017

@author: twatson
"""
#circuit_library...... each parameter is set to default....there is a dictionary of parameters to be et



CZgate = circuit()
CZgate.add_CPhase(name = 'CP_Q12', refgate = 'X1_Q2',
                control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                amplitude_control = detuning_amplitude, amplitude_target = 0, 
                length = detuning_time)

CZgate.add_Z(name='Z1_Q1', qubit = self.qubits[1], degree = phase)


DJ = circuit()
DJ.add_single_qubit_gate(name='X_Pi_Q1', qubit = self.qubits[0])
DJ.add_X(name='X1_Q2', refgate = 'X_Pi_Q1', qubit = self.qubits[1])
DJ.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X1_Q2', refpoint = 'start',
                           qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                           length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)

DJ.add_circuit(CZgate)
DJ.add_X(name='X2_Q2', refgate = 'CP_Q12', qubit = self.qubits[1])
DJ.add_single_qubit_gate(name='off_resonance2_Q1', refgate = 'X2_Q2', refpoint = 'start',
                           qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                           length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)



## sweep experiment

DJ.add_xsweep()
sweep_array = sweep_array(-0.03e9, 0.03e9, 31)
for i in sweeparray:
    for parameter in X_parameters:
        for gate in parameter.gate_paramater:
    DJ.set_parameter(gate, parameter, i)
    Circuits.add_circuit(DJ.create_cicuit)
    Circuits.add_parameter(parameter, i)

Circuit.add_circuit()
Circuit.add_circuit_array()
Circuit.add_x_sweep(parameter = ['X_Pi_Q1', 'frequency'], sweep_array = sweep_array(-0.03e9, 0.03e9, 31))
Circuit.add_y_sweep(parameter = ['X2_Pi_Q1','frequency'], sweep_array = sweep_array(-0.03e9, 0.03e9, 31))
Circuit.generate_circuit_array



experiment.qubit_number = 1

#DJ in an object that contains the information about the sweep. 

experiment.add_measurement('Rabi_Scan', ['Rabi3'], [DJ,], sequence2_cfg, sequence2_cfg_type)
experiment.generate_1D_sequence()
experiment.load_sequence()
# --------




experiment.add_X_parameter('Rabi_Scan', gate_parameter = ['X2_Q2', 'X1_Q2',], parameter = 'frequency', sweep_array = sweep_array(-0.03e9, 0.03e9, 31), element = 'Rabi3')
experiment.add_X_parameter('Rabi_Scan', gate_parameter = ['X2_Q2', 'X1_Q2',], parameter = 'duration', sweep_array = sweep_array(-0.03e9, 0.03e9, 31), element = 'Rabi3')



experiment.set_sweep(repetition = True, plot_average = False, count = 5)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)

