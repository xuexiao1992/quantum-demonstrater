# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:01:07 2017

@author: X.X
"""

import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from qcodes.instrument.base import Instrument
#from experiment import Experiment
from manipulation import Manipulation
import stationF006

#%%
from RB_C2 import convert_clifford_to_sequence, clifford_sets

class RB_Martinis(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.clifford_number = kw.pop('clifford_number', 0)
        self.sequence_number = kw.pop('sequence_number', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.phase_1 = kw.pop('phase_1', 116)
        self.phase_2 = kw.pop('phase_2', 175)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.clifford_number = kw.pop('clifford_number', self.clifford_number)
        self.sequence_number = kw.pop('sequence_number', self.sequence_number)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)


        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]

        print(clifford_gates)
        
        name = 'prepare_state'
        
        if self.Pi_amplitude == 0:
            length_1 = 10e-9
        else:
            length_1 = qubit_2.Pi_pulse_length
        self.add_single_qubit_gate(name = name, qubit = qubit_2, amplitude = self.Pi_amplitude, 
                                   length = length_1,)
        self.add_single_qubit_gate(name = name+'X2', refgate = name, refpoint = 'start',
                                   qubit = qubit_1, amplitude = self.Pi_amplitude, 
                                   length = length_1)
#        self.add_single_qubit_gate(name = 'off_resonance1', refgate = name, refpoint = 'start',
#                                   qubit = qubit_1, amplitude = 1.2, 
#                                   length = qubit_1.Pi_pulse_length, frequency_shift = -30e6)

        for i in range(len(clifford_gates)):
            print('go to next clifford : ', i)
            for j in range(len(clifford_gates[i])):
                gate = clifford_gates[i][j]
                
                if gate == 'I':
                    continue
                
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                amplitude = 1
                
                if 'X' in gate or 'Y' in gate:

                    if gate.startswith('X'):
                        axis = [1,0,0]
                    elif gate.startswith('mX'):
                        axis = [-1,0,0]
                    elif gate.startswith('Y'):
                        axis = [0,1,0]
                    else:
                        axis = [0,-1,0]
                    
                    length = qubit_1.Pi_pulse_length if gate.endswith('p') else qubit_1.halfPi_pulse_length
                    
#                    refgate = None if i+j == 0 else name
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = axis,
                                               amplitude = amplitude, length = length,)


                    self.add_single_qubit_gate(name = name+'Q2', refgate = refgate,
                                               qubit = qubit_2, axis = axis,
                                               amplitude = amplitude, length = length,)
#                    
#                    self.add_single_qubit_gate(name = 'off_resonance2'+name, refgate = name,
#                                               qubit = qubit_1, amplitude = 1.2, 
#                                               length = length, frequency_shift = -30e6)
#                    name = name +'Q2'

                elif gate == 'CZ':
                    
                    self.add_CPhase(name = name+'1', refgate = refgate, waiting_time = 10e-9,
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 30*0.5*-0.0254, amplitude_target = 30*0.5*0.04, 
                                    length = self.detuning_time)
                    self.add_Z(name='Z1_Q1', qubit = self.qubits[0], degree = self.phase_1)
                    self.add_Z(name='Z1_Q2', qubit = self.qubits[1], degree = self.phase_2)
                    
                    self.add_CPhase(name = name+'2', refgate = name+'1', 
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 0, amplitude_target = 0, 
                                    length = 10e-9)
                    name = name+'2'
                                
                elif gate == 'iSWAP_1':
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = [0,1,0],
                                               amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                    self.add_single_qubit_gate(name = name+'Q2_1', refgate = name, refpoint = 'start',
                                               qubit = qubit_2, axis = [-1,0,0],
                                               amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                
                elif gate.startswith('iSWAP'):
                    
                    if gate.endswith('S0'):
                        self.add_single_qubit_gate(name = name, refgate = refgate, 
                                                   qubit = qubit_1, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                        self.add_single_qubit_gate(name = name+'Q2_1', refgate = name, refpoint = 'start',
                                                   qubit = qubit_2, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                    elif gate.endswith('S1'):
                        self.add_single_qubit_gate(name = name+'Q1_1', refgate = refgate, 
                                                   qubit = qubit_1, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
                        
                        self.add_single_qubit_gate(name = name+'Q1_2', refgate = name+'Q1_1', 
                                                   qubit = qubit_1, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)

                        self.add_single_qubit_gate(name = name+'Q2_1', refgate = name+'Q1_1', refpoint = 'start',
                                                   qubit = qubit_2, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q2_2', refgate = name+'Q2_1',
                                                   qubit = qubit_2, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q2_3', refgate = name+'Q2_2',
                                                   qubit = qubit_2, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        name = name +'Q1_2'
                        
                    elif gate.endswith('S2'):
                        self.add_single_qubit_gate(name = name+'Q1_1', refgate = refgate, 
                                                   qubit = qubit_1, axis = [-1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                        self.add_single_qubit_gate(name = name+'Q1_2', refgate = name+'Q1_1', 
                                                   qubit = qubit_1, axis = [0,-1,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q1_3', refgate = name+'Q1_2', 
                                                   qubit = qubit_1, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)

                        self.add_single_qubit_gate(name = name+'Q2_1', refgate = name+'Q1_1', refpoint = 'start',
                                                   qubit = qubit_2, axis = [0,-1,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        
                        name = name +'Q1_3'
                    
                elif gate.startswith('CNOT'):
                    
                    if gate.endswith('S0'):
                        self.add_single_qubit_gate(name = name, refgate = refgate, 
                                                   qubit = qubit_1, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)

                    elif gate.endswith('S1'):
                        self.add_single_qubit_gate(name = name+'Q1_1', refgate = refgate, 
                                                   qubit = qubit_1, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
                        
                        self.add_single_qubit_gate(name = name+'Q1_2', refgate = name+'Q1_1', 
                                                   qubit = qubit_1, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)

                        self.add_single_qubit_gate(name = name+'Q2_1', refgate = name+'Q1_1', refpoint = 'start',
                                                   qubit = qubit_2, axis = [0,1,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q2_2', refgate = name+'Q2_1',
                                                   qubit = qubit_2, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        name = name +'Q1_2'
                        
                    elif gate.endswith('S2'):
                        self.add_single_qubit_gate(name = name+'Q1_1', refgate = refgate, 
                                                   qubit = qubit_1, axis = [-1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                        self.add_single_qubit_gate(name = name+'Q1_2', refgate = name+'Q1_1', 
                                                   qubit = qubit_1, axis = [0,-1,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q1_3', refgate = name+'Q1_2', 
                                                   qubit = qubit_1, axis = [1,0,0],
                                                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)

                        self.add_single_qubit_gate(name = name+'Q2_1', refgate = name+'Q1_1', refpoint = 'start',
                                                   qubit = qubit_2, axis = [-1,0,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        self.add_single_qubit_gate(name = name+'Q2_2', refgate = name+'Q2_1',
                                                   qubit = qubit_2, axis = [0,-1,0],
                                                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        name = name +'Q1_3'
                
                elif gate == 'SWAP_1':
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = [0,1,0],
                                               amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                    self.add_single_qubit_gate(name = name+'Q2_1', refgate = name, refpoint = 'start',
                                               qubit = qubit_2, axis = [0,-1,0],
                                               amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                elif gate == 'SWAP_2':
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = [0,-1,0],
                                               amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
                    self.add_single_qubit_gate(name = name+'Q2_1', refgate = name, refpoint = 'start',
                                               qubit = qubit_2, axis = [0,1,0],
                                               amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                
                elif gate == 'SWAP_3':
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = [0,1,0],
                                               amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                else:
                    raise NameError('Gate name not correct')
                    
        print('clifford_gates finished')
        
        return self

