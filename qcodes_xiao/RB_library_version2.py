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
#from RB_C2 import convert_clifford_to_sequence, clifford_sets




#from RB_test import convert_clifford_to_sequence, clifford_sets

class RB(Manipulation):

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
        self.start = kw.pop ('start', '1')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.clifford_number = kw.pop('clifford_number', self.clifford_number)
        self.sequence_number = kw.pop('sequence_number', self.sequence_number)
        
        self.start = kw.pop('start', self.start)

        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        
        self.start = kw.pop('start', self.start)
        
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]
        
        print(clifford_gates)
        
        name = 'prepare_state'
#        self.add_single_qubit_gate(name = name, qubit = qubit, amplitude = 1, 
#                                   length = qubit.Pi_pulse_length,)

        for i in range(len(clifford_gates)):
            print('go to next clifford : ', i)
            
            for j in range(len(clifford_gates[i])):
                
                gate = clifford_gates[i][j]
                amplitude = 0 if gate == 'I' else 1
                if gate.startswith('X'):
                    axis = [1,0,0]
                elif gate.startswith('mX'):
                    axis = [-1,0,0]
                elif gate.startswith('Y'):
                    axis = [0,1,0]
                else:
                    axis = [0,-1,0]
                    
                length = qubit.Pi_pulse_length if gate.endswith('p') else qubit.halfPi_pulse_length
            
                refgate = None if i+j == 0 else name
#                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                self.add_single_qubit_gate(name = name, refgate = refgate, 
                                       qubit = qubit, axis = axis,
                                       amplitude = amplitude, length = length,)
            
        print('clifford_gates finished')
        
        return self
    

#%%


from RB_test_version2 import convert_clifford_to_sequence, clifford_sets1, clifford_sets2

class RB_all(Manipulation):

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
        
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', 0.95)
        
        self.align = kw.pop('align', False)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.clifford_number = kw.pop('clifford_number', self.clifford_number)
        self.sequence_number = kw.pop('sequence_number', self.sequence_number)
        
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', self.off_resonance_amplitude)
        
        self.align = kw.pop('align', self.align)

        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        qubit_obj = {'qubit_1': qubit_1,
                     'qubit_2': qubit_2
                     }
        
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', self.off_resonance_amplitude)
        
        self.align = kw.pop('align', self.align)
        
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates1 = clifford_sets1[self.sequence_number][self.clifford_number]
        
        clifford_gates2 = clifford_sets2[self.sequence_number][self.clifford_number]
        
        print(clifford_gates1)
        print(clifford_gates2)
        
        clifford_gates = {
                'qubit_1': clifford_gates1,
                'qubit_2': clifford_gates2
                }

        axis = {'Xp': [1,0,0],
                'X9': [1,0,0],
                'mXp': [-1,0,0],
                'mX9': [-1,0,0],
                'Yp': [0,1,0],
                'Y9': [0,1,0],
                'mYp': [0,-1,0],
                'mY9': [0,-1,0]
                }
        
        length = {'Xp': qubit_1.Pi_pulse_length,
                  'X9': qubit_1.halfPi_pulse_length,
                  'mXp': qubit_1.Pi_pulse_length,
                  'mX9': qubit_1.halfPi_pulse_length,
                  'Yp': qubit_1.Pi_pulse_length,
                  'Y9': qubit_1.halfPi_pulse_length,
                  'mYp': qubit_1.Pi_pulse_length,
                  'mY9': qubit_1.halfPi_pulse_length,
                  'I': 0,# if not self.align else 10e-9,
                  'None': 0
                  }

        gate_name = 'prepare_state'
        
        length_total1 = 0
        length_total2 = 0

        for i in range(len(clifford_gates)):
            
            print('go to next clifford : ', i)
            
            length_clifford1 = sum([length[gate] for gate in clifford_gates1[i]])
            length_clifford2 = sum([length[gate] for gate in clifford_gates2[i]])
            
            length_total1 += length_clifford1
            length_total2 += length_clifford2
            
            for qubit in ['qubit_1', 'qubit_2']:
                
                clifford = clifford_gates[qubit][i]
                
                for j in range(len(clifford)):
                    
                    gate = clifford[j]
                    
                    if gate == 'I':
                        if length['I'] == 0:
                            continue
                        amp = self.offresonance_amplitude if qubit == 'qubit_1' else 0
                        freq_shift = -30e6 if qubit == 'qubit_1' else 0
                    else:
                        amp = 1
                    
                    axis = axis[gate]
                    length = length[gate]
                    
                    last_gate = gate_name
                    
                    if i+j == 0:
                        refgate = None if qubit == 'qubit_1' else last_gate
                        refpoint = 'start' if qubit == 'qubit_2' else None
                    else:
                        refgate = last_gate+qubit
                        refpoint = 'end'
                        
                    gate_name = 'C%d%d'%((i+1),(j+1))+gate
                    element_name = gate_name + qubit
                    
                    self.add_single_qubit_gate(name = element_name, refgate = refgate, refpoint = refpoint,
                                               qubit = qubit_obj[qubit], axis = axis,
                                               amplitude = amp, length = length, frequency_shift = freq_shift)
            if self.align:
                if length_clifford1 > length_clifford2:
                    self.add_single_qubit_gate(name = 'Null', refgate = last_gate+'qubit_2', refpoint = 'end',
                                               qubit = qubit_2, axis = [1,0,0],
                                               amplitude = 0, 
                                               length = length_clifford1-length_clifford2,)

                elif length_clifford2 > length_clifford1:
                    self.add_single_qubit_gate(name = 'off_resonance', refgate = last_gate+'qubit_1', refpoint = 'end',
                                               qubit = qubit_1, axis = [1,0,0],
                                               amplitude = self.off_resonance_amplitude, 
                                               length = length_clifford2-length_clifford1, frequency_shift = -30e6)
        
        
        if length_total1 < length_total2 and not self.align:
            self.add_single_qubit_gate(name = 'off_resonance', refgate = last_gate+'qubit_1', refpoint = 'end',
                                       qubit = qubit_1, axis = [1,0,0],
                                       amplitude = self.off_resonance_amplitude, 
                                       length = length_total2-length_total1, frequency_shift = -30e6)
         
        print('clifford_gates finished')
        
        return self
#%%


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
        self.phase_1 = kw.pop('phase_1', 178)
        self.phase_2 = kw.pop('phase_2', 302)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)

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
            length_1 = 5e-9
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
                                    amplitude_control = 30*0.5*-0.0267, amplitude_target = 30*0.5*0.02, 
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
                    else:
                        raise NameError('Gate name not correct')
                    
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
                    else:
                        raise NameError('Gate name not correct')
                
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

#%%


class RB_full(Manipulation):

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
                
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                amplitude = 1
                
                if gate == 'CZ':
                    
                    self.add_CPhase(name = name+'1', refgate = refgate, waiting_time = 10e-9,
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 30*0.5*-0.0256, amplitude_target = 30*0.5*0.04, 
                                    length = self.detuning_time)
                    self.add_Z(name='Z1_Q1', qubit = self.qubits[0], degree = self.phase_1)
                    self.add_Z(name='Z1_Q2', qubit = self.qubits[1], degree = self.phase_2)
                    
                    self.add_CPhase(name = name+'2', refgate = name+'1', 
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 0, amplitude_target = 0, 
                                    length = 10e-9)
                    name = name+'2'
                
                elif gate.startswith('Clifford') or gate.startswith('S1'):
                    
#                    group = 'Clifford' if gate.startswith('Cli') else 'S1'
                    clas = Clifford_gates if gate.startswith('Cli') else S1_gates
                    
                    for w in range(len(group)):
                        if group[w] == '_':
                            c1 = w
                        if group[w] == '/':
                            c2 = w
                    cli_1 = group[c1+1:c2]
                    cli_2 = group[c2+1:]
                    
                    total_duration = [0,0]
                    
                    for qu in [1,2]:
                        cli = cli_1 if qu == 1 else cli_2
                        qubit = qubit_1 if qu == 1 else qubit_2

                        for operation in Clifford_gates[cli_1]:
                            
                            if operation.startswith('X'):
                                axis = [1,0,0]
                            elif operation.startswith('mX'):
                                axis = [-1,0,0]
                            elif operation.startswith('Y'):
                                axis = [0,1,0]
                            else:
                                axis = [0,-1,0]
                    
                            length = qubit_1.Pi_pulse_length if gate.endswith('p') else qubit_1.halfPi_pulse_length
                            total_duration[qu-1] += length
                            
                            self.add_single_qubit_gate(name = name + 'Q' + str(qu) + '_' , refgate = refgate, 
                                                       qubit = qubit, axis = axis,
                                                       amplitude = amplitude, length = length,)
                    name = 'Q1'
                    if total_duration[0]<total_duration[1]:
                        off_resonance_duration = total_duration[1] - total_duration[0]
                        self.add_single_qubit_gate(name = 'off_resonance2'+name, refgate = name,
                                                   qubit = qubit_1, amplitude = 1.15, 
                                                   length = length, frequency_shift = -30e6)
#                       name = name +'Q2'

#%%

class RB_decoupling(Manipulation):

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
                
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                amplitude = 1
                
                if gate == 'CZ':
                    
                    self.add_CPhase(name = name+'1', refgate = refgate, waiting_time = 10e-9,
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 30*0.5*-0.0256, amplitude_target = 30*0.5*0.04, 
                                    length = self.detuning_time)
                    self.add_Z(name='Z1_Q1', qubit = self.qubits[0], degree = self.phase_1)
                    self.add_Z(name='Z1_Q2', qubit = self.qubits[1], degree = self.phase_2)
                    
                    self.add_CPhase(name = name+'2', refgate = name+'1', 
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 0, amplitude_target = 0, 
                                    length = 10e-9)
                    name = name+'2'
                
                elif gate.startswith('Clifford') or gate.startswith('S1'):
                    
#                    group = 'Clifford' if gate.startswith('Cli') else 'S1'
                    clas = Clifford_gates if gate.startswith('Cli') else S1_gates
                    
                    for w in range(len(group)):
                        if group[w] == '_':
                            c1 = w
                        if group[w] == '/':
                            c2 = w
                    cli_1 = group[c1+1:c2]
                    cli_2 = group[c2+1:]
                    
                    total_duration = [0,0]
                    
                    for qu in [1,2]:
                        cli = cli_1 if qu == 1 else cli_2
                        qubit = qubit_1 if qu == 1 else qubit_2

                        for operation in Clifford_gates[cli_1]:
                            
                            if operation.startswith('X'):
                                axis = [1,0,0]
                            elif operation.startswith('mX'):
                                axis = [-1,0,0]
                            elif operation.startswith('Y'):
                                axis = [0,1,0]
                            else:
                                axis = [0,-1,0]
                    
                            length = qubit_1.Pi_pulse_length if gate.endswith('p') else qubit_1.halfPi_pulse_length
                            total_duration[qu-1] += length
                            
                            self.add_single_qubit_gate(name = name + 'Q' + str(qu) + '_' , refgate = refgate, 
                                                       qubit = qubit, axis = axis,
                                                       amplitude = amplitude, length = length,)
                    name = 'Q1'
                    if total_duration[0]<total_duration[1]:
                        off_resonance_duration = total_duration[1] - total_duration[0]
                        self.add_single_qubit_gate(name = 'off_resonance2'+name, refgate = name,
                                                   qubit = qubit_1, amplitude = 1.15, 
                                                   length = length, frequency_shift = -30e6)
#                       name = name +'Q2'


                
                                
              