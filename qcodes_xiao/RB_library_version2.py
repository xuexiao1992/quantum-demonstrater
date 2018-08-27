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
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


#%%


def save_object(obj, obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


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
#from RB_test import convert_clifford_to_sequence, clifford_sets_1, clifford_sets_2
#from RB_test_version2 import convert_clifford_to_sequence, clifford_sets_1, clifford_sets_2


class RB_all_test(Manipulation):

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

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.clifford_number = kw.pop('clifford_number', self.clifford_number)
        self.sequence_number = kw.pop('sequence_number', self.sequence_number)

        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates = clifford_sets_1[self.sequence_number][self.clifford_number]
#        clifford_gates2 = clifford_sets2[self.sequence_number][self.clifford_number]
        print(clifford_gates)
        
        name = 'prepare_state'
        self.add_single_qubit_gate(name = name, qubit = qubit_1, amplitude = 1, 
                                   length = qubit_1.Pi_pulse_length,)
        self.add_single_qubit_gate(name = name+'Q2', qubit = qubit_2, amplitude = 1, 
                                   length = qubit_2.Pi_pulse_length,)

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
                    
                length = qubit_1.Pi_pulse_length if gate.endswith('p') else qubit_1.halfPi_pulse_length
                if gate == 'I':
                    length = 10e-9
                if 'Z' in gate:
                    if gate == 'Zp' or gate == 'mZp':
                        degree = 180
                    elif gate == 'Z9':
                        degree = 90
                    elif gate == 'mZ9':
                        degree = -90
                    else:
                        raise ValueError('Z gate degree is wrong')
                    self.add_Z(name='Z_%d%d'%((i+1),(j+1))+'qubit_1', qubit = qubit_1, degree = degree)
                    self.add_Z(name='Z_%d%d'%((i+1),(j+1))+'qubit_2', qubit = qubit_2, degree = degree)
                    continue
                    
#                refgate = None if i+j == 0 else name
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                
                amplitude_1 = amplitude if gate != 'I' else 0.9
                freq_shift = 0 if gate!='I' else -30e6
                amplitude_2 = amplitude
                
                self.add_single_qubit_gate(name = name, refgate = refgate, 
                                       qubit = qubit_1, axis = axis,
                                       amplitude = amplitude_1, length = length, frequency_shift = freq_shift)
                self.add_single_qubit_gate(name = name+'Q2', refgate = refgate, #refpoint = 'start',
                                       qubit = qubit_2, axis = axis,
                                       amplitude = amplitude_2, length = length,)
            
        print('clifford_gates finished')
        
        return self

#%%
from RB_test_version2 import convert_clifford_to_sequence, clifford_sets_1, clifford_sets_2
#from RB_test import convert_clifford_to_sequence, clifford_sets_1, clifford_sets_2

'''
clifford_sets_1 = load_object('clifford_sets_1')
clifford_sets_2 = load_object('clifford_sets_2')
'''
#clifford_sets_1 = load_object('interleave_clifford_sets_1')
#clifford_sets_2 = load_object('interleave_clifford_sets_2')

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
        
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', 1.15)
        
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.03)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.02)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.phase_1 = kw.pop('phase_1', 46)
        self.phase_2 = kw.pop('phase_2', 27)
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
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
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
        
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', self.off_resonance_amplitude)
        
        self.align = kw.pop('align', self.align)
        self.align = False
        te = 15e-9
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates1 = clifford_sets_1[self.sequence_number][self.clifford_number]
        
        clifford_gates2 = clifford_sets_2[self.sequence_number][self.clifford_number]
        
        print(clifford_gates1)
        print(clifford_gates2)
        
        clifford_gates = {
                'qubit_1': clifford_gates1,
                'qubit_2': clifford_gates2
                }

        rotating_axis = {'Xp': [1,0,0],
                         'X9': [1,0,0],
                         'mXp': [-1,0,0],
                         'mX9': [-1,0,0],
                         'Yp': [0,1,0],
                         'Y9': [0,1,0],
                         'mYp': [0,-1,0],
                         'mY9': [0,-1,0],
                         'I': [1,0,0],
                         'Zp_prep': [1,0,0]
                         }
        
        gate_length = {'Xp': qubit_1.Pi_pulse_length,
                       'X9': qubit_1.halfPi_pulse_length,
                       'mXp': qubit_1.Pi_pulse_length,
                       'mX9': qubit_1.halfPi_pulse_length,
                       'Yp': qubit_1.Pi_pulse_length,
                       'Y9': qubit_1.halfPi_pulse_length,
                       'mYp': qubit_1.Pi_pulse_length,
                       'mY9': qubit_1.halfPi_pulse_length,
                       'Zp': 0,
                       'mZp': 0,
                       'Z9': 0,
                       'mZ9': 0,
#                       'I': 0,
#                       'I': 10e-9,# if not self.align else 10e-9,
                       'I': qubit_1.Pi_pulse_length,
                       'Zp_prep': qubit_1.Pi_pulse_length,
#                       'Zp_prep': 0,
                       'None': 0
                       }

        gate_name_1 = 'prepare_state_1'
        gate_name_2 = 'prepare_state_2'
        
        length_total1 = 0
        length_total2 = 0

        for i in range(len(clifford_gates1)):
            print('go to next clifford : ', i)
            '''
            length_clifford1 = sum([gate_length[gate] for gate in clifford_gates1[i]])
            length_clifford2 = sum([gate_length[gate] for gate in clifford_gates2[i]])
            
            length_total1 += length_clifford1
            length_total2 += length_clifford2
            
            if len(clifford_gates1) != len(clifford_gates2) or length_clifford1 != length_clifford2 or length_total1 != length_total2:
                raise ValueError('clifford length different')
            
            if length_clifford1 != 2.5e-7 and 'I' not in clifford_gates1[i] and 'Zp_prep' not in clifford_gates1[i]:
                
                raise ValueError('clifford length different:', length_clifford1)
            '''
            for qubit in ['qubit_1', 'qubit_2']:
                clifford = clifford_gates[qubit][i]
                
                for j in range(len(clifford)):
                    gate = clifford[j]
                    '''
                    to be optimized about the preparation gate of I and Zp !!!
                    '''
                    if gate == 'I' or gate == 'Zp_prep':
                        
                        if gate_length['I'] == 0 and  gate == 'I':
                            continue
                        
                        amp = self.off_resonance_amplitude if qubit == 'qubit_1' else 0
                        freq_shift = -30e6 if qubit == 'qubit_1' else 0
                        
                        if gate == 'Zp_prep':
                            self.add_Z(name='Z_prep%d%d'%((i+1),(j+1))+qubit, qubit = qubit_obj[qubit], degree = 180)
                            if gate_length['Zp_prep'] == 0:
                                continue
                    
                    elif gate == 'Zp_prep':
                        amp = self.off_resonance_amplitude if qubit == 'qubit_1' else 0
                        freq_shift = -30e6 if qubit == 'qubit_1' else 0
                    
                    elif 'CZ' in gate:
                        last_gate_1 = gate_name_1
                        gate_name = 'CPhase%d%d'%((i+1),(j+1))
                        gate_name_1 = gate_name + '_waiting%d%d'%((i+1),(j+1))
                        gate_name_2 = gate_name + '_waiting%d%d'%((i+1),(j+1))
                        refgate = last_gate_1
                        
                        if gate == 'CZ_dumy':
                            pass
                        
                        elif gate == 'CZ':
                            
#                            gate_name = 'CPhase%d%d'%((i+1),(j+1))
                            
                            self.add_CPhase(name = gate_name, refgate = refgate, refpoint = 'end', waiting_time = te,
                                            control_qubit = qubit_1, target_qubit = qubit_2,
                                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                                            length = detuning_time)
                            
                            self.add_Z(name='Zcp%d%d'%((i+1,j+1)) + 'qubit_1', qubit = qubit_1, degree = self.phase_1)
                            self.add_Z(name='Zcp%d%d'%((i+1,j+1)) + 'qubit_2', qubit = qubit_2, degree = self.phase_2)
                            
#                            gate_name_1 = gate_name + '_waiting%d%d'%((i+1),(j+1))
#                            gate_name_2 = gate_name + '_waiting%d%d'%((i+1),(j+1))
                            
                            self.add_CPhase(name = gate_name_1, refgate = gate_name, refpoint = 'end', 
                                            control_qubit = qubit_1, target_qubit = qubit_2,
                                            amplitude_control = 0, amplitude_target = 0, 
                                            length = te)
                            
                            pass
                        else:
                            raise NameError('CZ gate not found')
                        continue
                    
                    elif 'Z' in gate:
                        if gate == 'Zp' or gate == 'mZp':
                            degree = 180
                        elif gate == 'Z9':
                            degree = 90
                        elif gate == 'mZ9':
                            degree = -90
                        else:
                            raise ValueError('Z gate degree is wrong')
                        self.add_Z(name='Z_%d%d'%((i+1),(j+1))+qubit, qubit = qubit_obj[qubit], degree = degree)
                        continue
                    
                    else:
                        amp = 1
                        freq_shift = 0
                    
                    axis = rotating_axis[gate]
                    length = gate_length[gate]
                    
                    last_gate_1 = gate_name_1
                    last_gate_2 = gate_name_2
                    
                    gate_name = 'C%d%d'%((i+1),(j+1)) + qubit
                    
                    if i+j == 0:
                        
                        if qubit == 'qubit_2':
                            refgate = 'C11'+'qubit_1'
                            refpoint = 'start'
                            refqubit = 'qubit_1'
                        else: 
                            refpoint = 'end'
                            refgate = None
                            refqubit = None
#                    elif j == 0:
#                    else:
#                        refgate = last_gate_1
#                        refpoint = 'start'
                    else:
                        refgate = last_gate_1 if qubit == 'qubit_1' else last_gate_2
                        refqubit = 'qubit_1' if qubit == 'qubit_1' else 'qubit_2'
                        refpoint = 'end'
                        

                    if qubit == 'qubit_1':
                        gate_name_1 = gate_name
                    else:
                        gate_name_2 = gate_name
                        
                    print(gate_name)
                    
                    self.add_single_qubit_gate(name = gate_name, refgate = refgate, refpoint = refpoint, refqubit = refqubit,
                                               qubit = qubit_obj[qubit], axis = axis,
                                               amplitude = amp, length = length, frequency_shift = freq_shift)
            
            '''
            if self.align:
                if length_clifford1 > length_clifford2:
                    self.add_single_qubit_gate(name = 'Null', refgate = last_gate+'qubit_2', refpoint = 'end',
                                               qubit = qubit_2, axis = [1,0,0],
                                               amplitude = 0, 
                                               length = length_clifford1-length_clifford2,)

                elif length_clifford2 > length_clifford1:
                    self.add_single_qubit_gate(name = 'off_resonance', refgate = 'C%d%d'%(i+1,j+1)+clifford_gates1[-1][-1]+'qubit_1', 
                                               refpoint = 'end',
                                               qubit = qubit_1, axis = [1,0,0],
                                               amplitude = self.off_resonance_amplitude, 
                                               length = length_clifford2-length_clifford1, frequency_shift = -30e6)
        '''
        
        if length_total1 != length_total2:
            raise ValueError('Length different')
        
        '''
        if len(clifford_gates1) > 0:
            ii = len(clifford_gates1)
            jj = len(clifford_gates1[-1])
            
            if length_total1 < length_total2 and not self.align:
                self.add_single_qubit_gate(name = 'off_resonance', refgate = 'C%d%d'%(ii,jj)+'qubit_1', 
                                           refpoint = 'end',
                                           qubit = qubit_1, axis = [1,0,0],
                                           amplitude = self.off_resonance_amplitude, 
                                           length = length_total2-length_total1, frequency_shift = -30e6)
        '''
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


                
                                
              