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
from copy import deepcopy
#%% by objects
class Finding_Resonance(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.waiting_time = kw.pop('waiting_time', 0)
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)
        return self

    def make_circuit(self, **kw):
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)

        frequency_shift = -25e6
        sweep_points = 1250
        sweep_range = 50e6
        burst_time = 100e-6
        start_freq = 0
        end_freq = sweep_range + start_freq
        self.add_single_qubit_gate(name = 'off_adiabatic', qubit = qubit, 
                                       amplitude = 1, length = burst_time, frequency_shift = [start_freq, end_freq])
        
        
        '''
        for i in range(int(sweep_points)):
            refgate = None if i==0 else last_gate
            new_gate = 'off_F_'+str(i)
            self.add_single_qubit_gate(name = new_gate, refgate = refgate, qubit = qubit_2, 
                                       amplitude = 1, length = burst_time/sweep_points, frequency_shift = frequency_shift)
            last_gate = new_gate
            frequency_shift += sweep_range/sweep_points
        '''
        return self

class Ramsey(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)
        self.phase_1 = kw.pop('phase_1', 0)
        self.phase_2 = kw.pop('phase_2', 0)
        self.off_resonance = kw.pop('off_resonance', False)
        
        self.echo = kw.pop('echo', False)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.off_resonance = kw.pop('off_resonance', self.off_resonance)
        self.echo = kw.pop('echo', self.echo)
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance = kw.pop('off_resonance', self.off_resonance)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        
        echo = kw.pop('echo', self.echo)
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        length = kw.get('duration_time', qubit.halfPi_pulse_length)

        self.add_X(name='X1_Q1', qubit = qubit,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        self.add_Z(name='Z1_Q1', qubit = qubit, degree = phase_2)
#        self.add_Z(name='Zde2_Q1', qubit = qubit, degree = 360 * 4e6*waiting_time)
        if echo:
            self.add_X(name='De_Q1', qubit = qubit, refgate = 'X1_Q1', waiting_time = waiting_time/2,
                       amplitude = amplitude, length = length*2, frequency_shift = frequency_shift)
        
            self.add_X(name='X2_Q1', refgate = 'De_Q1', qubit = qubit, waiting_time = waiting_time/2,
                       amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        else:
            self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = qubit, waiting_time = waiting_time,
                       amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        if off_resonance:
            self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1', refpoint = 'start', waiting_time = -30e-9,
                                       qubit = qubit_1, amplitude = 1,
                                       length = waiting_time + 2*length + 50e-9, frequency_shift = 0)
        
#        self.add_Y(name = 'Y1_Q2', refgate = 'X2_Q1', qubit = self.qubits[1],)
#
#        self.add_Y(name='Y1_Q1', refgate = 'X2_Q1', qubit = self.qubits[0], waiting_time = 500e-9,)
#
#        self.add_CPhase(name = 'CP_Q12', refgate = 'X2_Q1', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
#                        amplitude_control = 0.2, amplitude_target = -0.6, length = 10e-8, waiting_time = 400e-9)
#
#        self.add_X(name = 'X1_Q2', refgate = 'CP_Q12', qubit = self.qubits[1], waiting_time = 100e-9)
#
#        self.add_Z(name='Z1_Q1', qubit = self.qubits[0], degree = 45)
#        self.add_Z(name = 'Z1_Q2', qubit = self.qubits[1], degree = 45)
#
#        self.add_X(name = 'X2_Q2', refgate = 'X1_Q2', qubit = self.qubits[1], waiting_time = 800e-9)
##
#        self.add_X(name='X3_Q1', refgate = 'Y1_Q1', qubit = self.qubits[0], waiting_time = 250e-9,)
#
#        self.add_single_qubit_gate(name = 'T1_Q1', refgate = 'X3_Q1', qubit = self.qubits[0], amplitude = 0.1, waiting_time = self.waiting_time)

#        self.add_Z(name='Z1_Q1', qubit = self.qubits[0],)

        return self
    
    
    

#%%

class Rabi(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        self.length = kw.pop('duration_time', qubit.Pi_pulse_length)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        length = kw.get('duration_time', self.length)
        
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        print('length', length)

        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = qubit, amplitude = amplitude, #axis = [0,1,0], 
                                   length = length, frequency_shift = frequency_shift)
        
#        self.add_single_qubit_gate(name='Rabi_heating', refgate = 'Rabi_Oscillation', qubit = qubit, amplitude = amplitude, #axis = [0,1,0], 
#                                   length = 3.05e-6-length, frequency_shift = frequency_shift-20e6)

        return self

class CRot(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_1')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 0)
        self.amplitudepi = kw.pop('amplitudepi', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 275e-9)
    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        return self

    def make_circuit(self, **kw):
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        length = kw.get('duration_time', qubit.CRot_pulse_length)
        
        length = kw.get('duration_time', self.length)
        amplitude = kw.get('amplitude', self.amplitude)
        amplitudepi = kw.get('amplitudepi', self.amplitudepi)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)

        self.add_CPhase(name = 'CP_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude, amplitude_target = 0, length = length+150e-9)
        
        self.add_single_qubit_gate(name='CRot', refgate = 'CP_Q12', qubit = self.qubits[1], 
                                   refpoint = 'start', waiting_time = 100e-9, amplitude = amplitudepi, 
                                   length = length, frequency_shift = frequency_shift,)

        return self
    
class Rabi_detuning(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_1')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 275e-9)
    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        return self

    def make_circuit(self, **kw):
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        length = kw.get('duration_time', qubit.Pi_pulse_length)
        
        length = kw.get('duration_time', self.length)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)

        self.add_CPhase(name = 'CP_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude, amplitude_target = 0, length = length+150e-9)
        
        self.add_single_qubit_gate(name='Q1', refgate = 'CP_Q12', qubit = qubit, 
                                   refpoint = 'start', waiting_time = 100e-9, amplitude = 1, 
                                   length = length, frequency_shift = frequency_shift,)


        return self    

class Pi_CRot(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 275e-9)
    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        return self

    def make_circuit(self, qubit = 2, **kw):
        qubit_num = qubit
        
        qubit = self.qubits[int(qubit_num-1)]
        
        length = kw.get('duration_time', self.length)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = qubit, amplitude = amplitude, 
                                   length = length, frequency_shift = frequency_shift)
        
        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = qubit, 
                   waiting_time = waiting_time,
                   amplitude = amplitude, length = length, 
                   frequency_shift = frequency_shift, refphase = phase)

        self.add_CPhase(name = 'CP_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude, amplitude_target = 0, length = length+150e-9)
        
        

        return self

AllXY_array = [
        ['I', 'I'],
        ['Xpi', 'Xpi'], ['Ypi', 'Ypi'],
        ['Xpi', 'Ypi'], ['Ypi', 'Xpi'], 
        ['X', 'I'], ['Y', 'I'],
        ['X', 'Y'], ['Y', 'X'],
        ['X', 'Ypi'], ['Y', 'Xpi'],
        ['Xpi', 'Y'], ['Ypi', 'X'],
        ['X', 'Xpi'], ['Xpi', 'X'],
        ['Y', 'Ypi'], ['Ypi', 'Y'],
        ['Xpi', 'I'], ['Ypi', 'I'],
        ['X', 'X'], ['Y', 'Y'],
        ]
AllXY_array_2 = sorted(AllXY_array*2)

class AllXY(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        g = int(kw.pop('gate', 1)-1)
        gate = AllXY_array[g]
        for i in range(len(gate)):
            amplitude = 0 if gate[i] == 'I' else 1
            axis = [1,0,0] if gate[i].startswith('X') else [0,1,0]
            length = qubit.Pi_pulse_length if gate[i].endswith('pi') else qubit.halfPi_pulse_length
            
            name = 'G%d'%(i+1)
            refgate = None if i == 0 else 'G%d'%i
            
            self.add_single_qubit_gate(name = name, refgate = refgate, 
                                       qubit = qubit, axis = axis,
                                       amplitude = amplitude, length = length, 
                                       frequency_shift = frequency_shift)
            
        return self
    
#%% 
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
        
        qubit = Instrument.find_instrument(qubit_name)
        
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]
        
        print(clifford_gates)
        
        name = 'prepare_state'
        self.add_single_qubit_gate(name = name, qubit = qubit, amplitude = 1, 
                                   length = qubit.Pi_pulse_length,)

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
            
#                refgate = None if i+j == 0 else name
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                self.add_single_qubit_gate(name = name, refgate = refgate, 
                                       qubit = qubit, axis = axis,
                                       amplitude = amplitude, length = length,)
            
        print('clifford_gates finished')
        
        return self


class RBinterleavedCZ(Manipulation):

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
        self.control_qubit = kw.pop('control_qubit', 'qubit_2')
        self.phase_1 = kw.pop('phase_1', 90)
        self.phase_2 = kw.pop('phase_2', 60)        
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0277)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.00)       
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.control = kw.pop('control', 0)
        
    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.clifford_number = kw.pop('clifford_number', self.clifford_number)
        self.sequence_number = kw.pop('sequence_number', self.sequence_number)
        self.control_qubit = kw.pop('control_qubit', self.control_qubit)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.control = kw.pop('control', self.control)
        return self
    

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)

        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        control_qubit = kw.pop('control_qubit', self.control_qubit)
        target_qubit = 'qubit_1' if control_qubit == 'qubit_2' else 'qubit_2'
        C = int(control_qubit[-1])-1
        T = int(target_qubit[-1])-1
        
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)    
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)    
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        control =  kw.pop('control', self.control)
        self.sequence_number = int(kw.pop('sequence_number', self.sequence_number))
        self.clifford_number = int(kw.pop('clifford_number', self.clifford_number))
        
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]
        
        print(clifford_gates)
        
        name = 'prepare_state'
         
        self.add_single_qubit_gate(name = name, qubit = self.qubits[C], amplitude = control, 
                                   length = qubit_1.Pi_pulse_length,)
#        
#        self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = name, refpoint = 'start',
#                                       qubit = self.qubits[T], amplitude = 1.2, 
#                                       length = self.qubits[0].Pi_pulse_length, frequency_shift =-30e6)
            
        
        refgate = name
        
        for i in range(len(clifford_gates)):
            print('go to next clifford : ', i)
            for j in range(len(clifford_gates[i])):
                gate = clifford_gates[i][j]
                if gate.startswith('Z'):
                    name1 = 'C1%d%d'%((i+1),(j+1))+gate
                    name2 = 'C2%d%d'%((i+1),(j+1))+gate
                    name3 = 'C3%d%d'%((i+1),(j+1))+gate
                    self.add_CPhase(name = name1, refgate = refgate,
                                control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                amplitude_control = 0, amplitude_target = 0, 
                                length = 10e-9)
                
                    self.add_CPhase(name = name2, refgate = name1,
                                control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                                length = detuning_time)
                
                    self.add_CPhase(name = name3, refgate = name2,
                                control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                amplitude_control = 0, amplitude_target = 0, 
                                length = 10e-9)
                    
                    self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
                    self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
                    
                    refgate = deepcopy(name3)
                    
                else:
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
    #                refgate = None if i+j == 0 else name

                    name = 'C%d%d'%((i+1),(j+1))+gate
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                           qubit = self.qubits[T], axis = axis,
                                           amplitude = amplitude, length = length,)
                    refgate = deepcopy(name)         
                    
        print('clifford_gates finished')
        
        return self



class CPhase_Calibrate(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*-0.00)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.15)
        self.control_qubit = kw.pop('control_qubit', 'qubit_2')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.control_qubit = kw.pop('control_qubit', self.control_qubit)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        control_qubit = kw.pop('control_qubit', self.control_qubit)
        target_qubit = 'qubit_1' if control_qubit == 'qubit_2' else 'qubit_2'
        C = int(control_qubit[-1])-1
        T = int(target_qubit[-1])-1
        
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        
        
        
        
        self.add_single_qubit_gate(name='X_Pi_Q2', qubit = self.qubits[C], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        if control_qubit == 'qubit_2':
        
            self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X_Pi_Q2', refpoint = 'start',
                                       qubit = self.qubits[T], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].Pi_pulse_length, frequency_shift = frequency_shift-30e6)
        
        self.add_X(name='X1_Q2', refgate = 'X_Pi_Q2', qubit = self.qubits[T],  
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        if target_qubit  == 'qubit_2':
        
            self.add_X(name='off_resonance2_Q2', refgate = 'X1_Q2', refpoint = 'start', qubit = self.qubits[C],  
                   amplitude = off_resonance_amplitude, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift-30e6,)
        

        
        
        self.add_CPhase(name = 'CP_wait1', refgate = 'X1_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = 10e-9)
        
        self.add_CPhase(name = 'CP_Q12', refgate = 'CP_wait1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time)
        
        self.add_CPhase(name = 'CP_wait2', refgate = 'CP_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = 10e-9)
    
        
        self.add_Z(name='Z1_Q1', qubit = self.qubits[T], degree = phase)
        
        self.add_X(name='X2_Q2', refgate = 'CP_wait2', 
                   waiting_time = 0, qubit = self.qubits[T], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        if target_qubit  == 'qubit_2':
            self.add_single_qubit_gate(name='off_resonance3_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                       qubit = self.qubits[C], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)


        return self


class MultiCPhase_Calibrate(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*-0.00)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.15)
        self.control_qubit = kw.pop('control_qubit', 'qubit_2')
        self.cphase_number =  kw.pop('cphase_number',1)
        self.phase_1 = kw.pop('phase_1', 0)
        self.phase_2 = kw.pop('phase_2', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.control_qubit = kw.pop('control_qubit', self.control_qubit)
        self.cphase_number =  kw.pop('cphase_number',self.cphase_number)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        cphase_number = kw.pop('cphase_number',self.cphase_number)
        control_qubit = kw.pop('control_qubit', self.control_qubit)
        target_qubit = 'qubit_1' if control_qubit == 'qubit_2' else 'qubit_2'
        C = int(control_qubit[-1])-1
        T = int(target_qubit[-1])-1
        phase1 = kw.pop('phase_1', self.phase_1)
        phase2 = kw.pop('phase_2', self.phase_2)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        
        
        
        
        self.add_single_qubit_gate(name='X_Pi_Q2', qubit = self.qubits[C], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        if control_qubit == 'qubit_2':
        
            self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X_Pi_Q2', refpoint = 'start',
                                       qubit = self.qubits[T], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].Pi_pulse_length, frequency_shift = frequency_shift-30e6)
        
        self.add_X(name='X1_Q2', refgate = 'X_Pi_Q2', qubit = self.qubits[T],  
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        if target_qubit  == 'qubit_2':
        
            self.add_X(name='off_resonance2_Q2', refgate = 'X1_Q2', refpoint = 'start', qubit = self.qubits[C],  
                   amplitude = off_resonance_amplitude, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift-30e6,)
        
        refgatenow = 'X1_Q2'

        for i in range(0,cphase_number):

            name1 = 'CP_wait1%d'%(i)
            name2 = 'CP_Q12%d'%(i)
            name3 = 'CP_wait2%d'%(i)
            
            
            self.add_CPhase(name = name1, refgate = refgatenow,
                            control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                            amplitude_control = 0, amplitude_target = 0, 
                            length = 10e-9)
            
            self.add_CPhase(name = name2, refgate = name1,
                            control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                            amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                            length = detuning_time)
            
            self.add_CPhase(name = name3, refgate = name2,
                            control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                            amplitude_control = 0, amplitude_target = 0, 
                            length = 10e-9)
            
            self.add_Z(name='Z2_Q2', qubit = self.qubits[1], degree = phase2)
            self.add_Z(name='Z2_Q1', qubit = self.qubits[0], degree = phase1)
            
            refgatenow = deepcopy(name3)
        
        
        
        self.add_Z(name='Z1_Q1', qubit = self.qubits[T], degree = phase)
        
        self.add_X(name='X2_Q2', refgate = refgatenow, 
                   waiting_time = 0, qubit = self.qubits[T], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        if target_qubit  == 'qubit_2':
            self.add_single_qubit_gate(name='off_resonance3_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                       qubit = self.qubits[C], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)


        return self


class MeasureTminus(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.000)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*-0.000)
        
        self.detuning = kw.pop('detuning', 30*0.5*-0.030)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', self.detuning_amplitude2)
        self.detuning = kw.pop('detuning', self.detuning)
    
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        detuning = kw.get('detuning', self.detuning)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        self.add_X(name='X_Pi_Q1', qubit = self.qubits[0], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_X(name='X1_Q2', refgate = 'X_Pi_Q1', refpoint = 'start', qubit = self.qubits[1], 
                   amplitude = Pi_amplitude, length = self.qubits[1].Pi_pulse_length, 
                   frequency_shift = 0)

        self.add_CPhase(name = 'CP1', refgate = 'X1_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = 30e-9)        

        self.add_CPhase(name = 'CP2', refgate = 'CP1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude + detuning, amplitude_target = detuning_amplitude2 -0*detuning, 
                        length = detuning_time)
        


        return self

class Hahn(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
#        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.0272)
#        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*0.02)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.wait_time = kw.pop('wait_time', 80e-9)
#        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.target_qubit = kw.pop('target_qubit', 'qubit_1')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
#        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
#        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', self.detuning_amplitude2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.wait_time = kw.pop('wait_time', self.wait_time)
#        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.target_qubit = kw.pop('target_qubit', self.target_qubit)
        return self

    def make_circuit(self, **kw):
        
#        phase = kw.pop('phase', self.phase)
        wait_time = kw.pop('wait_time', self.wait_time)
#        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
#        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        target_qubit = kw.pop('target_qubit', self.target_qubit)
        control_qubit = 'qubit_1' if target_qubit == 'qubit_2' else 'qubit_2'
        C = int(control_qubit[-1])-1
        T = int(target_qubit[-1])-1
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        self.qubits[0] = qubit_1
        self.qubits[1] = qubit_2
        
        # Pi/2 on target
        self.add_X(name='X1_T', qubit = self.qubits[T],  
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        # Wait for time wait_time/2
        self.add_CPhase(name = 'wait1', refgate = 'X1_T',
                        control_qubit = self.qubits[C], target_qubit = self.qubits[T],
                        amplitude_control = 0, amplitude_target = 0,
                        length = wait_time/2)
        
        # Decoulping Pi on target
        self.add_single_qubit_gate(name='X_Pi_decouple_T', refgate = 'wait1',
                               qubit = self.qubits[T], amplitude = 1, 
                               length = self.qubits[T].Pi_pulse_length, frequency_shift = 0)
        
        # Wait for time wait_time/2
        self.add_CPhase(name = 'wait2', refgate = 'X_Pi_decouple_T',
                        control_qubit = self.qubits[C], target_qubit = self.qubits[T],
                        amplitude_control = 0, amplitude_target = 0,
                        length = wait_time/2)
        
         # Pi/2 on target
        self.add_X(name='X2_T', refgate = 'wait2', qubit = self.qubits[T],  
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)

        return self

class DCZ(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.0272)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*0.02)
        
        self.detuning_amplitude3 = kw.pop('detuning_amplitude3', self.detuning_amplitude)
        self.detuning_amplitude4 = kw.pop('detuning_amplitude4', self.detuning_amplitude2)
        
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.control_qubit = kw.pop('control_qubit', 'qubit_2')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', self.detuning_amplitude2)
        self.detuning_amplitude3 = kw.pop('detuning_amplitude3', self.detuning_amplitude3)
        self.detuning_amplitude4 = kw.pop('detuning_amplitude4', self.detuning_amplitude4)
        
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.control_qubit = kw.pop('control_qubit', self.control_qubit)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        
        detuning_amplitude3 = kw.pop('detuning_amplitude3', self.detuning_amplitude3)
        detuning_amplitude4 = kw.pop('detuning_amplitude4', self.detuning_amplitude4)
        
        
        detuning_amplitude2 = 30*0.5*0.02
        detuning_amplitude4 = 30*0.5*-0.02
        detuning_amplitude3 = detuning_amplitude + 30*0.5*0.083
        
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        control_qubit = kw.pop('control_qubit', self.control_qubit)
        target_qubit = 'qubit_1' if control_qubit == 'qubit_2' else 'qubit_2'
        C = int(control_qubit[-1])-1
        T = int(target_qubit[-1])-1
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        self.qubits[0] = qubit_1
        self.qubits[1] = qubit_2
        
        te = 10e-9
#        off_resonance_amplitude = 1.2
        
        '''
        if control_qubit == 'qubit_2':
            # first pi pulse
            
            if Pi_amplitude != 0:
            
                self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[1], amplitude = Pi_amplitude, 
                                           length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
                # pi/2 pulse
                self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'XPi_Q1', refpoint = 'start',
                                           qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                           length = self.qubits[0].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)
                
                self.add_X(name='X1_Q2',refgate ='off_resonance1_Q1', qubit = self.qubits[0], 
                           amplitude = 1, length = self.qubits[0].halfPi_pulse_length,) 
            else:
                self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[1], amplitude = Pi_amplitude, 
                                           length = self.qubits[0].halfPi_pulse_length, frequency_shift = 0)
                self.add_X(name='X1_Q2',refgate ='XPi_Q1', refpoint = 'start', qubit = self.qubits[0], 
                           amplitude = 1, length = self.qubits[0].halfPi_pulse_length,) 
                
            
        elif target_qubit == 'qubit_2':
            if Pi_amplitude != 0:
                self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[0], amplitude = Pi_amplitude, 
                                           length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
                
                self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'XPi_Q1', refpoint = 'start',
                                           qubit = self.qubits[1], amplitude = 0, 
                                           length = self.qubits[0].halfPi_pulse_length, frequency_shift = 0)
                
                self.add_X(name='X1_Q2',refgate ='off_resonance1_Q1', qubit = self.qubits[1], 
                           amplitude = 1, length = self.qubits[0].halfPi_pulse_length, )
            
            else:
                self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                           length = self.qubits[0].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)
            
                self.add_X(name='X1_Q2',refgate ='XPi_Q1', refpoint = 'start', qubit = self.qubits[1], 
                           amplitude = 1, length = self.qubits[0].halfPi_pulse_length, )
        
        '''
        self.add_single_qubit_gate(name='X_Pi_Q2', qubit = self.qubits[C], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        if control_qubit == 'qubit_2':
        
            self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X_Pi_Q2', refpoint = 'start',
                                       qubit = self.qubits[T], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].Pi_pulse_length, frequency_shift = frequency_shift-30e6)
        
        self.add_X(name='X1_Q2', refgate = 'X_Pi_Q2', qubit = self.qubits[T],  
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        if target_qubit  == 'qubit_2':
        
            self.add_X(name='off_resonance2_Q2', refgate = 'X1_Q2', refpoint = 'start', qubit = self.qubits[C],  
                   amplitude = off_resonance_amplitude, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift-30e6,)
        

        self.add_CPhase(name = 'wait1', refgate = 'X1_Q2',#''XPi_Q1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        

        self.add_CPhase(name = 'CP1_Q12', refgate = 'wait1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/4)
        
        self.add_CPhase(name = 'rCP1_Q12', refgate = 'CP1_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude3, amplitude_target = detuning_amplitude4, 
                        length = detuning_time/4)
        
        self.add_CPhase(name = 'wait2', refgate = 'rCP1_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        '''
        '''
#        if control_qubit == 'qubit_2':
    
        self.add_single_qubit_gate(name='XPi0_decouple', refgate = 'wait2',
                               qubit = self.qubits[0], amplitude = 1, 
                               length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
#        elif control_qubit == 'qubit_1':
        self.add_single_qubit_gate(name='XPi_decouple', refgate = 'XPi0_decouple', refpoint = 'start',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].Pi_pulse_length, frequency_shift = 0)
            
#            self.add_single_qubit_gate(name='off_resonance_pi_Q1', refgate = 'XPi_decouple', refpoint = 'start',
#                                           qubit = self.qubits[1], amplitude = 0, 
#                                           length = self.qubits[1].halfPi_pulse_length, frequency_shift = 0)
        '''
        '''
        self.add_CPhase(name = 'wait3', refgate = 'XPi_decouple',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'wait3',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/4)
        
        self.add_CPhase(name = 'rCP2_Q12', refgate = 'CP2_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude3, amplitude_target = detuning_amplitude4, 
                        length = detuning_time/4)
        
        self.add_CPhase(name = 'wait4', refgate = 'rCP2_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        '''
        '''
        
        
        if control_qubit == 'qubit_2':
            self.add_Z(name='Z2_Q2', qubit = self.qubits[0], degree = phase)
            self.add_X(name='X2_Q2', refgate = 'wait4', qubit = self.qubits[0], 
                       amplitude = 1, length = self.qubits[0].halfPi_pulse_length,) 
#                       frequency_shift = frequency_shift,)
        
        elif target_qubit == 'qubit_2':    
            self.add_Z(name='Z2_Q2', qubit = self.qubits[1], degree = phase)
            self.add_X(name='X2_Q2', refgate = 'wait4', qubit = self.qubits[1], 
                       amplitude = 1, length = self.qubits[0].halfPi_pulse_length,) 
#                       frequency_shift = frequency_shift,)
        
            self.add_single_qubit_gate(name='off_resonance2_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                       qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                       length = self.qubits[0].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)

        return self
    


class Sychpulses1(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].halfPi_pulse_length, frequency_shift = 0)
        
        self.add_single_qubit_gate(name='XPi_Q2', refgate = 'XPi_Q1', refpoint = 'start',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = 0)


        self.add_CPhase(name = 'wait1', refgate = 'XPi_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = detuning_time)
        

        self.add_CPhase(name = 'CP1_Q12', refgate = 'wait1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = 200e-9)

        return self


class Sychpulses2(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        self.add_CPhase(name = 'CP1_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = 200e-9)
        
        
        self.add_CPhase(name = 'wait1', refgate = 'CP1_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = detuning_time)
        
        self.add_single_qubit_gate(name='XPi_Q1', refgate = 'wait1', qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].halfPi_pulse_length, frequency_shift = 0)
        
        self.add_single_qubit_gate(name='XPi_Q2', refgate = 'wait1',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = 0)


        return self




    
class Charge_Noise(Manipulation):
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        self.add_X(name='X1_Q1', qubit = self.qubits[0], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)

        self.add_CPhase(name = 'CP1_Q12', refgate = 'X1_Q1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = detuning_time/2)
        
        self.add_single_qubit_gate(name='XPi2_Q1', refgate = 'X1_Q2',
                                   qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_single_qubit_gate(name='XPi2_Q2', refgate = 'X1_Q2',
                                   qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'XPi2_Q1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = detuning_time/2)
        
        self.add_Z(name='Z1_Q1', qubit = self.qubits[1], degree = phase)
        
        self.add_X(name='X2_Q1', refgate = 'CP2_Q12', qubit = self.qubits[0], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)

        return self
    
#%%

class Ramsey_00_11_basis(Manipulation):    

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.detuning_amplitude = kw.pop('detuning_amplitude', 30*0.5*-0.030)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', 30*0.5*-0.000)
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 0)
        self.wait_time = kw.pop('wait_time', 0)
        self.phase1 = kw.pop('phase1', 0)
        self.phase2 = kw.pop('phase2', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.detuning_amplitude = kw.pop('detuning_amplitude', self.detuning_amplitude)
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', self.detuning_amplitude2)
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.wait_time = kw.pop('wait_time', self.wait_time)
        self.phase1 = kw.pop('phase1', self.phase1)
        self.phase2 = kw.pop('phase2', self.phase2)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        return self

    def make_circuit(self, **kw):
        
        phase1 = kw.pop('phase1', self.phase1)
        phase2 = kw.pop('phase2', self.phase2)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        wait_time = kw.pop('wait_time', self.wait_time)
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        Pi_amplitude = kw.get('Pi_amplitude', self.Pi_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        te = 30e-9


        self.add_Y(name='XPi_Q1',qubit = self.qubits[0], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, frequency_shift = 0,)
        
        self.add_Y(name='X1_Q2',refgate ='XPi_Q1', refpoint = 'start',qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, frequency_shift = 0,)
        

##DCZ
        self.add_CPhase(name = 'wait1', refgate = 'X1_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        

        self.add_CPhase(name = 'CP1_Q12', refgate = 'wait1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'wait2', refgate = 'CP1_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_single_qubit_gate(name='XPi2_Q1', refgate = 'wait2',
                                   qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_single_qubit_gate(name='XPi2_Q2', refgate = 'wait2',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].Pi_pulse_length, frequency_shift = 0)
        
        self.add_CPhase(name = 'wait3', refgate = 'XPi2_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'wait3',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'wait4', refgate = 'CP2_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_Z(name='Z2_Q2', qubit = self.qubits[1], degree = phase2)
        self.add_Z(name='Z2_Q1', qubit = self.qubits[0], degree = phase1)
##        
        
        self.add_Y(name='X2_Q2', refgate = 'wait4', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        self.add_single_qubit_gate(name='off_resonance2_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)
        

        self.add_CPhase(name = 'wait44', refgate = 'X2_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = wait_time)

        
#
        self.add_Y(name='X3_Q2',refgate ='wait44', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, frequency_shift = 0,)
        
        self.add_single_qubit_gate(name='off_resonance3_Q1', refgate = 'X3_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)


        self.add_CPhase(name = 'wait5', refgate = 'X3_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        

        self.add_CPhase(name = 'CP3_Q12', refgate = 'wait5',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'wait6', refgate = 'CP3_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_single_qubit_gate(name='XPi4_Q1', refgate = 'wait6',
                                   qubit = self.qubits[0], amplitude = 1, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='XPi4_Q2', refgate = 'wait6',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].Pi_pulse_length, frequency_shift = 0)
        
        self.add_CPhase(name = 'wait7', refgate = 'XPi4_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_CPhase(name = 'CP4_Q12', refgate = 'wait7',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'wait8', refgate = 'CP4_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)
        
        self.add_Z(name='Z4_Q2', qubit = self.qubits[1], degree = phase2)
        self.add_Z(name='Z4_Q1', qubit = self.qubits[0], degree = phase1)
        
        
        self.add_Y(name='X4_Q2', refgate = 'wait8', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = 0,)
        
                
        self.add_Y(name='X4_Q1', refgate = 'X4_Q2',refpoint = 'start', qubit = self.qubits[0], 
                   amplitude = 1, length = self.qubits[0].halfPi_pulse_length, 
                   frequency_shift = 0,)
        
        

        return self

class Rabi_all(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 250e-9)
        self.T_amplitude = kw.get('T_amplitude', 0)
        self.phase_1 = kw.pop('phase_1', 0)
        self.phase_2 = kw.pop('phase_2', 0)
        self.third_tone = kw.pop('third_tone', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        self.T_amplitude = kw.get('T_amplitude', self.T_amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.third_tone = kw.pop('third_tone', self.third_tone)
        return self

    def make_circuit(self, **kw):
        
        amplitude = kw.get('amplitude', self.amplitude)
        T_amplitude = kw.get('T_amplitude', self.T_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        length = kw.get('duration_time', self.length)
        third_tone = kw.pop('third_tone', self.third_tone)
#        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = qubit_2, amplitude = amplitude, 
#                                   length = 250e-9, frequency_shift = 0)

        self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_X(name='X1_Q1', qubit = qubit_1, #refgate = 'Rabi_Oscillation',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
            
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        if third_tone:
            self.add_Y(name='X2_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                       amplitude = 0.4, length = length, frequency_shift = 0)

#        self.add_CPhase(name = 'CP_Q12', control_qubit = qubit_1, target_qubit = qubit_2,
#                        refgate = 'X1_Q2', refpoint = 'start', waiting_time = -100e-9,
#                        amplitude_control = 0, amplitude_target = T_amplitude, length = length+150e-9)

        return self


from RB_test import convert_clifford_to_sequence, clifford_sets

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

        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]
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
                    
#                refgate = None if i+j == 0 else name
                refgate = name
                name = 'C%d%d'%((i+1),(j+1))+gate
                
                amplitude_1 = amplitude if gate != 'I' else 1.2
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

#from RB_two_qubit import convert_clifford_to_sequence, clifford_sets
#
class RB_Marcus(Manipulation):

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
#        clifford_index = list((np.random.rand(self.clifford_number)*24).astype(int))

#        clifford_gates = convert_clifford_to_sequence(clifford_index)

        clifford_gates = clifford_sets[self.sequence_number][self.clifford_number]
#        clifford_gates2 = clifford_sets2[self.sequence_number][self.clifford_number]
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
                
                if 'Z' not in gate:
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
                    
                    amplitude_1 = 1 if gate != 'I' else 1.2
                    freq_shift = 0 if gate!='I' else -30e6
                    amplitude_2 = 1 if gate != 'I' else 0
                    name = name if gate != 'I' else 'off_resonance0'+name

            
#                    refgate = None if i+j == 0 else name
                    self.add_single_qubit_gate(name = name, refgate = refgate, 
                                               qubit = qubit_1, axis = axis,
                                               amplitude = amplitude_1, length = length, frequency_shift = freq_shift)


                    self.add_single_qubit_gate(name = name+'Q2', refgate = refgate,
                                               qubit = qubit_2, axis = axis,
                                               amplitude = amplitude_2, length = length,)
#                    
#                    self.add_single_qubit_gate(name = 'off_resonance2'+name, refgate = name,
#                                               qubit = qubit_1, amplitude = 1.2, 
#                                               length = length, frequency_shift = -30e6)
#                    name = name +'Q2'
#                    if (i+1) != len(clifford_gates) and clifford_gates[i+1][0] == 'Zp' and (j+1) == len(clifford_gates[i]):
#                        self.add_single_qubit_gate(name = 'X1'+name, refgate = name,
#                                                   qubit = qubit_2, amplitude = 1, 
#                                                   length = qubit_2.Pi_pulse_length, )
#                        self.add_single_qubit_gate(name = 'off_resonance3'+name, refgate = name,
#                                                   qubit = qubit_1, amplitude = 1.2, 
#                                                   length = qubit_1.Pi_pulse_length, frequency_shift = -30e6)
#                        name = 'X1'+name

#                    
                elif 'Z' in gate:
                    
                    self.add_CPhase(name = name+'1', refgate = refgate, waiting_time = 10e-9,
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 30*0.5*-0.0270, amplitude_target = 30*0.5*0.04, 
                                    length = self.detuning_time)
                    self.add_Z(name='Z1_Q1', qubit = self.qubits[0], degree = self.phase_1)
                    self.add_Z(name='Z1_Q2', qubit = self.qubits[1], degree = self.phase_2)
#                    self.add_single_qubit_gate(name=name+'Pi_Q1', refgate = name+'1', waiting_time = 20e-9,
#                                               qubit = self.qubits[0], amplitude = 1, 
#                                               length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
#        
#                    self.add_single_qubit_gate(name=name+'Pi_Q2', refgate = name+'1', waiting_time = 20e-9,
#                                               qubit = self.qubits[1], amplitude = 1, 
#                                               length = self.qubits[1].Pi_pulse_length, frequency_shift = 0)
#                    
#                    self.add_CPhase(name = name+'2', refgate = name+'Pi_Q2', waiting_time = 20e-9,
#                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
#                                    amplitude_control = 30*0.5*-0.0273, amplitude_target = 30*0.5*0.01, 
#                                    length = self.detuning_time/2)
                    
                    self.add_CPhase(name = name+'2', refgate = name+'1', 
                                    control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                                    amplitude_control = 0, amplitude_target = 0, 
                                    length = 10e-9)
                    name = name+'2'
#                    if gate == 'Zp':
#                        self.add_single_qubit_gate(name = 'X2'+name, refgate = name, waiting_time = 10e-9,
#                                                   qubit = qubit_2, amplitude = 1, 
#                                                   length = qubit_2.Pi_pulse_length, )
#                        self.add_single_qubit_gate(name = 'off_resonance4'+name, refgate = name, waiting_time = 10e-9,
#                                                   qubit = qubit_1, amplitude = 1.2, 
#                                                   length = qubit_1.Pi_pulse_length, frequency_shift = -30e6)
#                        name = 'X2'+name
                else:
                    raise NameError('Gate name not correct')
                    
#        self.add_single_qubit_gate(name = 'X2_F', refgate = name, qubit = qubit_2, amplitude = self.Pi_amplitude, 
#                                   length = qubit_2.Pi_pulse_length,)
#        self.add_single_qubit_gate(name = 'X2_F2', refgate = 'X2_F', qubit = qubit_1, amplitude = self.Pi_amplitude, 
#                                   length = qubit_2.Pi_pulse_length,)
#        self.add_single_qubit_gate(name = 'off_resonance2', refgate = 'X2_F', refpoint = 'start',
#                                   qubit = qubit_1, amplitude = 1.25, 
#                                   length = qubit_1.Pi_pulse_length, frequency_shift = -30e6)
                    
        print('clifford_gates finished')
        
        return self

class Wait(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 250e-9)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        return self

    def make_circuit(self, **kw):

        self.add_CPhase(name = 'wait4',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length =300e-9)

        
        return self



class Ramsey_all(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)
        self.detune_q1 = kw.pop('detune_q1', False)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        self.detune_q1 = kw.pop('detune_q1', self.detune_q1)
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        length = kw.get('duration_time', self.length)
        detune_q1 = kw.pop('detune_q1', self.detune_q1)
        
        self.add_X(name='X1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
                
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = -30e6)
        
        if detune_q1:
            self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = 360 * 4e6*waiting_time)
            self.add_Z(name='Zde2_Q2', qubit = qubit_2, degree = 360 * 4e6*waiting_time)

        self.add_X(name='X2_Q1', refgate = 'off_resonance_Q1', 
                   qubit = qubit_1, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        

        self.add_X(name='X2_Q2', refgate = 'X2_Q1', refpoint = 'start',
                   qubit = qubit_2, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        return self

class Ramsey_withnoise(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)
        self.detune_q1 = kw.pop('detune_q1', False)
        
        self.sigma1 = kw.pop('sigma1', 0)
        self.sigma2 = kw.pop('sigma2', 0)
        
        self.sigma3 = kw.pop('sigma3', 0)
        self.sigma4 = kw.pop('sigma4', 0)

        self.dummy = kw.pop('dummy', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        self.detune_q1 = kw.pop('detune_q1', self.detune_q1)
        
        self.sigma1 = kw.pop('sigma1', self.sigma1)
        self.sigma2 = kw.pop('sigma2', self.sigma2)
        self.sigma3 = kw.pop('sigma3', self.sigma3)
        self.sigma4 = kw.pop('sigma4', self.sigma4)
        self.dummy = kw.pop('dummy', 0)
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        length = kw.get('duration_time', self.length)
        detune_q1 = kw.pop('detune_q1', self.detune_q1)
        
        sigma1 = kw.pop('sigma1', self.sigma1)
        sigma2 = kw.pop('sigma2', self.sigma2)
        sigma3 = kw.pop('sigma3', self.sigma3)
        sigma4 = kw.pop('sigma4', self.sigma4)
        
        s1 = np.random.normal(0, sigma1, 1)
        s2 = np.random.normal(0, sigma2, 1)
        s3 = np.random.normal(0, sigma3, 1)
        s4 = np.random.normal(0, sigma4, 1)
        
        dummy = kw.pop('dummy',self.dummy)
        self.add_X(name='X1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
                
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = -30e6)


        self.add_CPhase(name = 'CP_Q12', refgate = 'X1_Q1', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = s1[0], amplitude_target = s2[0], length = waiting_time)
        

        self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = 360 * (4e6 +s3[0] )*waiting_time)
        self.add_Z(name='Zde2_Q2', qubit = qubit_2, degree = 360 * (4e6 +s4[0] )*waiting_time)

        self.add_X(name='X2_Q1', refgate = 'off_resonance_Q1', 
                   qubit = qubit_1, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        

        self.add_X(name='X2_Q2', refgate = 'X2_Q1', refpoint = 'start',
                   qubit = qubit_2, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        return self

class Ramsey_test(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.length = kw.pop('duration_time', self.length)
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        length = kw.get('duration_time', qubit_1.halfPi_pulse_length)

        self.add_X(name='X0_Q1', qubit = qubit_1,
                   amplitude = 1, length = waiting_time, frequency_shift = 30e6)

        self.add_X(name='X0_Q2', qubit = qubit_2, refgate = 'X0_Q1', refpoint = 'start',
                   amplitude = 1, length = waiting_time, frequency_shift = 30e6)

                
        
        self.add_X(name='X1_Q1',refgate = 'X0_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = 300e-9, frequency_shift = -30e6)

        self.add_X(name='X2_Q1', refgate = 'off_resonance_Q1', 
                   qubit = qubit_1, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)

        self.add_X(name='X2_Q2', refgate = 'X2_Q1', refpoint = 'start',
                   qubit = qubit_2, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        return self



class AllXY_all(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        return self

    def make_circuit(self, **kw):
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        g = int(kw.pop('gate', 1)-1)
        gate = AllXY_array[g]
        for i in range(len(gate)):
            amplitude = 0 if gate[i] == 'I' else 1
            axis = [1,0,0] if gate[i].startswith('X') else [0,1,0]
            length = qubit_2.Pi_pulse_length if gate[i].endswith('pi') else qubit_2.halfPi_pulse_length
            
            name = 'G%d'%(i+1)
            refgate = None if i == 0 else 'G%d_Q2'%i
            
            self.add_single_qubit_gate(name = name+'_Q2', refgate = refgate, 
                                       qubit = qubit_2, axis = axis,
                                       amplitude = amplitude, length = length, 
                                       frequency_shift = frequency_shift)
            self.add_single_qubit_gate(name = name+'_Q1', refgate = name+'_Q2', refpoint = 'start', 
                                       qubit = qubit_1, axis = axis,
                                       amplitude = amplitude, length = length, 
                                       frequency_shift = frequency_shift)
            
        return self


class Charge_Noise_Bob(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 116)
        self.phase_2 = kw.pop('phase_2', 175)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0278)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.03)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        self.add_Y(name='H1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='H1_Q2', qubit = qubit_2,
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_CPhase(name = 'CP1', refgate = 'H1_Q1', waiting_time = 10e-9,
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = self.detuning_time)
        
        self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_X_Pi(name='H2X_Q2', qubit = qubit_2, refgate = 'CP1', waiting_time = 10e-9,
                      amplitude = amplitude, length = qubit_2.Pi_pulse_length)
        self.add_Y(name='H2Y_Q2', qubit = qubit_2, refgate = 'H2X_Q2',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = 180)
        self.add_Z(name='Z2_Q2', qubit = qubit_2, degree = 180)
        
        self.add_X_Pi(name='H2X_Q1', qubit = qubit_2, refgate = 'H2X_Q2', refpoint = 'start',
                      amplitude = amplitude, length = qubit_1.Pi_pulse_length)
        self.add_Y(name='H2Y_Q1', qubit = qubit_1, refgate = 'H2X_Q1',
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = 90)
        
        '''
        self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = -30e6)

        self.add_X_Pi(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1',
                      amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        '''
        
        self.add_X(name='Re_X_Q1', qubit = qubit_1, refgate = 'H2Y_Q1', waiting_time = waiting_time,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_CPhase(name = 'CP2', refgate = 'Re_X_Q1', waiting_time = 10e-9,
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = self.detuning_time)
        
        self.add_Y(name='Re_Y_Q1', qubit = qubit_1, refgate = 'CP2', waiting_time = 10e-9,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_Z(name='Re_Z_Q1', qubit = qubit_2, degree = 180)
        self.add_Y(name='Re_Y_Q2', qubit = qubit_2, refgate = 'Re_Y_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)

        return self


class Charge_Noise_Bob2(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 100e-9)
        self.detuning_time = kw.pop('detuning_time', 60e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 62)
        self.phase_2 = kw.pop('phase_2', 10)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0275)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.02)
        self.DFS = kw.pop('DFS', 0)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.DFS = kw.pop('DFS', self.DFS)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        return self

    def make_circuit(self, **kw):
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        DFS = kw.pop('DFS', self.DFS)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        amplitude = 1
        
        self.add_Y(name='Y1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='Y1_Q2', qubit = qubit_2, refgate = 'Y1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_CPhase(name = 'CP1', refgate = 'Y1_Q1', waiting_time = 10e-9,
                        control_qubit = qubit_1, target_qubit = qubit_2,
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = detuning_time)
        
        self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
        self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        
        if DFS == 1:
            self.add_Z(name='Z1dfs_Q1', qubit = qubit_1, degree = 180)
            self.add_Z(name='Z1dfs_Q2', qubit = qubit_2, degree = 180)
        
        if decoupled_qubit == 'qubit_1':
            self.add_X(name='X1_Q1', qubit = qubit_1, refgate = 'CP1', waiting_time = 10e-9,
                       amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
            
#            self.add_Z(name='Ztry_Q1', qubit = qubit_1, degree = 90)
        
        elif decoupled_qubit == 'qubit_2':
            self.add_X(name='X1_Q1', qubit = qubit_1, refgate = 'CP1', waiting_time = 10e-9,
                       amplitude = off_resonance_amplitude, length = qubit_1.halfPi_pulse_length, frequency_shift = -30e6)
            self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'CP1', waiting_time = 10e-9,
                       amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
            
#            self.add_Z(name='Ztry_Q2', qubit = qubit_2, degree = 90)
        else:
            raise NameError('decoupled qubit not found')
        
        '''
        above is to prepare state to |00> + |11>
        '''
        
        self.add_single_qubit_gate(name='off_resonance', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = -30e6)
        
        if add_dephase:
            dephase = int(np.random.rand()*2//1)
            if dephase:
                self.add_Z(name='Zde_Q1', qubit = qubit_1, degree = 180)
                self.add_Z(name='Zde_Q2', qubit = qubit_2, degree = 180)
            
        '''
        above is for waiting
        '''
        if decoupled_qubit == 'qubit_2':
            self.add_X(name='X2_Q1', qubit = qubit_1, refgate = 'off_resonance', #waiting_time = 10e-9,
                       amplitude = off_resonance_amplitude, length = qubit_1.halfPi_pulse_length, frequency_shift = -30e6)
            self.add_X(name='X2_Q2', qubit = qubit_2, refgate = 'off_resonance', #waiting_time = 10e-9,
                       amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        elif decoupled_qubit == 'qubit_1':
            self.add_X(name='X2_Q1', qubit = qubit_1, refgate = 'off_resonance', #waiting_time = waiting_time,
                       amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        else:
            raise NameError('decoupled qubit not found')
        
        self.add_CPhase(name = 'CP2', refgate = 'X2_Q1', waiting_time = 10e-9,
                        control_qubit = qubit_1, target_qubit = qubit_2,
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = detuning_time)
        
        self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z2_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_Z(name='Z2i_Q1', qubit = qubit_1, degree = 90)
        self.add_Z(name='Z2i_Q2', qubit = qubit_2, degree = 90)
        
        if DFS == 0:
            self.add_Z(name='Z1dfs_Q1', qubit = qubit_1, degree = 180)
            self.add_Z(name='Z1dfs_Q2', qubit = qubit_2, degree = 180)
            
        self.add_Y(name='Y2_Q1', qubit = qubit_1, refgate = 'CP2', waiting_time = 10e-9,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='Y2_Q2', qubit = qubit_2, refgate = 'Y2_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        return self


class Charge_Noise_Bob3(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.15)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 90)
        self.phase_2 = kw.pop('phase_2', 60)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0277)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.00)
        self.DFS = kw.pop('DFS', 0)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')
        self.decoupled_cphase = kw.pop('decoupled_cphase', False)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.DFS = kw.pop('DFS', self.DFS)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        self.decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        return self

    def make_circuit(self, **kw):
        decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        DFS = kw.pop('DFS', self.DFS)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        amplitude = 1
        off_resonance_amplitude = 1.2
        DFS = DFS%2
        
        te = 10e-9

        self.add_X(name='X1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        '''
        '''
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP1', refgate = 'X1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
            
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        '''
        '''
        if decoupled_cphase:
            self.add_CPhase(name = 'CP11', refgate = 'X1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Xpi_1_Q1', qubit = qubit_1, refgate = 'CP11', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Xpi_1_Q2', qubit = qubit_2, refgate = 'Xpi_1_Q1', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP1', refgate = 'Xpi_1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        
        '''
        '''
            
        if DFS == 1:
            self.add_Z(name='Z1dfs_Q1', qubit = qubit_1, degree = 180)

        self.add_X(name='X2_Q1', qubit = qubit_1, refgate = 'CP1', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X2_Q2', qubit = qubit_2, refgate = 'X2_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_Z(name='Z90_1_Q1', qubit = qubit_1, degree = 90)
        '''
        above is to prepare state to |00> + |11>
        '''
#        
#        self.add_single_qubit_gate(name='off_resonance', refgate = 'X2_Q1',
#                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
#                                   length = waiting_time, frequency_shift = -30e6)
        
        self.add_CPhase(name = 'off_resonance', refgate = 'X2_Q1', waiting_time = 0,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = 0, amplitude_target = 0, 
                            length = waiting_time)
        
        
        
        ##

        
        
        
        
        self.add_Z(name='Zde2_Q1', qubit = qubit_2, degree = 360 * 4e6*waiting_time)
        
#        if DFS == 1:
#            self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = -360 * 4e6*waiting_time)
#        if DFS != 1:
#            self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = 360 * 4e6*waiting_time)
        
        if add_dephase and 0:
            dephase = int(np.random.rand()*2//1)
            if dephase:
                self.add_Z(name='Zde_Q1', qubit = qubit_1, degree = 180)
                self.add_Z(name='Zde_Q2', qubit = qubit_2, degree = 180)
            
        '''
        above is for waiting
        '''
        
        self.add_Z(name='Z90_2_Q1', qubit = qubit_1, degree = 90)
        
        self.add_X(name='X3_Q1', qubit = qubit_1, refgate = 'off_resonance', #waiting_time = waiting_time,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='X3_Q2', qubit = qubit_2, refgate = 'X3_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        if DFS == 1:
            self.add_Z(name='Z2dfs_Q1', qubit = qubit_1, degree = 180)
            
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP2', refgate = 'X3_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
            
            self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z2_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z2i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z2i_Q2', qubit = qubit_2, degree = 90)
        
        if decoupled_cphase:
            self.add_CPhase(name = 'CP21', refgate = 'X3_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Xpi_2_Q1', qubit = qubit_1, refgate = 'CP21', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Xpi_2_Q2', qubit = qubit_2, refgate = 'Xpi_2_Q1', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP2', refgate = 'Xpi_2_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
            
        self.add_X(name='X4_Q1', qubit = qubit_1, refgate = 'CP2', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X4_Q2', qubit = qubit_2, refgate = 'X4_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        return self



class Charge_Noise_Bob_withaddednoise(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.15)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 90)
        self.phase_2 = kw.pop('phase_2', 60)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0277)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.00)
        self.DFS = kw.pop('DFS', 0)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')
        self.decoupled_cphase = kw.pop('decoupled_cphase', False)
        
        self.sigma1 = kw.pop('sigma1', 0)
        self.sigma2 = kw.pop('sigma2', 0)

        self.sigma3 = kw.pop('sigma3', 0)
        self.sigma4 = kw.pop('sigma4', 0)

        self.dummy = kw.pop('dummy', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.DFS = kw.pop('DFS', self.DFS)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        self.decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        
        self.sigma1 = kw.pop('sigma1', self.sigma1)
        self.sigma2 = kw.pop('sigma2', self.sigma2)
        self.sigma3 = kw.pop('sigma3', self.sigma3)
        self.sigma4 = kw.pop('sigma4', self.sigma4)
        self.dummy = kw.pop('dummy', 0)
        
        return self

    def make_circuit(self, **kw):
        decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        DFS = kw.pop('DFS', self.DFS)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        amplitude = 1
        off_resonance_amplitude = 1.2
        
        
        sigma1 = kw.pop('sigma1', self.sigma1)
        sigma2 = kw.pop('sigma2', self.sigma2)
        sigma3 = kw.pop('sigma3', self.sigma3)
        sigma4 = kw.pop('sigma4', self.sigma4)
        
        s1 = np.random.normal(0, sigma1, 1)
        s2 = np.random.normal(0, sigma2, 1)
        s3 = np.random.normal(0, sigma3, 1)
        s4 = np.random.normal(0, sigma4, 1)
        print(s1)
        print(s2)
        print(s3)
        print(s4)
        DFS = DFS%2
        
        te = 10e-9

        self.add_X(name='X1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        '''
        '''
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP1', refgate = 'X1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
            
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        '''
        '''
        if decoupled_cphase:
            self.add_CPhase(name = 'CP11', refgate = 'X1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Xpi_1_Q1', qubit = qubit_1, refgate = 'CP11', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Xpi_1_Q2', qubit = qubit_2, refgate = 'Xpi_1_Q1', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP1', refgate = 'Xpi_1_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        
        '''
        '''
            
        if DFS == 1:
            self.add_Z(name='Z1dfs_Q1', qubit = qubit_1, degree = 180)

        self.add_X(name='X2_Q1', qubit = qubit_1, refgate = 'CP1', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X2_Q2', qubit = qubit_2, refgate = 'X2_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_Z(name='Z90_1_Q1', qubit = qubit_1, degree = 90)
        '''
        above is to prepare state to |00> + |11>
        '''
#        
#        self.add_single_qubit_gate(name='off_resonance', refgate = 'X2_Q1',
#                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
#                                   length = waiting_time, frequency_shift = -30e6)
       



#        
        self.add_CPhase(name = 'off_resonance', refgate = 'X2_Q1', waiting_time = 0,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = s1[0], amplitude_target =s1[0], 
                            length = waiting_time)

#        self.add_CPhase(name = 'off_resonance0', refgate = 'X2_Q1', waiting_time = 0,
#                            control_qubit = qubit_1, target_qubit = qubit_2,
#                            amplitude_control = 0, amplitude_target = 0, 
#                            length = waiting_time/2)
#        
#        self.add_X(name='Xhahn_Q1', qubit = qubit_1, refgate = 'off_resonance0', waiting_time = 0,
#                   amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
#        
#        self.add_X(name='Xhahn_Q2', qubit = qubit_2, refgate = 'Xhahn_Q1', refpoint = 'start',
#                   amplitude = amplitude, length = qubit_2.Pi_pulse_length,)        
#        
#
#        self.add_CPhase(name = 'off_resonance', refgate = 'Xhahn_Q2', waiting_time = 0,
#                            control_qubit = qubit_1, target_qubit = qubit_2,
#                            amplitude_control = 0, amplitude_target = 0, 
#                            length = waiting_time/2)
        
        
        
        
        
        self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = 360 * (0e6 +s3[0] )*waiting_time)
        self.add_Z(name='Zde2_Q2', qubit = qubit_2, degree = 360 * (4e6 -s3[0] )*waiting_time)
        
#        if DFS == 1:
#            self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = -360 * 4e6*waiting_time)
#        if DFS != 1:
#            self.add_Z(name='Zde2_Q1', qubit = qubit_1, degree = 360 * 4e6*waiting_time)
        
        if add_dephase and 0:
            dephase = int(np.random.rand()*2//1)
            if dephase:
                self.add_Z(name='Zde_Q1', qubit = qubit_1, degree = 180)
                self.add_Z(name='Zde_Q2', qubit = qubit_2, degree = 180)
            
        '''
        above is for waiting
        '''
        
        self.add_Z(name='Z90_2_Q1', qubit = qubit_1, degree = 90)
        
        self.add_X(name='X3_Q1', qubit = qubit_1, refgate = 'off_resonance', #waiting_time = waiting_time,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='X3_Q2', qubit = qubit_2, refgate = 'X3_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        if DFS == 1:
            self.add_Z(name='Z2dfs_Q1', qubit = qubit_1, degree = 180)
            
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP2', refgate = 'X3_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
            
            self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z2_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z2i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z2i_Q2', qubit = qubit_2, degree = 90)
        
        if decoupled_cphase:
            self.add_CPhase(name = 'CP21', refgate = 'X3_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Xpi_2_Q1', qubit = qubit_1, refgate = 'CP21', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Xpi_2_Q2', qubit = qubit_2, refgate = 'Xpi_2_Q1', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP2', refgate = 'Xpi_2_Q1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
            self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
            
        self.add_X(name='X4_Q1', qubit = qubit_1, refgate = 'CP2', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_X(name='X4_Q2', qubit = qubit_2, refgate = 'X4_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        return self

class Grover(Manipulation):

     def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.2)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.detuning_time = kw.pop('detuning_time', 60e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 66)
        self.phase_2 = kw.pop('phase_2', 23)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0277)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.02)
        self.DFS = kw.pop('DFS', 0)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')

     def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.DFS = kw.pop('DFS', self.DFS)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        return self

     def make_circuit(self, **kw):
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        DFS = kw.pop('DFS', self.DFS)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        amplitude = 1
        off_resonance_amplitude = 1.2
        
        self.add_Y(name='Y1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='Y1_Q2', qubit = qubit_2, refgate = 'Y1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        
        self.add_CPhase(name = 'CP1', refgate = 'Y1_Q1', waiting_time = 10e-9,
                        control_qubit = qubit_1, target_qubit = qubit_2,
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = detuning_time)
        
        self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
        '''
        self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
        self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        '''
            
        self.add_single_qubit_gate(name='off_resonance', refgate = 'CP1', waiting_time = 10e-9,
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = -30e6)
        
        self.add_Y(name='Y2_Q1', qubit = qubit_1, refgate = 'off_resonance', #waiting_time = 10e-9,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='Y2_Q2', qubit = qubit_2, refgate = 'Y2_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        


        '''
        above is to prepare state to |00> + |11>
        '''
        
        self.add_CPhase(name = 'CP2', refgate = 'Y2_Q1', waiting_time = 10e-9,
                        control_qubit = qubit_1, target_qubit = qubit_2,
                        amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                        length = detuning_time)
        
        self.add_Z(name='Z2_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z2_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_Z(name='Z2i_Q1', qubit = qubit_1, degree = 180)
        self.add_Z(name='Z2i_Q2', qubit = qubit_2, degree = 180)
        
        self.add_Y(name='Y4_Q1', qubit = qubit_1, refgate = 'CP2', waiting_time = 10e-9,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
        self.add_Y(name='Y4_Q2', qubit = qubit_2, refgate = 'Y4_Q1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        return self
