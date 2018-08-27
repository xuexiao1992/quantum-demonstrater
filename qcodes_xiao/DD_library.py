# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:21:38 2018

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


#%%

Bootstrap_array = [
        ['X', 'I'], 
        ['Y', 'I'],
        
        ['Xpi', 'X'], 
        ['Ypi', 'Y'],
        ['X', 'Ypi'], 
        ['Y', 'Xpi'],
        
        ['X', 'Y'],
        ['Y', 'X'], 
        ['Y', 'Xpi', 'X'],
        ['X', 'Xpi', 'Y'], 
        ['Y', 'Ypi', 'X'], 
        ['X', 'Ypi', 'Y'],
        ]

class Bootstrap(Manipulation):

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
        self.off_resonance_amplitude = kw.pop('off_resonacne_amplitude', 1.15)
        self.phase_error = kw.pop('phase_error', 0)
        self.error_gate = kw.pop('error_gate', None)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', self.qubits)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', self.pulsar)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.off_resonance_amplitude = ('off_resonacne_amplitude', self.off_resonance_amplitude)
        self.phase_error = kw.pop('phase_error', self.phase_error)
        self.error_gate = kw.pop('error_gate', self.error_gate)
        return self

    def make_circuit(self, **kw):
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        off_resonance_amplitude = ('off_resonacne_amplitude', self.off_resonance_amplitude)
        phase_error = kw.pop('phase_error', self.phase_error)
        error_gate = kw.pop('error_gate', self.error_gate)
        
        g = int(kw.pop('gate', 1)-1)
        gate = Bootstrap_array[g]
        
        for i in range(len(gate)):
            
            amplitude = 0 if gate[i] == 'I' else 1
            '''
            if gate[i] == 'I':
                amplitude = off_resonance_amplitude if qubit_name == 'qubit_1' else 0
                frequency_shift = -30e6 if qubit_name == 'qubit_1' else 0
            '''
            axis = [1,0,0] if gate[i].startswith('X') else [0,1,0]
            length = qubit.Pi_pulse_length if gate[i].endswith('pi') else qubit.halfPi_pulse_length
            
            name = 'G%d'%(i+1)
            refgate = None if i == 0 else 'G%d'%i
            
            if gate[i] == error_gate:
                self.add_Z(name='error_phase', qubit = qubit, degree = phase_error)
            
            self.add_single_qubit_gate(name = name, refgate = refgate, 
                                       qubit = qubit, axis = axis,
                                       amplitude = amplitude, length = length, 
                                       frequency_shift = frequency_shift)
            
            if gate[i] == error_gate:
                self.add_Z(name='error_phase_2', qubit = qubit, degree = -phase_error)
                
        return self
    


#%%
class Rabi_all_with_detuning(Manipulation):

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
        self.max_length = kw.pop('max_duration', 3e-6)
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
        
        self.max_length = kw.pop('max_duration', self.max_length)
        
        self.T_amplitude = kw.get('T_amplitude', self.T_amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.third_tone = kw.pop('third_tone', self.third_tone)
        return self

    def make_circuit(self, **kw):
        
        amplitude = kw.get('amplitude', self.amplitude)
        
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        
        T_amplitude = kw.get('T_amplitude', self.T_amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
#        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        length = kw.get('duration_time', self.length)
        max_length = kw.pop('max_duration', self.max_length)
        third_tone = kw.pop('third_tone', self.third_tone)
#        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = qubit_2, amplitude = amplitude, 
#                                   length = 250e-9, frequency_shift = 0)

        self.add_Z(name='Z1_Q1', qubit = qubit_1, degree = phase_1)
        self.add_Z(name='Z1_Q2', qubit = qubit_2, degree = phase_2)
        
        self.add_X(name='X1_Q1', qubit = qubit_1, #refgate = 'Rabi_Oscillation',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
            
        self.add_X(name='X1_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        refgate = 'X1_Q1' if length > 0 else None
        
        self.add_single_qubit_gate(name = 'off_resonance_pulse', qubit = qubit_1, 
                                   refgate = refgate, refpoint = 'end',
                                   amplitude = off_resonance_amplitude, length = max_length-length, frequency_shift = -30e6)
        
        
        if third_tone:
            self.add_Y(name='X2_Q2', qubit = qubit_2, refgate = 'X1_Q1', refpoint = 'start',
                       amplitude = 0.4, length = length, frequency_shift = 0)

#        self.add_CPhase(name = 'CP_Q12', control_qubit = qubit_1, target_qubit = qubit_2,
#                        refgate = 'X1_Q2', refpoint = 'start', waiting_time = -100e-9,
#                        amplitude_control = 0, amplitude_target = T_amplitude, length = length+150e-9)

        return self


#%%
class Rabi_back_forth(Manipulation):

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
        
        Pi_length = qubit.Pi_pulse_length
        Pi_num = length//Pi_length
        last_length = length%qubit.Pi_pulse_length
#        driving_length = 0
        
#        while driving_length < length:
        for i in range(Pi_num):   
            if i%2 == 0:
                self.add_single_qubit_gate(name='Rabi_Oscillation_%d'%i, qubit = qubit, amplitude = amplitude, axis = [1,0,0], 
                                           length = length, frequency_shift = frequency_shift)
            else:
                self.add_single_qubit_gate(name='Rabi_Oscillation_%d'%i, qubit = qubit, amplitude = amplitude, axis = [-1,0,0], 
                                           length = length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='Rabi_heating', refgate = 'Rabi_Oscillation', qubit = qubit, amplitude = amplitude, #axis = [0,1,0], 
                                   length = 3.05e-6-length, frequency_shift = frequency_shift-20e6)

        return self


#%%

class Hahn_all(Manipulation):

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
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)
        self.phase = kw.pop('phase', 0)
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
        self.phase = kw.pop('phase', self.phase)
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
        
        
        self.add_single_qubit_gate(name='off_resonance_1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time/2, frequency_shift = -30e6)
        

        self.add_single_qubit_gate(name='decouple_Q1', refgate = 'off_resonance_1',
                               qubit = qubit_1, amplitude = 1, 
                               length = qubit_1.Pi_pulse_length, frequency_shift = 0)
        
        self.add_single_qubit_gate(name='decouple_Q2', refgate = 'decouple_Q1', refpoint = 'start',
                                   qubit = qubit_2, amplitude = 1, 
                                   length = qubit_2.Pi_pulse_length, frequency_shift = 0)
        
        
        self.add_single_qubit_gate(name='off_resonance_2', refgate = 'decouple_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time/2, frequency_shift = -30e6)

        self.add_X(name='X2_Q1', refgate = 'off_resonance_2', 
                   qubit = qubit_1, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        

        self.add_X(name='X2_Q2', refgate = 'X2_Q1', refpoint = 'start',
                   qubit = qubit_2, waiting_time = 0,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        return self


#%%
        
    
class DCZ_ledge(Manipulation):
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
       
        self.ledge_amplitude1 = kw.pop('ledge_amplitude1', 30*0.5*-0.020)
        self.ledge_amplitude2 = kw.pop('ledge_amplitude2', 30*0.5*0.020)
        
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
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
        self.detuning_amplitude2 = kw.pop('detuning_amplitude2', self.detuning_amplitude2)
        self.detuning_amplitude3 = kw.pop('detuning_amplitude3', self.detuning_amplitude3)
        self.detuning_amplitude4 = kw.pop('detuning_amplitude4', self.detuning_amplitude4)
        
        self.ledge_amplitude1 = kw.pop('ledge_amplitude1', self.ledge_amplitude1)
        self.ledge_amplitude2 = kw.pop('ledge_amplitude2', self.ledge_amplitude2)
        
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
        
        ledge_amplitude1 = kw.pop('ledge_amplitude1', self.ledge_amplitude1)
        ledge_amplitude2 = kw.pop('ledge_amplitude2', self.ledge_amplitude2)
        
        
#        detuning_amplitude = 30*0.5*0.032
#        detuning_amplitude2 = 30*0.5*0.0
#        detuning_amplitude3 = detuning_amplitude + 30*0.5*0.080
#        detuning_amplitude4 = 30*0.5*-0.02
        
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
        ledge_t = 20e-9
#        off_resonance_amplitude = 1.2
        
     
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

        self.add_CPhase(name = 'CP1_ledge1', refgate = 'wait1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = ledge_amplitude1, amplitude_target = ledge_amplitude2, 
                        length = ledge_t)

        self.add_CPhase(name = 'CP1_Q12', refgate = 'CP1_ledge1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'CP1_ledge2', refgate = 'CP1_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = ledge_amplitude1, amplitude_target = ledge_amplitude2, 
                        length = ledge_t)
#        
        self.add_CPhase(name = 'wait2', refgate = 'CP1_ledge2',
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
        
        self.add_CPhase(name = 'CP2_ledge1', refgate = 'wait3',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = ledge_amplitude1, amplitude_target = ledge_amplitude2, 
                        length = ledge_t)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'CP2_ledge1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_CPhase(name = 'CP2_ledge2', refgate = 'CP2_Q12',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = ledge_amplitude1, amplitude_target = ledge_amplitude2, 
                        length = ledge_t)
#        
        self.add_CPhase(name = 'wait4', refgate = 'CP2_ledge2',
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

        
    

#%%


class DCZ_ramp(Manipulation):
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
       
        self.ledge_amplitude1 = kw.pop('ledge_amplitude1', 30*0.5*-0.020)
        self.ledge_amplitude2 = kw.pop('ledge_amplitude2', 30*0.5*0.020)
        
        self.Pi_amplitude = kw.pop('Pi_amplitude', 0)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.phase = kw.pop('phase', 0)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1.15)
        self.control_qubit = kw.pop('control_qubit', 'qubit_2')
        self.ramp_time = kw.pop('ramp_time', 20e-9)

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
        
        self.ledge_amplitude1 = kw.pop('ledge_amplitude1', self.ledge_amplitude1)
        self.ledge_amplitude2 = kw.pop('ledge_amplitude2', self.ledge_amplitude2)
        
        self.Pi_amplitude = kw.pop('Pi_amplitude', self.Pi_amplitude)
        self.frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.phase = kw.pop('phase', self.phase)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.control_qubit = kw.pop('control_qubit', self.control_qubit)
        self.ramp_time = kw.pop('ramp_time', self.ramp_time)
        return self

    def make_circuit(self, **kw):
        
        phase = kw.pop('phase', self.phase)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        
        detuning_amplitude = kw.get('detuning_amplitude', self.detuning_amplitude)
        detuning_amplitude2 = kw.get('detuning_amplitude2', self.detuning_amplitude2)
        
        detuning_amplitude3 = kw.pop('detuning_amplitude3', self.detuning_amplitude3)
        detuning_amplitude4 = kw.pop('detuning_amplitude4', self.detuning_amplitude4)
        
        ledge_amplitude1 = kw.pop('ledge_amplitude1', self.ledge_amplitude1)
        ledge_amplitude2 = kw.pop('ledge_amplitude2', self.ledge_amplitude2)
        ramp_time = kw.pop('ramp_time', self.ramp_time)
        
#        detuning_amplitude = 30*0.5*0.032
#        detuning_amplitude2 = 30*0.5*0.0
#        detuning_amplitude3 = detuning_amplitude + 30*0.5*0.080
#        detuning_amplitude4 = 30*0.5*-0.02
        
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
        
        te = 100e-9
        ledge_t = ramp_time
#        off_resonance_amplitude = 1.2
        
     
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
                       amplitude = off_resonance_amplitude, length = self.qubits[1].halfPi_pulse_length+30e-9, 
                       frequency_shift = frequency_shift-30e6,)
        


        self.add_CPhase(name = 'wait1', refgate = 'X1_Q2',#''XPi_Q1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = 0, amplitude_target = 0, 
                        length = te)

        self.add_Ramp(name = 'CP1_ledge1', refgate = 'wait1',
                      control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                      amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, ramp = 'up',
                      length = ledge_t)

        self.add_CPhase(name = 'CP1_Q12', refgate = 'CP1_ledge1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_Ramp(name = 'CP1_ledge2', refgate = 'CP1_Q12',
                      control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                      amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, ramp = 'down', 
                      length = ledge_t)
#        
        self.add_CPhase(name = 'wait2', refgate = 'CP1_ledge2',
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
        
        self.add_Ramp(name = 'CP2_ledge1', refgate = 'wait3',
                      control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                      amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, ramp = 'up',
                      length = ledge_t)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'CP2_ledge1',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, 
                        length = detuning_time/2)
        
        self.add_Ramp(name = 'CP2_ledge2', refgate = 'CP2_Q12',
                      control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                      amplitude_control = detuning_amplitude, amplitude_target = detuning_amplitude2, ramp = 'down',
                      length = ledge_t)
#        
        self.add_CPhase(name = 'wait4', refgate = 'CP2_ledge2',
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
                                       length = self.qubits[0].halfPi_pulse_length+30e-9, frequency_shift = frequency_shift-30e6)

        return self




