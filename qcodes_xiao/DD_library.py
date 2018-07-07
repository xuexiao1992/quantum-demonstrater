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
