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
