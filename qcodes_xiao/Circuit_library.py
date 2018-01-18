# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:45:09 2018

@author: twatson
"""
import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from qcodes.instrument.base import Instrument
#from experiment import Experiment
from manipulation_new import Manipulation2
from Circuit import Circuit
import stationF006


class Ramsey2(Circuit):
    
    def __init__(self, name, pulsar, **kw):
        
        super().__init__(name, pulsar, **kw)
        
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_1')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
            
        qubit = Instrument.find_instrument(self.qubit)    
        
        self.waiting_time = kw.pop('waiting_time', 0)
        self.amplitude = kw.pop('amplitude', 1)
        self.frequency_shift = kw.pop('frequency_shift', 0)
        self.length = kw.pop('duration_time', 125e-9)
        self.phase_1 = kw.pop('phase_1', 0)
        self.phase_2 = kw.pop('phase_2', 0)
        self.off_resonance = kw.pop('off_resonance', False)
        
        self.add_X(name='X1_Q1', qubit = qubit,
                   amplitude = self.amplitude, length = self.length, frequency_shift = self.frequency_shift)
        
        self.add_Z(name='Z1_Q1', qubit = qubit, degree = self.phase_2)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = qubit, waiting_time = self.waiting_time,
                   amplitude = self.amplitude, length = self.length, frequency_shift = self.frequency_shift)
        
        