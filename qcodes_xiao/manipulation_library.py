# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:01:07 2017

@author: X.X
"""

import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from manipulation import Manipulation
import stationF006
#%%


#%% by objects
class Finding_Resonance(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.waiting_time = kw.pop('waiting_time', 0)
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

    def make_circuit(self, qubit = 'Qubit_1'):

#        N = qubit[-1]-1

        self.add_single_qubit_gate(name = 'T1_Q1', qubit = self.qubits[1], amplitude = 1, length = 200e-9, frequency_shift = 0)
#        self.add_single_qubit_gate(name = 'T2_Q1', refpoint = 'start', waiting_time = 0,
#                                   qubit = self.qubits[1], amplitude = 1, length = 200e-9, frequency_shift = -10e6)
#        self.add_CPhase(name = 'CP_Q12', refgate = 'T1_Q1', refpoint = 'start', control_qubit = self.qubits[0], target_qubit = self.qubits[0],
#                        amplitude_control = 0.5, amplitude_target = -0.6, length = 200e-9, waiting_time = 0)
#        self.add_single_qubit_gate(name = 'T2_Q1', refgate = 'T1_Q1', qubit = self.qubits[1], amplitude = 1, 
#                                   length = 500e-9, frequency_shift = 2e6, waiting_time = 50e-9)

#        self.add_X(name='X2_Q1', refgate = 'T1_Q1', refpoint = 'start', qubit = self.qubits[1], waiting_time = 0)
#        self.add_Y(name='Y1_Q1', qubit = self.qubits[0],)

        return self

class Ramsey(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.waiting_time = kw.pop('waiting_time', 0)
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

    def make_circuit(self,):

        self.add_X(name='X1_Q1', qubit = self.qubits[1],)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = self.qubits[1], waiting_time = self.parameter1,)

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
        self.waiting_time = kw.pop('waiting_time', 0)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None

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

    def make_circuit(self, qubit = 2, **kw):
        
        qubit_num = qubit
        
        qubit = self.qubits[int(qubit_num-1)]
        
        length = kw.pop('duration_time', qubit.Pi_pulse_length)
        amplitude = kw.pop('amplitude', 1)

        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = self.qubits[1], amplitude = amplitude, length = length)

        return self

class CRot_Readout(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)

        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

        return self

    def make_circuit(self, waiting_time = 0):

        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = self.qubits[0], amplitude = self.parameter1, length = self.parameter2)

        return self


class Grover(Manipulation):

    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)

        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

        return self

    def make_circuit(self, waiting_time = 0):

        self.add_Y(name='Y1_Q1', qubit = self.qubits[0],)

        self.add_Y(name='Y1_Q2', refgate = 'Y1_Q1', qubit = self.qubits[0],)

        return self





class new_ex(Manipulation):

    def __init__(self, name,waiting_time, **kw):

        super().__init__(name,)

#        self.refphase = {}

        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)
#        self.qubits = kw.pop('qubits', None)
#        if self.qubits is not None:
#            self.qubits_name = [qubit.name for qubit in self.qubits]
#            self.refphase = {qubit.name: 0 for qubit in self.qubits}
#        self.pulsar = kw.pop('pulsar', None)

    def __call__(self, **kw):

#        self.qubits = kw.pop('qubits', None)
#        if self.qubits is not None:
#            self.qubits_name = [qubit.name for qubit in self.qubits]
#            self.refphase = {qubit.name: 0 for qubit in self.qubits}
#        self.pulsar = kw.pop('pulsar', None)
        self.parameter1 = kw.pop('parameter1', 0)
        self.parameter2 = kw.pop('parameter2', 0)

#        if self.pulsar is not None:
#            self.clock = self.pulsar.clock
#
#            for c in self.pulsar.channels:
#                chan = self.pulsar.channels[c]
#                delay = chan['delay'] if not(self.ignore_delays) else 0.
#                self.define_channel(name=c, type=chan['type'],
#                                    high=chan['high'], low=chan['low'],
#                                    offset=chan['offset'],
#                                    delay=delay)
        return self

    def make_circuit(self,):

        self.add_X(name='X1_Q1', qubit = self.qubits[0],)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = self.qubits[0], waiting_time = self.parameter1,)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = self.qubits[0], waiting_time = self.parameter2,)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = self.qubits[0], waiting_time = 1,)

        return self


#sweep_x(name = 'X1_Q1', parameter = 'waiting_time')
#sweep_y


#%% by functions


#def make_manipulation(manipulation = Manipulation(name = 'Manip'), qubits = [], **kw):
#
#    waiting_time = kw.pop('waiting_time', None)
#    amplitude = kw.pop('amplitude', None)
#
#    manip = make_Ramsey(manipulation = manipulation, qubits = qubits, waiting_time = waiting_time)
#
#    return manip
#
#
#
#def make_Ramsey(manipulation = Manipulation(name = 'Manip'), qubits = [], waiting_time = 0, **kw):
#
#    qubit_1 = qubits[0]
#
#    manipulation.add_X(name='X1_Q1', qubit = qubit_1,)
#
#    manipulation.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = qubit_1, waiting_time = waiting_time,)
#
#    return manipulation
#
#def make_Rabi(manipulation = Manipulation(name = 'Manip'), qubits = [], **kw):
#
#    qubit_1 = qubits[0]
#
#    duration_time = kw.pop('duration_time', qubit_1.Pi_pulse_length)
#
#    manipulation.add_single_qubit_gate(name = 'Rabi')
#
#    return manipulation
#
#def calibrate_X_Pi(manipulation = Manipulation(name = 'Manip'), qubits = [],**kw):
#
#    qubit_1 = qubits[0]
#
#    duration_time = kw.pop('duration_time')
#    repetition = kw.pop('repetitions')
#
#    manipulation.add_X(name = 'X1', qubit = qubit_1,)
#
#    for i in range(repetition):
#        manipulation.add_single_qubit_gate(name = 'X_Pi_%d'%(i+1),qubit = qubit_1, length = duration_time)
#
#    return manipulation
