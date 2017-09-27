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
        return self

    def make_circuit(self, **kw):
        
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        
        qubit_name = kw.pop('qubit', self.qubit)
        
        qubit = Instrument.find_instrument(qubit_name)
        
        length = kw.get('duration_time', qubit.halfPi_pulse_length)
        
        
        self.add_X(name='X1_Q1', qubit = qubit,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)

        self.add_X(name='X2_Q1', refgate = 'X1_Q1', qubit = qubit, waiting_time = waiting_time,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        
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
        self.length = kw.pop('duration_time', 250e-9)

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
        
        amplitude = kw.get('amplitude', self.amplitude)
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)
        print('length', length)

        self.add_single_qubit_gate(name='Rabi_Oscillation', qubit = qubit, amplitude = amplitude, 
                                   length = length, frequency_shift = frequency_shift)

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
        frequency_shift = kw.pop('frequency_shift', self.frequency_shift)

        self.add_CPhase(name = 'CP_Q12', control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = amplitude, amplitude_target = 0, length = length+150e-9)
        
        self.add_single_qubit_gate(name='CRot', refgate = 'CP_Q12', qubit = self.qubits[1], 
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
        
        self.add_single_qubit_gate(name='X_Pi_Q1', qubit = self.qubits[0], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_X(name='X1_Q2', refgate = 'X_Pi_Q1', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X1_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)

        self.add_CPhase(name = 'CP_Q12', refgate = 'X1_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = detuning_time)
        
        self.add_Z(name='Z1_Q1', qubit = self.qubits[1], degree = phase)
        
        self.add_X(name='X2_Q2', refgate = 'CP_Q12', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        self.add_single_qubit_gate(name='off_resonance2_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)


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
        
        self.add_single_qubit_gate(name='XPi_Q1', qubit = self.qubits[0], amplitude = Pi_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = 0)
        
        self.add_X(name='X1_Q2', refgate = 'XPi_Q1', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        self.add_single_qubit_gate(name='off_resonance1_Q1', refgate = 'X1_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)

        self.add_CPhase(name = 'CP1_Q12', refgate = 'X1_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = detuning_time/2)
        
        self.add_single_qubit_gate(name='XPi2_Q1', refgate = 'CP1_Q12',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[0].Pi_pulse_length, frequency_shift = frequency_shift-30e6)
        
        self.add_single_qubit_gate(name='XPi2_Q2', refgate = 'CP1_Q12',
                                   qubit = self.qubits[1], amplitude = 1, 
                                   length = self.qubits[1].Pi_pulse_length, frequency_shift = 0)
        
        self.add_CPhase(name = 'CP2_Q12', refgate = 'XPi2_Q2',
                        control_qubit = self.qubits[0], target_qubit = self.qubits[1],
                        amplitude_control = detuning_amplitude, amplitude_target = 0, 
                        length = detuning_time/2)
        
        self.add_Z(name='Z1_Q1', qubit = self.qubits[1], degree = phase)
        
        self.add_X(name='X2_Q2', refgate = 'CP2_Q12', qubit = self.qubits[1], 
                   amplitude = 1, length = self.qubits[1].halfPi_pulse_length, 
                   frequency_shift = frequency_shift,)
        
        self.add_single_qubit_gate(name='off_resonance2_Q1', refgate = 'X2_Q2', refpoint = 'start',
                                   qubit = self.qubits[0], amplitude = off_resonance_amplitude, 
                                   length = self.qubits[1].halfPi_pulse_length, frequency_shift = frequency_shift-30e6)

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
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',1)
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
        
        
        self.add_X(name='X1_Q1', qubit = qubit_1,
                   amplitude = amplitude, length = length, frequency_shift = frequency_shift)
        
        self.add_single_qubit_gate(name='off_resonance_Q1', refgate = 'X1_Q1',
                                   qubit = qubit_1, amplitude = off_resonance_amplitude, 
                                   length = waiting_time, frequency_shift = frequency_shift-30e6)

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





