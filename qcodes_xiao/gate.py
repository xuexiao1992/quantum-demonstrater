# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:35:17 2017

@author: X.X
"""


import numpy as np
from scipy import constants as C

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
import math
from qubit import Qubit

#%% gate

class Gate:
    
    def __init__(self, name, **kw):

        self.name = name                                 ## name of a gate, e.g. 'X_Q1' 'CZ_Q12'
        self.qubit_name = 'qubit'
        self.qubits = []
        

##      Gates will be an object that consists all the pulse information in the operation,
##      but it will just store the pulses in information and the links in time domain 
##      between different pulses, but not adding it into an Element 

#%% single qubit gate

class Single_Qubit_Gate(Gate):
    
    def __init__(self, name, qubit, rotating_axis = [1, 0, 0], amplitude = None, frequency = None, refphase = 0):
        super().__init__(name,)
        
        self.qubit = qubit.name
        self.amplitude = qubit.IQ_amplitude if amplitude == None else amplitude
        self.frequency = qubit.frequency if frequency == None else frequency
        self.channel_I = qubit.microwave_gate['channel_I']
        self.channel_Q = qubit.microwave_gate['channel_Q']
        self.channel_PM = qubit.microwave_gate['channel_PM']
        self.channel_FM = qubit.microwave_gate['channel_FM']
        self.channel_VP = qubit.plunger_gate['channel_VP']        
        
        self.Pi_pulse_length = qubit.Pi_pulse_length
        self.halfPi_pulse_length = qubit.halfPi_pulse_length
        
        
        self.voltage_pulse_length = 0
        
        self.refphase = refphase*C.pi/180
        
        self.axis = np.array(rotating_axis)
        self.axis_angle = np.arctan(self.axis[1]/self.axis[0]) if self.axis[0]!=0 else C.pi/2
        
#        self.pulses = {
#                ##  'microwave': None,
#                ##  'voltage': None
#                }      ## this will be the returned value and used in the manipulation object
        
        self.pulses = [None, None, None, None]            ## [microwave1_I, microwave1_Q, voltage, microwave2_I, microwave2_Q]
        
    
    def XY_rotation(self, degree = 90, length = None, waiting_time = 0, refgate = None, refpoint = 'end'):
#        global phase
#        IQ_Modulation = self.frequency
        if length is not None:
            pulse_length = length 
        else:
            pulse_length = self.halfPi_pulse_length if degree == 90 else degree*self.Pi_pulse_length/180

        pulse_amp = self.amplitude
        
        ## voltage pulse is not used here
        voltage_pulse = SquarePulse(channel = self.channel_VP, name = '%s_voltage_pulse'%self.name,
                                    amplitude = 0, length = pulse_length + waiting_time)
        
        PM_pulse = SquarePulse(channel = self.channel_PM, name = '%s_PM_pulse'%self.name,
                               amplitude = 2, length = pulse_length+200e-9)
        
        if 1:
            microwave_pulse_I = SquarePulse(channel = self.channel_I, name = '%s_microwave_pulse_I'%self.name, 
                                            amplitude = pulse_amp*np.cos(self.refphase + self.axis_angle), 
                                            length = pulse_length)
            
            microwave_pulse_Q = SquarePulse(channel = self.channel_Q, name = '%s_microwave_pulse_Q'%self.name, 
                                            amplitude = pulse_amp*np.sin(self.refphase + self.axis_angle), 
                                            length = pulse_length)
            
            
        elif 0: ##here frequency and phase is not yet ready!!!!!!!!!!!
            microwave_pulse_I = CosPulse(channel = self.channel_I, name = '%s_microwave_pulse_I'%self.name, frequency = 1e6,
                                         phase = 0, amplitude = 0.2*np.cos(self.refphase + self.axis_angle),
                                         length = pulse_length)
            
            microwave_pulse_Q = CosPulse(channel = self.channel_Q, name = '%s_microwave_pulse_Q'%self.name, frequency = 1e6, 
                                         phase = 0, amplitude = 0.2*np.sin(self.refphase + self.axis_angle),
                                         length = pulse_length)
            
            
        self.pulses[0] = {
                'pulse': voltage_pulse,
                'pulse_name': voltage_pulse.name,
                'refpulse': None if refgate == None else refgate[0]['pulse_name'],
                'refpoint': refpoint,
                'waiting': 0
                }
               
        
        self.pulses[1] = {
                'pulse': microwave_pulse_I,
                'pulse_name': microwave_pulse_I.name,
                'refpulse': None if refgate == None else refgate[-2]['pulse_name'],                   ## name of the refpulse
                'refpoint': refpoint,
                'waiting': waiting_time
                }
        
        self.pulses[2] = {
                'pulse': microwave_pulse_Q,
                'pulse_name': microwave_pulse_Q.name,
                'refpulse': '%s_microwave_pulse_I'%self.name,
                'refpoint': 'start',
                'waiting': 0
                }
        ##  here you just construct a dictionary which contains all the information of pulses you use
        self.pulses[3] = {
                'pulse': PM_pulse,
                'pulse_name': PM_pulse.name,
                'refpulse': '%s_microwave_pulse_I'%self.name,
                'refpoint': 'start',
                'waiting': -200e-9
                }


        return True
   
    
    
    def Z_rotation(self, degree, name = 'Z_rotation', refphase = 0, waiting_time = 0, refrotation = None):
#        global refphase
#        refphase += degree
        return True
    
    
#    def Arbitrary_rotation(self, degree, rotating_axis = [1,0,0]):
#        theta = np.arccos(rotating_axis[2])
#        phi = np.arccos(rotating_axis[0]/np.sin(theta))
#        self.Z_rotation(degree = -phi, name = 'step1')
#        self.Y_rotation(degree = -theta, name = 'step2', refrotation = 'step1')
#        self.Z_rotation(degree = degree, name = 'step3', refrotation = 'step2')
#        self.Y_rotation(degree = theta, name = 'step4', refrotation = 'step3')
#        self.Z_rotation(degree = phi, refrotation = 'step4')
#        return True



#%% twp qubit gate


class Two_Qubit_Gate(Gate):
    
    def __init__(self, name, control_qubit, target_qubit, refphase = 0,):
        
        super().__init__(name,)
        
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        
        """
        the three parameters below are very important, 
        """
        self.frequency = target_qubit.frequency
        self.refphase = refphase
        self.Pi_pulse_length = target_qubit.Pi_pulse_length
        
        self.detuning_amplitude_C = 0.15
        self.detuning_amplitude_T = -0.1
        
        self.detuning_length_Pi = 10e-6
        self.channel_VP1 = control_qubit.plunger_gate['channel_VP']
        self.channel_VP2 = target_qubit.plunger_gate['channel_VP']
        self.channel_I = target_qubit.microwave_gate['channel_I']
        self.channel_Q = target_qubit.microwave_gate['channel_Q']
        
        self.channel_FM = target_qubit.microwave_gate['channel_FM']
        self.channel_PM = target_qubit.microwave_gate['channel_PM']
        
        self.pulses = []
        
        
        
    
    def CRotation_gate(self, exchanging_time, refgate):
        self.detuning(refpulse = refgate[-1]['pulse_name'])
        
        self.X_Pi_gate(refpulse = self.pulses[-1]['pulse_name'])
        
        return True
    
    def CPhase_gate(self, name = 'CPhase_gate', rotating_phase = 180, refgate = None, waiting_time = 0):
        
        length = rotating_phase*self.detuning_length_Pi/180
        
        self.detuning(name = name, length = length,
                       waiting_time = waiting_time, refpulse = None if refgate == None else refgate[-1]['pulse_name'])
        return True
    
    def CNot_gate(self, exchanging_time, delay_time, refgate):
        self.X_gate(refpulse = refgate[-1]['pulse_name'])
        self.CPhase_gate(refpulse = self.pulses[-1]['pulse_name'])
        self.X_gate(refpulse = self.pulses[-1]['pulse_name'])
        return True
   

    def detuning(self, name = 'detuning_pulse', length = 0, waiting_time = 0, refpulse = None, refpoint = 'end'):
        
        detuning_pulse_C = SquarePulse(channel = self.channel_VP1, name = '%s_detuning_pulse_C'%self.name, 
                                       amplitude = self.detuning_amplitude_C, length = length, 
                                       refpulse = refpulse)
        
        detuning_pulse_T = SquarePulse(channel = self.channel_VP2, name = '%s_detuning_pulse_T'%self.name, 
                                       amplitude = self.detuning_amplitude_T, length = length, 
                                       refpulse = refpulse)
        
        detuning_pulse_C = {
                'pulse': detuning_pulse_C,
                'pulse_name': detuning_pulse_C.name,
                'refpulse': None if refpulse == None else refpulse,
                'refpoint': refpoint,
                'waiting': waiting_time
                }
        
        detuning_pulse_T = {
                'pulse': detuning_pulse_T,
                'pulse_name': detuning_pulse_T.name,
                'refpulse': '%s_detuning_pulse_C'%self.name,
                'refpoint': 'start',
                'waiting': 0
                }
                
        self.pulses.append(detuning_pulse_C)
        self.pulses.append(detuning_pulse_T)

        return True
    
    
    def XY_rotation(self, name, degree = 90, waiting_time = 0, refpulse = None, refpoint = 'end'):

        microwave_pulse_I = SquarePulse(channel = self.channel_I, name = '%s_microwave_pulse_I'%name, 
                   amplitude = np.cos(self.refphase), length = degree*self.Pi_pulse_length/180)
        
        microwave_pulse_Q = SquarePulse(channel = self.channel_Q, name = '%s_microwave_pulse_Q'%name, 
                   amplitude = np.sin(self.refphase), length = degree*self.Pi_pulse_length/180)
        
        component_I = {
                'pulse': microwave_pulse_I,
                'pulse_name': microwave_pulse_I.name,
                'refpulse': refpulse,                   ## name of the refpulse
                'refpoint': refpoint,
                'waiting': waiting_time
                }
        
        component_Q = {
                'pulse': microwave_pulse_Q,
                'pulse_name': microwave_pulse_Q.name,
                'refpulse': '%s_microwave_pulse_I'%name,
                'refpoint': 'start',
                'waiting': 0
                }
        
        self.pulses.append(component_I)
        self.pulses.append(component_Q)



    
    def X_gate(self, name = 'X_halfPi', waiting_time = 0, refpulse = None, refpoint = 'end'):
        
        self.XY_rotation(name = name, degree = 90, waiting_time = waiting_time, refpulse = refpulse, refpoint = refpoint)
        
        return 0
    
    def X_Pi_gate(self, name = 'X_Pi', waiting_time = 0, refpulse = None, refpoint = 'end'):
        
        self.XY_rotation(name = name, degree = 180, waiting_time = waiting_time, refpulse = refpulse, refpoint = refpoint)
        
        return 0
    
    def Y_gate(self, name = 'Y_halfPi', waiting_time = 0, refpulse = None, refpoint = 'end'):
        
        self.XY_rotation(name = name, degree = 90, waiting_time = waiting_time, refpulse = refpulse, refpoint = refpoint)
        
        return 0
    
    def Y_Pi_gate(self, name = 'Y_Pi', waiting_time = 0, refpulse = None, refpoint = 'end'):
        
        self.XY_rotation(name = name, degree = 180, waiting_time = waiting_time, refpulse = refpulse, refpoint = refpoint)
        
        return 0
    
       
   
class CPhase_Gate(Two_Qubit_Gate):
    
    def __init__(self, name = 'CPhase_gate', control_qubit = None, target_qubit = None, 
                 rotating_phase = 180, refgate = None, refpoint = 'end', waiting_time = 0):
        
        super().__init__(name, control_qubit, target_qubit)
        
        length = rotating_phase*self.detuning_length_Pi/180
        
        self.detuning(name = name, length = length, waiting_time = waiting_time, refpoint = refpoint,
                      refpulse = None if refgate == None else refgate[-2]['pulse_name'])
        
    def __call__(self, **kw):
        
        return self
       
        
class CNot_Gate(Two_Qubit_Gate):
    
    def __init__(self, name = 'CPhase_gate', control_qubit = None, target_qubit = None, 
                 rotating_phase = 180, refgate = None, refpoint = 'end', waiting_time = 0):
        
        super().__init__(name, control_qubit, target_qubit)
        
        length = rotating_phase*self.detuning_length_Pi/180
        
        self.X_gate(name = name + '_X1', waiting_time = waiting_time, refpoint = refpoint,
                    refpulse = None if refgate == None else refgate[-2]['pulse_name'])

        self.detuning(name = name+'_detuning', length = length, waiting_time = 1e-6, 
                      refpulse = self.pulses[-1]['pulse_name'])
        
        self.X_gate(name = name+'_X2', waiting_time = 1e-6, refpulse = self.pulses[-1]['pulse_name'])
        
    def __call__(self, **kw):
        
        return self
   

class CRotation_Gate(Two_Qubit_Gate):
    
    def __init__(self, name = 'CPhase_gate', control_qubit = None, target_qubit = None, 
                 rotating_phase = 180, refgate = None, refpoint = 'end', waiting_time = 0):
        
        super().__init__(name, control_qubit, target_qubit)
        
        length = rotating_phase*self.detuning_length_Pi/180
        
        self.detuning(name = name+'_detuning', length = length, waiting_time = waiting_time, refpoint = refpoint, 
                      refpulse = None if refgate == None else refgate[-2]['pulse_name'])
        
        self.X_Pi_gate(name = name+'_X', refpulse = self.pulses[-1]['pulse_name'], refpoint = 'center', waiting_time = 2e-6)
        
  