# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:35:17 2017

@author: think
"""


import numpy as np
from scipy import constants as C

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse

from qubit import Qubit


class Gate:

    def __init__(self, name, **kw):

        self.name = name                                 ## name of a gate, e.g. 'X_Q1' 'CZ_Q12'
        self.qubit_name = 'qubit'
        self.qubits = []
<<<<<<< HEAD


##      Gate will be an object that consists all the pulse information in the operation,
=======


##      Gates will be an object that consists all the pulse information in the operation,
>>>>>>> b7918d212502aadde6e5a17a361c1a522f6003bd
##      but it will just store the pulses in information and the links in time domain
##      between different pulses, but not adding it into an Element



class Single_Qubit_Gate(Gate):

    def __init__(self, name, qubit, rotating_axis = (1, 0, 0), frequency = None, refphase = 0, **kw):
        super().__init__(name, **kw)
        self.qubit = qubit.name
        self.frequency = qubit.frequency if frequency == None else frequency
        self.channel_I = qubit.microwave_gate['channel_I']
        self.channel_Q = qubit.microwave_gate['channel_Q']
        self.channel_PM = qubit.microwave_gate['channel_PM']
        self.channel_FM = qubit.microwave_gate['channel_FM']
        self.channel_VP = qubit.plunger_gate['channel_VP']
#        self.add(CosPulse(channel = self.channel, name = 'first pulse', frequency = qubit.frequency,
#                          amplitude = 0, length = 0,))

#        self.add(SquarePulse(channel = self.channel, name = 'another pulse', frequency = qubit.frequency,
#                          amplitude = 0, length = 0, refpulse='first pulse', refpoint='start'))
        self.axis = np.array(rotating_axis)
#        self.degree = degree
        self.Pi_pulse_length = qubit.Pi_pulse_length
        self.voltage_pulse_length = 0
        self.refphase = refphase*np.pi/180
#        self.pulses = {
#                #  'microwave': None,
#                ##  'voltage': None
#                }      ## this will be the returned value and used in the manipulation object

        self.pulses = [None, None, None]            ## [microwave1_I, microwave1_Q, voltage, microwave2_I, microwave2_Q]

#        if (self.axis**2).sum != 1:
#            self.axis =self.axis/np.sqrt((self.axis**2).sum)            ## normalize the axis





    def XY_rotation(self, degree = 90, waiting_time = 0, refgate = None):
#        global phase
#        IQ_Modulation = self.frequency
        microwave_pulse_I = SquarePulse(channel = self.channel_I, name = '%s_microwave_pulse_I'%self.name,
                   amplitude = np.cos(self.refphase), length = degree*self.Pi_pulse_length/180)

        microwave_pulse_Q = SquarePulse(channel = self.channel_Q, name = '%s_microwave_pulse_Q'%self.name,
                   amplitude = np.sin(self.refphase), length = degree*self.Pi_pulse_length/180)

        voltage_pulse = SquarePulse(channel = self.channel_VP, name = '%s_voltage_pulse'%self.name,
                   amplitude = 1, length = degree*self.Pi_pulse_length/180)




        self.pulses[0] = {
                'pulse': voltage_pulse,
                'pulse_name': voltage_pulse.name,
                'refpulse': None if refgate == None else refgate[0]['pulse_name'],
                'refpoint': 'end',
                'waiting': waiting_time
                }



        self.pulses[1] = {
                'pulse': microwave_pulse_I,
                'pulse_name': microwave_pulse_I.name,
                'refpulse': None if refgate == None else refgate[-1]['pulse_name'],                   ## name of the refpulse
                'refpoint': 'end',
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




class Two_Qubit_Gate(Gate):

    def __init__(self, name, control_qubit, target_qubit, operation = '', **kw):
        super().__init__(name, **kw)


        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.frequency = target_qubit.frequency
        self.refphase = 0
        self.Pi_pulse_length = qubit.Pi_pulse_length

        self.channel_VP1 = control_qubit.plunger_gate['channel_VP']
        self.channel_VP2 = target_qubit.plunger_gate['channel_VP']
        self.channel_I = target_qubit.microwace_gate['channel_I']
        self.channel_Q = target_qubit.microwace_gate['channel_Q']
        self.channel_FM = target_qubit.microwace_gate['channel_FM']
        self.channel_PM = target_qubit.microwace_gate['channel_PM']

        self.pulses = []




    def CRotation_gate(self, exchanging_time, refgate):
        self.detuning(refpulse = refgate[-1]['pulse_name'])

        self.X_Pi_gate(refpulse = self.pulses[-1]['pulse_name'])
        return True

    def CPhase_gate(self, rotating_phase, refgate):
        self.detuning(refpulse = refgate[-1]['pulse_name'])
        return 0

    def CNot_gate(self, exchanging_time, delay_time, refgate):
        self.X_gate(refpulse = refgate[-1]['pulse_name'])
        self.CPhase_gate(refpulse = self.pulses[-1]['pulse_name'])
        self.X_gate(refpulse = self.pulses[-1]['pulse_name'])
        return True


    def detuning(self, name = 'detuning', waiting_time = 0, refpulse = None):
        detuning_pulse = SquarePulse(channel = self.channel_VP1, name = name,
                          amplitude = 0, length = 0, start = waiting_time, refpulse = refpulse)

        detuning = {
                'pulse': detuning_pulse,
                'pulse_name': detuning_pulse.name,
                'refpulse': None if refpulse == None else refpulse[0]['pulse_name'],
                'refpoint': 'end',
                'waiting': waiting_time
                }

        self.pulses.append(detuning)

        return 0


    def XY_rotation(self, degree = 90, waiting_time = 0, refpulse = None):

        microwave_pulse_I = SquarePulse(channel = self.channel_I, name = '%s_microwave_pulse_I'%self.name,
                   amplitude = np.cos(self.refphase), length = degree*self.Pi_pulse_length/180)

        microwave_pulse_Q = SquarePulse(channel = self.channel_Q, name = '%s_microwave_pulse_Q'%self.name,
                   amplitude = np.sin(self.refphase), length = degree*self.Pi_pulse_length/180)

        component_I = {
                'pulse': microwave_pulse_I,
                'pulse_name': microwave_pulse_I.name,
                'refpulse': refpulse,                   ## name of the refpulse
                'refpoint': 'end',
                'waiting': waiting_time
                }

        component_Q = {
                'pulse': microwave_pulse_Q,
                'pulse_name': microwave_pulse_Q.name,
                'refpulse': '%s_microwave_pulse_I'%self.name,
                'refpoint': 'start',
                'waiting': 0
                }

        self.pulses.append(component_I)
        self.pulses.append(component_Q)




    def X_gate(self, name = 'X_halfPi', waiting_time = 0, refpulse = None):

        self.XY_rotation(name = name, degree = 90, waiting_time = waiting_time, refpulse = refpulse)

        return 0

    def X_Pi_gate(self, name = 'X_Pi', waiting_time = 0, refpulse = None):

        self.XY_rotation(name = name, degree = 180, waiting_time = waiting_time, refpulse = refpulse)

        return 0

    def Y_gate(self, name = 'Y_halfPi', waiting_time = 0, refpulse = None):

        self.XY_rotation(name = name, degree = 90, waiting_time = waiting_time, refpulse = refpulse)

        return 0

    def Y_Pi_gate(self, name = 'Y_Pi', waiting_time = 0, refpulse = None):

        self.XY_rotation(name = name, degree = 180, waiting_time = waiting_time, refpulse = refpulse)

        return 0


#
#class Voltage_Pulse(Gates):
#
