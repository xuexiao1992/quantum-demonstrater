# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:55:59 2017

@author: X.X
"""

import numpy as np

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from gate import Single_Qubit_Gate, Two_Qubit_Gate, CPhase_Gate, CNot_Gate, CRotation_Gate


class Manipulation(Element):


    def __init__(self, name, pulsar, qubits = [], operations = {}, **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates

        super().__init__(name, pulsar, **kw)
        self.operations = {}

        self.refphase = {}       ##  {'Qubit_1': 0, 'Qubit_2': 0}  this is to keep track of the Z rotation

        self.total_time = 0     ## used for finally adding the gate voltage pulse

        self.VP_before = 250e-9
        self.VP_after = 250e-9
        self.length = 0

        self.qubits = qubits
        self.refphase = {qubit.name: 0 for qubit in self.qubits}


    def _add_all_pulses_of_qubit_gate(self, name = None, qubit_gate = None):

        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name


        self.operations[name] = qubit_gate.pulses            ## add this operation into the dic, in order to track total numbers of operation
                                                                    ## and also could name each pulse

        for i in range(len(qubit_gate.pulses)):              ## need to modify, not use i as number but
            PULSE = qubit_gate.pulses[i]                     ## directly go through the element
            pulse = PULSE['pulse']
            pulse_name = PULSE['pulse_name']
            start = PULSE['waiting']
            refpulse = PULSE['refpulse']
            refpoint = PULSE['refpoint']
            self.add(pulse = pulse, name = pulse_name, start = start, refpulse = refpulse, refpoint = refpoint)
##      this part above is to add all the pulses in the Gate into the element
        return True




    def add_single_qubit_gate(self, name = None, qubit = None, axis = [1,0,0], degree = 90,
                              amplitude = None, length = 50e-9, frequency_shift = 0, refphase = 0,
                              refgate = None, refpoint = 'end', waiting_time = 0, refpoint_new = 'start'):
        # no idea yet  perhaps call the element.add() function but just to add the first pulse
        # and record the information of the last pulse
        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name
        tvals, wfs = self.waveforms()   ## 960?
        channel_id = qubit.microwave_gate['channel_I']
        time = len(tvals[channel_id])*10e-9
        IQ_phase = 2*np.pi*time/frequency_shift if frequency_shift != 0 else 0
        single_qubit_gate = Single_Qubit_Gate(name = name, qubit = qubit, rotating_axis = axis,
                                              frequency_shift = frequency_shift, amplitude = amplitude, refphase = refphase,
                                              IQ_phase = IQ_phase)

        if axis[0]!=0 or axis[1]!=0:
            if axis[2]!=0:
                raise ValueError('should be either in X-Y plane or Z axis')
            else:
                single_qubit_gate.XY_rotation(degree = degree, length = length, waiting_time = waiting_time,
                                              refgate = None if refgate == None else self.operations[refgate], refpoint = refpoint)

        else:
            if axis[2] == 0:
                raise ValueError('should be either in X-Y plane or Z axis')
            else:
#               single_qubit_gate.Z_rotation(degree = 90)
                self.refphase[qubit.name] += degree                     ## Z rotation equals to change of refphase

        self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = single_qubit_gate)


        return True




    def add_CPhase(self, name = None, control_qubit = None, target_qubit = None, length = 0,
                   amplitude_control = 0, amplitude_target = 0, refgate = None, refpoint = 'end',
                   waiting_time = 0, refpoint_new = 'start'):

        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name

        cphase_gate = CPhase_Gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
                                  amplitude_control = amplitude_control, amplitude_target = amplitude_target,
                                  length = length,
                                  refgate = None if refgate == None else self.operations[refgate],
                                  refpoint = refpoint, waiting_time = waiting_time,)

        self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = cphase_gate)

        return True





    def add_CNot(self, name = None, control_qubit = None, target_qubit = None,
                   refgate = None, refpoint = 'end',
                   waiting_time = 0, refpoint_new = 'start'):

        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name

        cnot_gate = CNot_Gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
                              refgate = None if refgate == None else self.operations[refgate],
                              refpoint = refpoint, waiting_time = waiting_time,)

        self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = cnot_gate)

        return True




    def add_CRotation(self, name = None, control_qubit = None, target_qubit = None,
                   refgate = None, refpoint = 'end',
                   waiting_time = 0, refpoint_new = 'start'):

        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name

        crot_gate = CRotation_Gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
                                   refgate = None if refgate == None else self.operations[refgate],
                                   refpoint = refpoint, waiting_time = waiting_time,)

        self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = crot_gate)

        return True






    def add_X_Pi(self, name = None, qubit = None, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start'):
        # these are some quick calls for some often-used gate

        refphase = self.refphase[qubit.name]

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [1, 0, 0], degree = 180,
                                   refgate = refgate, refpoint = refpoint, refphase = refphase,
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)
        return True


    def add_Y_Pi(self, name = None, qubit = None, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start'):

        refphase = self.refphase[qubit.name]

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 1, 0], degree = 180,
                                   refgate = refgate, refpoint = refpoint, refphase = refphase,
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)

        return True

    def add_X(self, name = None, qubit = None, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start'):

        refphase = self.refphase[qubit.name]

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [1, 0, 0], degree = 90,
                                   length = qubit.halfPi_pulse_length, refgate = refgate, refpoint = refpoint, refphase = refphase,
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)

        return True

    def add_Y(self, name = None, qubit = None, refgate = None, refpoint = 'end',
                 waiting_time = 0, refpoint_new = 'start'):

        refphase = self.refphase[qubit.name]

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 1, 0], degree = 90, length = qubit.halfPi_pulse_length,
                                   refgate = refgate, refpoint = refpoint, refphase = refphase,
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)

        return True

    def add_Z(self, name = None, qubit = None, degree = 0, refgate = None, refpoint = 'end',
                waiting_time = 0, refpoint_new = 'start'):

        self.refphase[qubit.name] += degree


        return True

#    def add_Arbitrary_rotation(self, name = '', qubit, axis = [1,0,0], refgate = None, refpoint = 'end',
#                               waiting_time = 0, refpoint_new = 'start'):
#
#        return True
#
##
#    def add_CRotation(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end',
#                      waiting_time = 0, refpoint_new = 'start'):
#
#        refphase = self.refphase[target_qubit.name]
#
#        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
#                                refgate = None, refpoint = 'end',
#                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
#
#        return True

#    def add_CPhase(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end',
#                   waiting_time = 0, refpoint_new = 'start'):
#
#        refphase = self.refphase[target_qubit.name]
#
#        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
#                                refgate = None, refpoint = 'end',
#                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
#
#        return True

#    def add_CNot(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end',
#                  waiting_time = 0, refpoint_new = 'start'):
#
#        refphase = self.refphase[target_qubit.name]
#
#        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
#                                refgate = None, refpoint = 'end',
#                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
#
#        return True
