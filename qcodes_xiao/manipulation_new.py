# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:59:41 2017

@author: twatson
"""

import numpy as np

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import SquarePulse
from gate import Single_Qubit_Gate, Two_Qubit_Gate, CPhase_Gate, CNot_Gate, CRotation_Gate


class Manipulation2(Element):


    def __init__(self, name, pulsar, qubits = [], operations = {}, **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates

        super().__init__(name, pulsar, **kw)
        self.operations = {}

        self.refphase = {}       ##  {'Qubit_1': 0, 'Qubit_2': 0}  this is to keep track of the Z rotation

        self.total_time = 0     ## used for finally adding the gate voltage pulse

        self.VP_before = 250e-9
#        self.VP_after = 250e-9
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
        
    
    def make_single_qubit_gate(self, name, parameters):
        
        print(parameters)
        axis =  parameters['axis']
        degree =  parameters['degree']
        amplitude=  parameters['amplitude']
        length =  parameters['length']
        frequency_shift =  parameters['frequency_shift']
        waiting_time =  parameters['waiting_time']
        
        qubit =  parameters['qubit']
        refgate=  parameters['refgate']
        refpoint =  parameters['refpoint']       
        refpoint_new =  parameters['refpoint_new']
        
        refphase = self.refphase[qubit.name]
        
        # no idea yet  perhaps call the element.add() function but just to add the first pulse
        # and record the information of the last pulse
        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name
        
        if frequency_shift != 0 and not name.startswith('off'):
            tvals, wfs = self.waveforms()   ## 960?
            channel_id = qubit.microwave_gate['channel_I']
            Time = len(tvals[channel_id])*10e-9
            IQ_phase = 2*np.pi*Time/frequency_shift #if frequency_shift != 0 else 0
        else:
            IQ_phase =0
        refphase = self.refphase[qubit.name]
        
        
        
        single_qubit_gate = Single_Qubit_Gate(name = name, qubit = qubit, rotating_axis = axis,
                                              frequency_shift = frequency_shift, amplitude = amplitude, refphase = refphase,
                                              IQ_phase = IQ_phase)


        if axis[0]!=0 or axis[1]!=0:
            if axis[2]!=0:
                raise ValueError('should be either in X-Y plane or Z axis')
            else:
                single_qubit_gate.XY_rotation(degree = degree, length = length, waiting_time = waiting_time,
                                              refgate = None if refgate == None else self.operations[refgate], refpoint = refpoint)
                self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = single_qubit_gate)

        else:
            if axis[2] == 0:
                raise ValueError('should be either in X-Y plane or Z axis')
            else:
#               single_qubit_gate.Z_rotation(degree = 90)
                self.refphase[qubit.name] += degree                     ## Z rotation equals to change of refphase

        


        return True
 


       
    def make_CPhase(self, name, parameters):
         
        control_qubit =  parameters['control_qubit']
        target_qubit =  parameters['target_qubit']
        amplitude_target =  parameters['amplitude_target']
        refgate =  parameters['refgate']
        
        
        
        length =  parameters['length']
        amplitude_control=  parameters['amplitude_control']
        refpoint =  parameters['refpoint']
        waiting_time =  parameters['waiting_time']

         

        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name

        cphase_gate = CPhase_Gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
                                  amplitude_control = amplitude_control, amplitude_target = amplitude_target,
                                  length = length,
                                  refgate = None if refgate == None else self.operations[refgate],
                                  refpoint = refpoint, waiting_time = waiting_time,)

        self._add_all_pulses_of_qubit_gate(name = name, qubit_gate = cphase_gate)

        return True


   