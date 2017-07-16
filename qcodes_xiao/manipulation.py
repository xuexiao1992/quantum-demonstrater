# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:55:59 2017

@author: think
"""

import numpy as np

from pycqed.measurement.waveform_control.element import Element
from Gates import Single_Qubit_Gate, Two_Qubit_Gate


class Manipulation(Element):
    
    
    def __init__(self, name, qubits_name = [], operations = {}, **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates
         super().__init__(name, **kw)
         self.operations = {}
                  
         self.refphase = {}       ##  {'Qubit_1': 0, 'Qubit_2': 0}  this is to keep track of the Z rotation
                  
         for qubitname in qubits_name:
             self.refphase[qubitname] = 0
             

             
    def add_single_qubit_gate(self, name = None, qubit = None, axis = [1,0,0], degree = 90,
                              frequency = None, refphase = 0,
                              refgate = None, refpoint = 'end', waiting_time = 0, refpoint_new = 'start'):
        # no idea yet  perhaps call the element.add() function but just to add the first pulse
        # and record the information of the last pulse
        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name
        
        single_qubit_gate = Single_Qubit_Gate(name = name, qubit = qubit, aixs = axis, 
                                              frequency = frequency, refphase = refphase)
        
        if axis[0] == 1 or axis[1] == 1:
            single_qubit_gate.XY_rotation(degree = degree, 
                                          refgate = None if refgate == None else self.operations[refgate])
        else:
#            single_qubit_gate.Z_rotation(degree = 90)
            self.refphase[qubit.name] += degree                     ## Z rotation equals to change of refphase
        
        self.operations[name] = single_qubit_gate.pulses            ## add this operation into the dic, in order to track total numbers of operation
                                                                    ## and also could name each pulse
       
        for i in range(len(single_qubit_gate.pulses)):              ## need to modify, not use i as number but 
            PULSE = single_qubit_gate.pulses[i]                     ## directly go through the element
            pulse = PULSE['pulse']
            pulse_name = PULSE['pulse_name']
            start = PULSE['waiting']
            refpulse = PULSE['refpulse']
            refpoint = PULSE['refpoint']
#            position = PULSE['position']
#            pulse_name = pulse.name
            self.add(pulse = pulse, name = pulse_name, start = start, refpulse = refpulse, refpoint = refpoint)
#            
            

                
##      this part above is to add all the pulses in the Gate into the element  
        
        return True
    
    
    
    
    
    def add_two_qubit_gate(self, name = None, control_qubit = None, target_qubit = None, 
                           refgate = None, refpoint = 'end',
                           waiting_time = 0, refpoint_new = 'start'):
        
        if name in self.operations.keys():
            raise NameError('Name already used')            ## need to stop the program or maybe randomly give a name
        
        two_qubit_gate = Two_Qubit_Gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit)
        
        self.operations[name] = two_qubit_gate.pulses 
        
        for i in range(len(two_qubit_gate.pulses)):              ## need to modify, not use i as number but 
            PULSE = two_qubit_gate.pulses[i]                     ## directly go through the element
            pulse = PULSE['pulse']
            pulse_name = PULSE['pulse_name']
            start = PULSE['waiting']
            refpulse = PULSE['refpulse']
            refpoint = PULSE['refpoint']
#            position = PULSE['position']
#            pulse_name = pulse.name
            self.add(pulse = pulse, name = pulse_name, start = start, refpulse = refpulse, refpoint = refpoint)
        
        
        for pulse in two_qubit_gate.pulses:
            self.add(pulse = pulse['pulse'], name = pulse['name'],
                     start = pulse['waiting'], refpulse = pulse['waiting'], refpoint = pulse['refpoint'])
                 
        return True







    def add_X_Pi(self, name = None, qubit = None, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start'):
        # these are some quick calls for some often-used gate
        
        refphase = self.refphase[qubit.name]
        
        self.add_single_qubit_gate(name = name, qubit = qubit, degree = np.pi, 
                                   refgate = refgate, refpoint = refpoint, refphase = refphase, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)
        return True
    
    
    def add_Y_Pi(self, name = None, qubit = None, refgate = None, refpoint = 'end', 
                  waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[qubit.name]
        
        self.add_single_qubit_gate(name = None, qubit = qubit, axis = [0, 1, 0], degree = np.pi, 
                                   refgate = None, refpoint = 'end', refphase = refphase, 
                                   waiting_time = 0, refpoint_new = refpoint_new)
        
        return True
    
    def add_X(self, name = None, qubit = None, refgate = None, refpoint = 'end', 
                  waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[qubit.name]
        
        self.add_single_qubit_gate(name = name, qubit = qubit, degree = np.pi/2, 
                                   refgate = refgate, refpoint = refpoint, refphase = refphase, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)
        
        return True
    
    def add_Y(self, name = None, qubit = None, refgate = None, refpoint = 'end', 
                 waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[qubit.name]
        
        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 1, 0], degree = np.pi/2, 
                                   refgate = None, refpoint = 'end', refphase = refphase, 
                                   waiting_time = 0, refpoint_new = refpoint_new)
        
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
#    
    def add_CRotation(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end', 
                      waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[target_qubit.name]
        
        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit, 
                                refgate = None, refpoint = 'end',
                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
        
        return True
    
    def add_CPhase(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end', 
                   waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[target_qubit.name]
        
        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit, 
                                refgate = None, refpoint = 'end',
                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
        
        return True
    
    def add_CNot(self, name = None, control_qubit = None, target_qubit = None, refgate = None, refpoint = 'end', 
                  waiting_time = 0, refpoint_new = 'start'):
        
        refphase = self.refphase[target_qubit.name]
        
        self.add_two_qubit_gate(name = name, control_qubit = control_qubit, target_qubit = target_qubit,
                                refgate = None, refpoint = 'end',
                                refphase = refphase, waiting_time = 0, refpoint_new = refpoint_new)
        
        return True
        
        

    
    