# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:32:40 2017

@author: twatson
"""
from manipulation_new import Manipulation

class Circuit:
    
    
    def __init__(self, name, pulsar, qubits = [], **kw):
         
         self.circuit = {}
         self.name = name
         self.pulsar = pulsar
         self.qubits = qubits

         
         
    def add_single_qubit_gate(self, name = None, qubit = None, axis = [1,0,0], degree = 90,
                              amplitude = 1, length = 50e-9, frequency_shift = 0,
                              refgate = None, refpoint = 'end', waiting_time = 0, refpoint_new = 'start'):
        #add to some dictionary, name, type of gate, gate parameters.  This dictionary can be change for a sweep...
        
        parameters = {'gate': 'single_qubit_gate', 'qubit' : qubit, 'axis' :axis, 'degree' : degree, 'amplitude' : amplitude,
                      'length' :length, 'frequency_shift' : frequency_shift, 'refgate': refgate,
                      'refpoint': refpoint, 'waiting_time' : waiting_time, 'refpoint_new' : refpoint_new}
        
        self.circuit.update({name : parameters})
        
        
    def add_CPhase(self, name = None, control_qubit = None, target_qubit = None, length = 0,
                        amplitude_control = 0, amplitude_target = 0, refgate = None, refpoint = 'end',
                        waiting_time = 0, refpoint_new = 'start'):
         
        parameters = {'gate': 'CPhase', 'control_qubit' : control_qubit, 'target_qubit' :target_qubit, 'length' : length,
                      'amplitude_control' : amplitude_control, 'amplitude_target' :amplitude_target, 'refgate' : refgate, 'refpoint' : refpoint, 
                       'waiting_time' : waiting_time, 'refpoint_new' : refpoint_new}
        
        self.circuit.update({name : parameters})
        return True        
    
    
    
    def add_X_Pi(self, name, qubit, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start', **kw):
        # these are some quick calls for some often-used gate
        
        length = kw.pop('length', qubit.Pi_pulse_length)

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [1, 0, 0], degree = 180,
                                   length = length, refgate = refgate, refpoint = refpoint,
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)
        return True


    def add_Y_Pi(self, name, qubit, refgate = None, refpoint = 'end',
                  waiting_time = 0, refpoint_new = 'start', **kw):


        
        length = kw.pop('length', qubit.Pi_pulse_length)

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 1, 0], degree = 180,
                                   length = length, refgate = refgate, refpoint = refpoint, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new)

        return True

    def add_X(self, name, qubit, refgate = None, refpoint = 'end',
              waiting_time = 0, refpoint_new = 'start', **kw):

        
        length = kw.pop('length', qubit.halfPi_pulse_length)

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [1, 0, 0], degree = 90,
                                   length = length, refgate = refgate, refpoint = refpoint, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new, **kw)

        return True
    

    def add_Y(self, name, qubit, refgate = None, refpoint = 'end',
                 waiting_time = 0, refpoint_new = 'start', **kw):


        
        length = kw.pop('length', qubit.halfPi_pulse_length)

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 1, 0], degree = 90, 
                                   length = length, refgate = refgate, refpoint = refpoint, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new, **kw)

        return True
    

    def add_Z(self, name = None, qubit = None, degree = 0, refgate = None, refpoint = 'end',
                waiting_time = 0, refpoint_new = 'start', **kw):

        self.add_single_qubit_gate(name = name, qubit = qubit, axis = [0, 0, 1], degree = 90, 
                                   length = 0, refgate = refgate, refpoint = refpoint, 
                                   waiting_time = waiting_time, refpoint_new = refpoint_new, **kw)

        return True    
    
    
    def make_circuit(self,):
        
        manip_elem = Manipulation('test', self.pulsar, qubits = self.qubits)
        for name in self.circuit:
            if self.circuit[name]['gate'] == 'single_qubit_gate':
               manip_elem.make_single_qubit_gate(name, self.circuit[name])
               print('making single qubit gate')
            if self.circuit[name]['gate'] == 'CPhase':
               manip_elem.make_CPhase(name,  self.circuit[name])
               print('making single qubit gate')               
        return manip_elem
         
         