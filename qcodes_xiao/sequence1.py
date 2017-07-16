# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:00:48 2017

@author: think
"""

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.element import Element


class Manipulation(Sequence):
    
    
    def __init__(self, name, qubits, operations = [], **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates
         super().__init__(name, **kw)
         self.operations = operations
         for operation in self.operations:
             self.append_element(element = operation)
         
        
    def benchmarking
        
        
        
class manipulation1(Manipulation):
    
     def __init__(self, name, qubits, operations, **kw):
         super().__init__(name, qubits, operations, **kw)
     
        
     def
    
    