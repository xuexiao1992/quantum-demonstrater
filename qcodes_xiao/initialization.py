# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:33:19 2017

@author: think
"""


import numpy as np

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import SquarePulse

from Gates import Single_Qubit_Gate, Two_Qubit_Gate


class Initialization(Element):
    
    def __init__(self, name, qubits_name = [], **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates
         
        super().__init__(name, **kw)
        
        self.operations = {}
                  
        self.refphase = {}       ##  {'Qubit_1': 0, 'Qubit_2': 0}  this is to keep track of the Z rotation
    
        self.channel = None
        
        self.length = 0
        for qubitname in qubits_name:
             self.refphase[qubitname] = 0
             
    
    def unload(self, qubit):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='UnLoad', channel=self.channel, amplitude=0.03, start=0, length=10e-6), name='UnLoad')
        
        return True
    
    
    def shuttle(self, qubit, neighbor):                   ## shuttle a electron in one dot to its neighbor dot
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.channel_neighbor = neighbor.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='ShuttleOut', channel=self.channel, amplitude=0.03, length=10e-6), name='ShuttleOut')
        
        self.add(SquarePulse(name='ShuttleIn', channel=self.channel_neighbor, amplitude=0.03, length=10e-6), 
                 name='ShuttleIn', start = 0, refpulse = 'ShuttleIn', refpoint = 'start')
        
        return True
    
    
    def fast_relax(self, qubit):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='Initialize', channel='ch1', amplitude=0.05, length=5e-6), name='Initialize')
        
        return True
    
    def Elzerman(self, qubit):
    
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='Initialize', channel='ch1', amplitude=0.05, length=5e-6), name='Initialize')
        
        return True
        
    
#    def run_all(self, qubit):
#        self.unload()
#        self.initialize()
#        return True