# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:59:54 2017

@author: think
"""

import numpy as np

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import SquarePulse

from Gates import Single_Qubit_Gate, Two_Qubit_Gate

class Readout(Element):
    
    def __init__(self, name, qubits_name = [], **kw):            ## operation is a set of objects: basic one(two) qubit(s) gates
         
        super().__init__(name, **kw)
         
        self.operations = {}
                  
        self.refphase = {}       ##  {'Qubit_1': 0, 'Qubit_2': 0}  this is to keep track of the Z rotation
    
        self.channel = None
        
        for qubitname in qubits_name:
             self.refphase[qubitname] = 0
             
             
             
    def Prep_Read(self, qubit):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='Prep_Read', channel=self.channel, amplitude=0.03, length=10e-6), name='Prep_Read')
        
        return True
    
    
    
    def shuttle(self, qubit, neighbor):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.channel_neighbor = neighbor.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='ShuttleOut', channel=self.channel, amplitude=0.03, length=10e-6), name='ShuttleOut')
        
        self.add(SquarePulse(name='ShuttleIn', channel=self.channel_neighbor, amplitude=0.03, length=10e-6), 
                 name='ShuttleIn', refpulse = 'ShuttleIn', refpoint = 'start')
        
        return True
    
    def Elzerman(self, qubit):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='Elzerman', channel=self.channel, amplitude=0.03, length=10e-6), name='Elzerman')
        
        return True
    
    def Pauli_Blockade(self, qubit):
        
        self.channel = qubit.plunger_gate['channel_VP']
        
        self.add(SquarePulse(name='Pauli_Blockade', channel=self.channel, amplitude=0.03, length=10e-6), name='Pauli_Blockade')
        
        return True
    
