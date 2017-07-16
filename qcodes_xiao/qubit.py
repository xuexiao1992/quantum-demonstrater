# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:12:15 2017

@author: think
"""

from qcodes.instrument.base import Instrument

class Qubit(Instrument):
    
    def __init__(self, name, gates = [], **kw):
        super().__init__(name, **kw)
#        self.msmt_suffix = '_' + name  # used to append to measuremnet labels
#        self._operations = {}
#        self.add_parameter('operations',
#                           docstring='a list of all operations available on the qubit',
#                           get_cmd=self._get_operations)
        self.T1 = 0
        self.T2 = 0
        self.frequency = 0
        self.Pi_pulse_length = 50e-9
        
        
        
        self.Rabi_frequency = 0
        self.gates = {}      #{number:voltage}
        
        self.neighbor = {}               ##        neighbor quantum dots
        
#        self.refphase = 0               ##          used for Z rotation
        
#        for gate_name in gates:
#            self.gates[gate_name] = {
#                    'name': 0,
#                    'number': 0,
#                    'voltage': 0,
#                    'function': 0,             ##   confinement/plunger
#                    'channel': 0,              ##   awg_channel
#                    'microwave': 0,            ##   0/1    no/yes
#                    }
         
    
    
    def define_gate(self, gate_name, gate_number, gate_function = 'confinement', channel_DC = None, 
                    microwave = 0, channel_I = None, channel_Q = None, channel_PM = None, channel_FM = None, 
                    channel_VP = None):
        
        self.gates[gate_name] = {
                'name': gate_name,
                'number': gate_number,
#                'voltage': gate_voltage,
                'function': gate_function,          ## confinement/plunger
#                'channel_DC': channel_DC,           ## DC_channel
                'microwave': microwave,             ## 0/1 no/tes
                'voltage': 0
                }
        
        if gate_function == 'plunger':
            self.plunger_gate = self.gates[gate_name]
            self.plunger_gate['channel_VP'] = channel_VP
        
        if microwave == 1:
            self.microwave_gate = self.gates[gate_name]
            self.microwave_gate['channel_I'] = channel_I
            self.microwave_gate['channel_Q'] = channel_Q
            self.microwave_gate['channel_PM'] = channel_PM
            self.microwave_gate['channel_FM'] = channel_FM
        return True
            
#    def define_neighbor(self, neighbor_qubit, CPhase_time, CPhase_detuning, CRotation_freq_up, CRotation_freq_down):
#        self.neighbor[neighbor_name] = {
#                'position': neighbor_position,
#                'shared_gates': {},
#                'CPhase_time': CPhase_time,
#                'CPhase_detuning': CPhase_detuning,
#                'CRotation_freq_up': CRotation_freq_up,
#                'CRotation_freq_down': CRotation_freq_down
#                }
#        return True
    
    
    
    
#    def set_gate_channel(self, gate_name):
#        return self.gates[gate_name]['channel']
    
#    def set_gate_voltage(self, gate_name):                  ## remain empty, will be completed with ivvi
#        return self.gates[gate_name]['volatage']
#
#    
#    def get_gate_voltage(self, gate_name):
#        return self.gates[gate_name]['volatage']
#    
#    def get_T1(self):
#        return self.T1
#    
#        
#    def get_T2(self):
#        return self.T2
#    
#    def get_freqency(self):
#        return self.frequency
#    
#    def get_Rabi(self):
#        return self.Rabi
#    
#    def get_state(self):
#        return self.state
