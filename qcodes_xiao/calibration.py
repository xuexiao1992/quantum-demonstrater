# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:42:13 2017

@author: think
"""

import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.element import Element
from Gates import Single_Qubit_Gate#, Two_Qubit_Gate
from manipulation import Manipulation
from experiment import Experiment

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet
from qcodes.data.data_array import DataArray



class Calibration(Experiment):
    
    def __init__(self, name, qubits, awg, awg2, pulsar, **kw):                ## name = 'calibration_20170615_Xiao'...
        
        super().__init__(name, qubits, awg, awg2, pulsar, **kw)
        
        self.Pi_length = [qubit.Pi_pulse_length for qubit in qubits]
        self.X_length = 0
        self.Rabi_power = 0
        
#        self.qubits_name = qubits_name
        
#        self.calibration_sequence = Sequence()
        self.sweep_inside_sequence = False
        
        self.formatter = HDF5FormatMetadata()
        self.data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
        self.data_location = '2017-08-18/20-40-19_T1_Vread_sweep'

    
    
    def calibrate_qubit_fequency_by_continuous_wave(self, qubit, ):
        
        qubit_frequency = 0
        
        qubit.frequency = qubit_frequency
        
        return qubit_frequency
    
    
    def calibrate_qubit_frequency_by_single_pulse(self, qubit, frequency = [0,0,0], burst_time = [0,0,0]):      ## [start:end:steps]
        
        qubit_frequency = qubit.frequency
        
        if self.sweep_inside_sequence is False:
            
            Sweep_Frequency = parameter[frequency[0]:frequency[1]:frequency[2]]
            
            Sweep_BurstTime = parameter[burst_time[0]:burst_time[1]:burst_time[2]]
    
            LOOP = Loop(sweep_values = Sweep_Frequency).loop(sweep_values = Sweep_BurstTime).each(measured_parameter)
    
            data_set = LOOP.get_data_set(location = None, loc_record = {'name': 'Chevron Pattern', 'label': 'frequency-burst_time'}, io = self.data_IO,)
    
            data_set = Loop.run()
        
        elif self.sweep_inside_sequence is True:
            
            
        
        return qubit_frequency




    def calibrate_Rabi_frequency(self, qubit,):
        
        Rabi_frequency = 0
        
        self.Rabi_element()
        
        self.run_all()
        
        qubit.Rabi_frequency = Rabi_frequency
        
        return Rabi_frequency
    
    
    
    def run_Ramsey(self, name = 'Ramsey', sweep = 'yes'):
        
#        self.experiment['Ramsey'] = self.Ramsey_element()
#        self.sweep_matrix
        
        self.Ramsey_element()
        
        self.run_all(name = name)
        
#        self.update_calibrated_parameter()
        
        return True
    
    
    def run_Chevron(self, name = 'Chevron',):
        
        self.Chevron_element()
        
        self.run_all(name = name)
        
        return True
    
    
    
    
    def update_calibrated_parameter(self, ):
        
        return True
    
    
    def Chevron_element(self, name = 'Chevron', qubit = None, frequency = 0, length = 5e-7):
        
        ## here [frequency] will be the parameter sweeped
        
        ## here you need I/Q modulation to tune the frequency
        
        for i in range(len(self.sweep_matrix)):
            
            frequency = self.sweep_matrix[i]['frequency']
            
            chevron = Manipulation(name = 'Manipulation_%d'%i, qubits_name = self.qubits_name, pulsar = self.pulsar)
            
            chevron.add_X_Pi(name = name+'X1', length = length, qubit = self.qubit)
            
            self.element['Manipulation_%d'%i] = chevron
            
        return True
    
    
    
    def Rabi_element(self, name = 'Rabi', qubit = None, frequency = 0, length = 5e-7):
        
        ## here [length] or [power] will be the parameter sweeped
        ## [power] sweep is done by directly sending commands to Microwave VSG, not here
        ## [amplitude] is an alternative way to sweep the power, but maybe not used, [power] is more often used
        ## but if use [amplitude], then use I/Q modulation instead
        
        for i in range(len(self.sweep_matrix)):
            
            power = self.sweep_matrix[i]['power']
            
            length = self.sweep_matrix[i]['length']
            
            amplitude = self.sweep_matrix[i]['amplitude']
        
            rabi = Manipulation(name = 'Manipulation_%d'%i, qubits_name = self.qubits_name, pulsar = self.pulsar)
        
            rabi.add_X_Pi(name = name+'X1', length = length, qubit = self.qubit)
        
            self.element['Manipulation_%d'%i] = chevron
            
        return True
        
    
    
    def Ramsey_element(self, name = 'Ramsey', frequency = None, length = 5e-7, waiting_time = 10e-6):
        
        ## here [frequency] will be the parameter sweeped
        
        ## here you need I/Q modulation to tune the frequency
        
        for i in range(len(self.sweep_matrix)):
            
            frequency = self.sweep_matrix[i]['frequency']
        
            ramsey = Manipulation(name = 'Manipulation_%d'%i, qubits_name = self.qubits_name, pulsar = self.pulsar)
        
            ramsey.add_X(name = name+'X1', length = length, qubit = self.qubit)
        
            ramsey.add_X(name = name+'X2', length = length, 
                         qubit = self.qubit, refgate = name+'X1', waiting_time = waiting_time)
        
            self.element['Manipulation_%d'%i] = ramsey
        
#        self.experiment['Ramsey'] = ramsey
#        self.element.append(ramsey)
        
        return True
    
    def qubit_rough_frequency_element(self,):
        
        return True
    
#    
#    def Sweep(self, start_value, stop_value, points):
#        
#        sweep_array = np.linspace(start_value, stop_value, points)
#        
#        return True