# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:42:13 2017

@author: think
"""


import numpy as np
from scipy.optimize import curve_fit

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.sequence import Sequence
from gate import Single_Qubit_Gate#, Two_Qubit_Gate
from manipulation import Manipulation
from sequencer import Sequencer
#from initialize import Initialize
#from readout import Readout
from qubit import Qubit
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse

from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray

import stationF006
#from stationF006 import station
from copy import deepcopy
from manipulation_library import Ramsey
from experiment_version2 import Experiment
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
import time
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability, set_digitizer


#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
#    x_new = x/1e6
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#%%



class Calibration(Experiment):
    
    def __init__(self, name, qubits, awg, awg2, pulsar, **kw):                ## name = 'calibration_20170615_Xiao'...
        
        super().__init__(name, qubits, awg, awg2, pulsar, **kw)
        
        self.Pi_length = [qubit.Pi_pulse_length for qubit in qubits]
        self.X_length = 0
        self.Rabi_power = 0
        
#        self.qubits_name = qubits_name
        
#        self.calibration_sequence = Sequence()
        self.sweep_inside_sequence = False
        
#        self.formatter = HDF5FormatMetadata()
        self.data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
        self.calibration_data_location = self.data_location+'_calibration'
        
        self.data_set = None
        
        self.calibration = 'Ramsey'
        self.experiment = 'AllXY'
        
        self.calibrated_parameter = None
#        self.dig = digitizer_param(name='digitizer', mV_range = 1000, memsize=4e9, seg_size=seg_size, posttrigger_size=posttrigger_size)
        

    def set_sweep_with_calibration(self, repetition = False, plot_average = False, **kw):
        
        self.plot_average = plot_average
        
        for i in range(self.qubit_number):
            d = i if i == 1 else 2
            self.plot.append(QtPlot(window_title = 'raw plotting qubit_%d'%d, remote = False))
#            self.plot.append(MatPlot(figsize = (8,5)))
            if plot_average:
#                self.average_plot.append(QtPlot(window_title = 'average data qubit_%d'%d, remote = False))
                self.average_plot.append(MatPlot(figsize = (8,5)))
        
        for seq in range(len(self.sequencer)):
            self.sequencer[seq].set_sweep()
            self.make_all_segment_list(seq)
        
        if repetition is True:
            count = kw.pop('count', 1)

            Count_Calibration = StandardParameter(name = 'Count', set_cmd = self.function)
            Sweep_Count_Calibration = Count_Calibration[1:2:1]
            
            Count = StandardParameter(name = 'Count', set_cmd = self.delete)
            Sweep_Count = Count[1:count+1:1]
            
            self.digitizer, self.dig = set_digitizer(self.digitizer, len(self.X_sweep_array), self.qubit_number, self.seq_repetition)
            
            loop1 = Loop(sweep_values = Sweep_Count_Calibration).each(self.dig)
            
            Loop_Calibration = loop1.with_bg_task(bg_final_task = self.update_calibration,)
            calibrated_parameter = update_calibration
            if self.Loop is None:
                self.Loop = Loop(sweep_values = Sweep_Count).each(calibrated_parameter, self.dig)
            else:
                raise TypeError('calibration not set')
                
        return True
    
    def delete(self, x):
        
        self.close()
        
        return True
    
    def switch_sequencer(self, experiment = 'AllXY'):
        
        sequencer = self.find_sequencer(experiment)
        self.load_sequence(experiment)
        self.digitizer, d = set_digitizer(self.digitizer, sequencer.dimension_1, self.qubit_number, self.seq_repetition)
        return True
    
    def update_calibration(self,):
        
#        self.calibrated_parameter = StandardParameter(name = 'Count', get_cmd = self.calibrate_by_ramsey)

        frequency = self.calibrate_by_Ramsey()
        
#        self.switch_sequencer(self.experiment)
        
        return frequency
    
    def set_calibration(self,)
    def calibrate_by_Ramsey(self,):
        
        
        DS = self.calculate_average_data(measurement = 'calibration')

        x = DS.frequency_shift_set.ndarray
        y = DS.digitizerqubit_1.ndarray
        
        pars, pcov = curve_fit(Func_Gaussian, x, y,)
        frequency_shift = pars[1]
        
        frequency = self.vsg2.frequency()+frequency_shift
        self.vsg2.frequency(frequency)
        
        return frequency
    
    def run_experiment_with_calibration(self, calibration, experiment):
        
        calibration_sequencer = self.find_sequencer(calibration)
        
        measurement_sequencer = self.find_sequencer(experiment)
        
        self.load_sequence(calibration)
        
        self.close()
        
        self.load_sequence(experiment)
        self.close()
        
        return True


    def set_calibration_sweep(self, **kw):
        
        real_time_calibration = True
        
        def update_parameter(seq_num, x):
            
            return True
        
        def switch_sequence(seq_num, x):
            
            self.close()
            
            self.load_sequence(measurement = '1')
            
            Count = StandardParameter(name = 'Count', set_cmd = self.function)
            Sweep_Count = Count[1:2:1]
            Loop = Loop(sweep_values = Sweep_Count).each(self.dig)
            DS = Loop.run()
            DSP = convert_to_probability(DS,0.025,self.qubit_number,self.seq_repetition,'cal', sweep_array = self.X_sweep_array)
            self.close()
            
            self.load_sequence(measurement = '2')
            
            return True
        
        if real_time_calibration is True:
            count = kw.pop('count', 1)
#        if self.Loop is None:
            Count = StandardParameter(name = 'Count', set_cmd = self.function)
            Sweep_Count = Count[1:count+1:1]
            if self.Loop is None:
                self.Loop = Loop(sweep_values = Sweep_Count).each(self.dig)
            else:
                LOOP = self.Loop
                self.Loop = Loop(sweep_values = Sweep_Count).each(LOOP)
        
        return True


    def _QCoDeS_Loop(self, measured_parameter, sweeped_parameter, sweep_value = [0,0,0], **kw):
        
        Sweep_Values = sweeped_parameter[sweep_value[0]:sweep_value[1]:sweep_value[2]]
        
#        Sweep_Values2 = sweeped_parameter2[sweep_value2[0]:sweep_value2[1]:sweep_value2[2]]
        
        LOOP = Loop(sweep_values = Sweep_Values).each(measured_parameter)
        
        data_set = LOOP.get_data_set(location = None, loc_record = {'name': 'Chevron Pattern', 'label': 'frequency-burst_time'}, io = self.data_IO,)
        
        data_set = LOOP.run()
        
        return data_set
    
    
