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
from data_set_plot import set_digitizer


#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
#    x_new = x/1e6
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#%%



class Calibration(Experiment):
    
    def __init__(self, name, label, qubits, awg, awg2, pulsar, **kw):                ## name = 'calibration_20170615_Xiao'...
        
        super().__init__(name, label, qubits, awg, awg2, pulsar, **kw)
        
#        self.calibration_sequence = Sequence()
        self.sweep_inside_sequence = False
        
#        self.formatter = HDF5FormatMetadata()
#        self.data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
        self.calibration_data_location = self.data_location+'_calibration'
        
        self.calibration = 'Ramsey'
        self.experiment = 'AllXY'
        self.data_location = None
        self.data_IO = None
        self.calibrated_parameter = None
#        self.dig = digitizer_param(name='digitizer', mV_range = 1000, memsize=4e9, seg_size=seg_size, posttrigger_size=posttrigger_size)
        
    
    def switch_sequence(self,):
#        self.close()
#        self.load_sequence()
        self.digitizer, self.dig = set_digitizer(self.digitizer, self.dimension_1, self.qubit_number, self.seq_repetition,self.X_sweep_array)
        
        return True
    
    def set_calibration_parameter(self,):
        
        self.calibrated_parameter = StandardParameter(name = 'Count_Cal', get_cmd = self.calibrate_by_Ramsey)
        
        return True
    
    def calibrate_by_Ramsey(self,):
        
        self.data_set = None
        
        self.switch_sequence()
        self.Loop.bg_task = None
        self.run_experiment()
        self.convert_to_probability()
        DS = self.calculate_average_data()

        x = DS.frequency_shift_set.ndarray
        y = DS.digitizerqubit_1.ndarray
        try: 
            pars, pcov = curve_fit(Func_Gaussian, x, y, bounds = ((0, x[0]), (1,x[-1])))
            frequency_shift = pars[1]
        except RuntimeWarning:
            print('fitting not converging')
            frequency_shift = 0
        
        
        frequency = self.vsg2.frequency()+frequency_shift
        self.vsg2.frequency(frequency)
        
#        self.reset()

        return frequency
    
    def run_experiment(self,):
        
        print('run experiment')

        self.awg.all_channels_on()

        self.awg2.all_channels_on()
        
        self.vsg.status('On')
        self.vsg2.status('On')

        self.pulsar.start()
        
        time.sleep(1)
        
        if self.Loop is not None:
            
            self.data_set = self.Loop.get_data_set(location = self.data_location, 
                                                   loc_record = {'name': self.name, 'label': self.label}, 
                                                   io = self.data_IO, write_period = self.write_period)
            self.Loop.run()
            
            self.awg.stop()
            self.awg2.stop()
            
            self.vsg.status('Off')
            self.vsg2.status('Off')
            
#            self.convert_to_probability()
#            self.plot_probability()
#            self.plot_average_probability()

        return self.data_set

