# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:31:17 2017

@author: X.X
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.sequence import Sequence
from gate import Single_Qubit_Gate#, Two_Qubit_Gate
from manipulation import Manipulation
from sequencer import Sequencer
#from initialize import Initialize
#from readout import Readout
from qubit import Qubit
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
from qcodes.instrument.base import Instrument
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray

from qcodes.actions import Task

import stationF006
#from stationF006 import station
from copy import deepcopy
from manipulation_library import Ramsey
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
import time
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot

from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability, set_digitizer, seperate_data, average_probability

#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
#    x_new = x/1e6
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#%%
class Experiment:

    def __init__(self, name, label, qubits, awg, awg2, pulsar, **kw):
        self.name = name
        self.label = label

        self.awg = awg
        self.awg2 = awg2
        self.qubits = qubits
        
        self.vsg = kw.pop('vsg',None)
        self.vsg2 = kw.pop('vsg2',None)
        self.digitizer = kw.pop('digitizer', None)
        
        self.awg_file = None

        self.channel_I = [qubit.microwave_gate['channel_I'] for qubit in qubits]
        self.channel_Q = [qubit.microwave_gate['channel_Q'] for qubit in qubits]
        self.channel_VP = [qubit.plunger_gate['channel_VP'] for qubit in qubits]
        self.channel_PM = [qubit.microwave_gate['channel_PM'] for qubit in qubits]

        self.qubit_number = len(qubits)
        self.readnames = []

        self.sequence = Sequence(name = name)

        self.sweep_point1 = 0
        self.sweep_point2 = 0

        self.sweep_loop1 = {}
        self.sweep_loop2 = {}

        self.sweep_loop = {
            'loop1': self.sweep_loop1,
            'loop2': self.sweep_loop2,
            }

        self.sweep2_loop1 = {}
        self.sweep2_loop2 = {}

        self.sweep_set = {}         ## {''}
        self.sweep_type = 'NoSweep'
        
        self.dig = None
        
        self.Loop = None
        self.X_parameter = None
        self.Y_parameter = None
        self.X_parameter_type = None
        self.Y_parameter_type = None
        self.X_sweep_array = np.array([])
        self.Y_sweep_array = np.array([])
        self.X_sweep_points = np.array([0])
        self.X_sweep_all_arrays = []
        self.X_all_parameters = []
        self.X_all_measurements = []
        self.Y_measurement = None
        self.current_yvalue = 0
        
        
        self.calibration_qubit = None
        
        self.seq_repetition = 100
        
        self.formatter = HDF5FormatMetadata()
#        self.formatter = HDF5Format()
        self.saveraw = False
        
        self.write_period = None
        
        self.data_IO = DiskIO(base_location = 'D:\\Data\\'+self.name)
        self.data_location = time.strftime("%Y-%m-%d/%H-%M-%S/") + name + label
        self.calibration_data_location = self.data_location+'_calibration'
        
        self.data_set = None
        
        self.calibration_data_set = None
        
        self.threshold = 0.025
        
        self.readout_time = 0.6e-3
        
#        self.ordered_data = new_data(location = self.data_location+'_ordered', io = self.data_IO, 
#                                         write_period = self.write_period, formatter = self.formatter)
#        
#        self.probability_data = new_data(location = self.data_location+'_probability', io = self.data_IO, 
#                                         write_period = self.write_period, formatter = self.formatter)
        self.ordered_data = []
        self.probability_data = []
        self.averaged_data = []
        self.plot = []
        self.average_plot = []
        
        
#        self.averaged_data = new_data(location = self.data_location+'_averaged_data', io = self.data_IO, 
#                                      write_period = self.write_period, formatter = self.formatter)
        
        self.plot_average = False

        self.manip_elem = []
        self.manip2_elem = None
        self.manipulation_elements = {
#                'Rabi': None,
#                'Ramsey': None
                }
        
        self.sequence_cfg = []      ## [segment1, segment2, segment3]
        self.sequence_cfg_type = {}
        
        self.sequence2_cfg = []
        self.sequence2_cfg_type = {}
        
        self.seq_cfg = []
        self.seq_cfg_type = []
        
        self.sequencer = []
        
        self.dimension_1 = 0
        self.dimension_2 = 0
        self.Count = 0

        self.segment = {
#                'init': self.initialize_segment,
#                'manip': self.manipulation_segment,
#                'read': self.readout_segment,
#                'init2': self.initialize2_segment,
#                'manip2': self.manipulation2_segment,
#                'read2': self.readout2_segment,
                }

        self.repetition = {
#                'init': self.initialize_repetition,
#                'manip': None,
#                'read': self.readout_repetition,
#                'init2': self.initialize2_repetition,
#                'manip2': None,
#                'read2': self.readout2_repetition,
                }
        
        self.element = {}           ##  will be used in    pulsar.program_awg(myseq, self.elements)
         ##  a dic with {'init': ele, 'manip_1': ele, 'manip_2': ele, 'manip_3': ele, 'readout': ele,......}
        self.elts = []

        self.pulsar = pulsar

        self.channel = {}          ## unsure if it is necessary yet

        self.experiment = {}        ## {'Ramsey': , 'Rabi': ,}  it is a list of manipulation element for different experiment

        self.sweep_matrix = np.array([])
        
        self.digitizer_trigger_channel = 'ch5_marker1'
        self.digitier_readout_marker = 'ch6_marker1'
        self.occupied_channel1 = 'ch2_marker2'
        self.occupied_channel2 = 'ch5_marker2'
    
    
    def reset(self, name = None, label = None, **kw):
        
        name = self.name if name is None else name
        label = self.label if label is None else label
        
        self.__init__(name, label, self.qubits, self.awg, self.awg2, self.pulsar, vsg = self.vsg, vsg2=self.vsg2, digitizer = self.digitizer)
        
    def add_manip_elem(self, name, manip_elem, seq_num):
        
        self.manipulation_elements[name] = manip_elem
#        self.manip_elem.append([])
        if seq_num > len(self.manip_elem):
            for i in range(seq_num - len(self.manip_elem)):
                self.manip_elem.append([])
        self.manip_elem[seq_num-1].append(manip_elem)
        print('33333333',self.manip_elem)
        return True
    
    def find_sequencer(self, name):
        
        for sequencer in self.sequencer:
            if name == sequencer.name:
                return sequencer
                break
        return 'sequencer not found'
        
    
    def add_measurement(self, measurement_name, manip_name, manip_elem, sequence_cfg, sequence_cfg_type):
        
        sequencer = Sequencer(name = measurement_name, qubits=self.qubits, awg=self.awg, awg2=self.awg2, pulsar=self.pulsar,
                              vsg=self.vsg, vsg2=self.vsg2,digitizer=self.digitizer)
        
        
        i = len(self.sequencer)
        self.sequencer.append(sequencer)
        seq_cfg = deepcopy(sequence_cfg)
        seq_cfg_type = deepcopy(sequence_cfg_type)
        for seg in range(len(seq_cfg_type)):
            if seq_cfg_type[seg].startswith('manip'):
                Name = manip_name.pop(0)
                Elem = manip_elem.pop(0)
                seq_cfg[seg]['step1'].update({'manip_elem': Name})
                self.add_manip_elem(name = Name, manip_elem = Elem, seq_num = i+1)
                self.sequencer[i].manipulation_elements[Name] = Elem
        self.sequencer[i].sequence_cfg = seq_cfg
        self.sequencer[i].sequence_cfg_type = seq_cfg_type
        self.seq_cfg.append(seq_cfg)
        self.seq_cfg_type.append(seq_cfg_type)
        
        self.digitizer, self.dig = set_digitizer(self.digitizer, 1, self.qubit_number, self.seq_repetition, self.threshold, 
                                                 self.X_sweep_array, self.Y_sweep_array, self.saveraw, self.readout_time)

#        self.sequencer[i].sweep_loop1 = self.sweep_loop1[i]
#        self.sequencer[i].sweep_loop2 = self.sweep_loop2[i]
#        self.sequencer[i].set_sweep()
        
        return self

    def __call__(self,):
        
        return self
    
    def add_X_parameter(self, measurement, parameter, sweep_array, **kw):
        
        sequencer = self.find_sequencer(measurement)
        
        if len(sequencer.X_sweep_array) == 0:
            sequencer.X_sweep_array = sweep_array 
        
            self.X_sweep_array = np.append(self.X_sweep_array, sweep_array)
            self.X_sweep_points  = np.append(self.X_sweep_points, self.X_sweep_points[-1] + len(sweep_array))
        # so I want to know the structure of the multiple sweeps for plotting and saving...
            self.X_sweep_all_arrays.append(sweep_array)

        
        element = kw.pop('element', None)
        
        if type(parameter) is StandardParameter:#isinstance(obj, Tektronix_AWG5014)
            
            self.digitizer, self.dig = set_digitizer(self.digitizer, 1, self.qubit_number, self.seq_repetition,self.threshold, 
                                                     0, 0, self.saveraw, self.readout_time)
            
            step = (sweep_array[1]-sweep_array[0])
            
            Sweep_Value = parameter[sweep_array[0]:sweep_array[-1]+step:step]
    
            self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig)
            
            self.X_parameter = parameter.full_name
            self.X_parameter_type = 'Out_Sequence'
            
            sequencer.X_parameter = parameter.full_name
            sequencer.X_parameter_type = 'Out_Sequence'
            
        elif type(parameter) is str:
            
#            sequencer.digitizer, sequencer.dig = set_digitizer(self.digitizer, len(sweep_array), self.qubit_number, self.seq_repetition)
            
            self.digitizer, self.dig = set_digitizer(self.digitizer, len(self.X_sweep_array), self.qubit_number, self.seq_repetition, self.threshold, 
                                                     self.X_sweep_array, self.Y_sweep_array, self.saveraw, self.readout_time)
            
            i = len(sequencer.sweep_loop1)+1
            para = 'para'+str(i)
            loop = 'loop1_'+para
            sequencer.sweep_loop1[para] = sweep_array
            for seg in range(len(sequencer.sequence_cfg_type)):
                if sequencer.sequence_cfg_type[seg].startswith('manip'):
                    if sequencer.sequence_cfg[seg]['step1']['manip_elem'] == element:
                        sequencer.sequence_cfg[seg]['step1'].update({parameter: loop})
                elif sequencer.sequence_cfg_type[seg].startswith('read'):
                    for w in range(len(element)):
                        if element[w] == '_':
                            break
                    if element[:w] == sequencer.sequence_cfg_type[seg]:
                        step = 'step'+element[-1]
                        sequencer.sequence_cfg[seg][step].update({parameter: loop})
                elif sequencer.sequence_cfg_type[seg].startswith('init'):
                    if element.startswith(sequencer.sequence_cfg_type[seg]):
                        step = 'step'+element[-1]
                        sequencer.sequence_cfg[seg][step].update({parameter: loop})
#                    break
            
            self.X_parameter = parameter
            self.X_parameter_type = 'In_Sequence'
            
            sequencer.X_parameter = parameter
            sequencer.X_parameter_type = 'In_Sequence'
            
        if measurement not in self.X_all_measurements:
            self.X_all_parameters.append(sequencer.X_parameter)
            self.X_all_measurements.append(measurement)
            self.dimension_1 += len(sweep_array)
            self.sweep_type = '1D'
        
        return True
    
    def add_Y_parameter(self, measurement, parameter, sweep_array, **kw):
        
        with_calibration = kw.pop('with_calibration', False)
        if with_calibration:
            calibration_task = Task(func = self.update_calibration)
        
        sequencer = self.find_sequencer(measurement)
        self.Y_measurement = measurement
        self.Y_sweep_array = sweep_array
        
        element = kw.pop('element', None)
        
        if type(parameter) is StandardParameter:
            
            step = (sweep_array[1]-sweep_array[0])
            
            Sweep_Value = parameter[sweep_array[0]:sweep_array[-1]+step:step]
            
            if self.Loop is None:
#                calibration = calibration_task if parameter
                if with_calibration:
                    self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig, calibration_task)
                else:
                    self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig)
            else:
                LOOP = self.Loop
                self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(LOOP)
            
            self.Y_parameter = parameter.full_name
            self.Y_parameter_type = 'Out_Sequence'
        else:
            for para in sequencer.sweep_loop2:
                if len(sequencer.sweep_loop2[para]) == 0:
                    break
            i = len(sequencer.sweep_loop2)+1
            para = 'para'+str(i)
            loop = 'loop2_'+para
            sequencer.sweep_loop2[para] = sweep_array

            for seg in range(len(sequencer.sequence_cfg_type)):
                if sequencer.sequence_cfg_type[seg].startswith('manip'):
                    if sequencer.sequence_cfg[seg]['step1']['manip_elem'] == element:
                        sequencer.sequence_cfg[seg]['step1'].update({parameter: loop})
                elif sequencer.sequence_cfg_type[seg].startswith('read'):
                    for w in range(len(element)):
                        if element[w] == '_':
                            break
                    if element[:w] == sequencer.sequence_cfg_type[seg]:
                        step = 'step'+element[-1]
                        sequencer.sequence_cfg[seg][step].update({parameter: loop})
                elif sequencer.sequence_cfg_type[seg].startswith('init'):
                    if element.startswith(sequencer.sequence_cfg_type[seg]):
                        step = 'step'+element[-1]
                        sequencer.sequence_cfg[seg][step].update({parameter: loop})
            
            self.Y_parameter = parameter
            self.Y_parameter_type = 'In_Sequence'
            
            Y = StandardParameter(name = self.Y_parameter, set_cmd = self.update_Y_parameter)
            Sweep_Value = Y[0:len(self.Y_sweep_array):1]
            
            if self.Loop is None:
                if with_calibration:
                    print('calibration')
                    self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig, calibration_task)
                else:
                    self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig)
#                self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(self.dig)
            elif self.Loop is not None and self.Y_measurement is None:
                print('no calibration')
                LOOP = self.Loop
                self.Loop = Loop(sweep_values = Sweep_Value, delay = 2).each(LOOP)
            
            
        
        self.dimension_2 = len(sweep_array)
        self.sweep_type = '2D'
        
        return True
    
    def function(self, x):
        
        return True
    
    def function2(self,):
        
        self.load_sequence()
        
        set_digitizer(self.digitizer, len(self.X_sweep_array), self.qubit_number, self.seq_repetition, self.threshold, 
                      self.X_sweep_array, self.Y_sweep_array, self.saveraw, self.readout_time)
        
        self.pulsar.start()
        
        return True
    
    def update_Y_parameter(self, idx_j):
        
        self.pulsar.stop()
        self.sequence = Sequence(name = self.name)
        self.elts = []
        for sequencer in self.sequencer:
            sequencer.elts = []
            sequencer.sequence = Sequence(name = sequencer.name)
        self.generate_1D_sequence(idx_j = idx_j)
        self.load_sequence()
        self.awg.all_channels_on()
        self.awg2.all_channels_on()
        
        self.vsg.status('On')
        self.vsg2.status('On')

        self.pulsar.start()
        
        time.sleep(2)

        return
    
    def update_calibration(self,):
        if self.calibration_qubit == 'all' or self.calibration_qubit == 'qubit_1':
            self.calibrate_by_Ramsey(1)
        if self.calibration_qubit == 'all' or self.calibration_qubit == 'qubit_2':
            self.calibrate_by_Ramsey(0)
        self.current_yvalue = self.current_yvalue+1
        return True

    def set_sweep(self, repetition = False, with_calibration = False, plot_average = False, **kw):
        
        self.plot_average = plot_average
#==============================================================================
# I dont think you should set up the plot windo if you are not live plotting
#         
#         for i in range(self.qubit_number):
#             d = i if i == 1 else 2
#             if i >=2:
#                 d = i+1
#             self.plot.append(QtPlot(window_title = 'raw plotting qubit_%d'%d, remote = False))
# #            self.plot.append(MatPlot(figsize = (8,5)))
#             if plot_average:
# #                self.average_plot.append(QtPlot(window_title = 'average data qubit_%d'%d, remote = False))
#                 self.average_plot.append(MatPlot(figsize = (8,5)))
#==============================================================================
        
        for seq in range(len(self.sequencer)):
            self.sequencer[seq].set_sweep()
            self.make_all_segment_list(seq)
        
        """
        loop function
        """
        
        if repetition is True:
            count = kw.pop('count', 1)
            self.Count = count
            
            Count = StandardParameter(name = 'Count', set_cmd = self.function)
            
            if with_calibration:
                calibration = kw.pop('calibration',None)
                if calibration is not None:
                    calibrated_parameter = calibration.calibrated_parameter
                    load_task = Task(func = self.function2)
                    load_task2 = Task(func = calibration.load_sequence)
                else:
                    calibration_task = Task(func = self.update_calibration)
                Count = StandardParameter(name = 'Count', set_cmd = self.function)
                
                
            
            Sweep_Count = Count[1:count+1:1]
            if self.Loop is None:
                if with_calibration:
                    if calibration is not None:
                        self.Loop = Loop(sweep_values = Sweep_Count).each(load_task2, calibrated_parameter, load_task, self.dig)
                    else:
                        self.Loop = Loop(sweep_values = Sweep_Count).each(self.dig, calibration_task,)
                else:
                    self.Loop = Loop(sweep_values = Sweep_Count).each(self.dig)
            else:
                LOOP = self.Loop
                self.Loop = Loop(sweep_values = Sweep_Count).each(LOOP)
            
        return True
    
    def live_plotting(self,):
#        loop_num = self.dimension_1 if self.X_parameter_type is 'In_Sequence' else 1
        self.convert_to_probability()
        
        for i in range(self.qubit_number):
            self.plot[i].update()
        
        if self.plot_average:
            self.calculate_average_data()
            for i in range(self.qubit_number):
                self.average_plot[i].update()
        
        return self.probability_data
    
    def plot_save(self):
        
        for i in range(self.qubit_number):
            self.plot[i].save()
#            self.plot[i].save('C:/Users/LocalAdmin/Documents/test_data/{}/{}.png'.format(self.name, time.strftime("%H-%M-%S")))
#            if len(self.average_plot) != 0:
#                self.average_plot[i].save('C:/Users/LocalAdmin/Documents/test_data/{}/{}.png'.format(self.name, time.strftime("%H-%M-%S")))
        return True

#       
    
#    def convert_to_probability(self,):
#        sweep_array = self.X_sweep_array if len(self.X_sweep_array) != 0 else None
#        name = self.X_parameter if self.X_parameter is not None else self.name
#        probability_dataset = convert_to_probability(data_set = self.data_set, threshold = self.threshold, qubit_num = self.qubit_number, 
#                                                     repetition = self.seq_repetition, name = name, sweep_array = sweep_array)
#        for parameter in probability_dataset.arrays:
#            if parameter in self.probability_data.arrays:
#                self.probability_data.arrays[parameter].ndarray = probability_dataset.arrays[parameter].ndarray
#            else:
#                self.probability_data.arrays[parameter] = probability_dataset.arrays[parameter]
#        return probability_dataset
#    
#    def calculate_average_data(self, measurement = 'self'):
#        
#        i = 0
#        data_array = []
#        sequencer = None if measurement is 'self' else self.find_sequencer(measurement)
#        average_data = self.averaged_data if measurement is 'self' else new_data(io = self.data_IO, formatter = self.formatter)
##        probability_data = self.probability_data if measurement is 'self' else sequencer.probability_data
#        for parameter in self.probability_data.arrays:
#            if len(self.probability_data.arrays[parameter].ndarray.shape) == 2:
#                data = deepcopy(self.probability_data.arrays[parameter].ndarray)
#                data = np.average(data, axis = 0)
#                is_setpoint = self.probability_data.arrays[parameter].is_setpoint
#                name = self.probability_data.arrays[parameter].name
#                array_id = self.probability_data.arrays[parameter].array_id
#                
#                data_array.append(DataArray(preset_data = data, name = name, array_id = array_id, is_setpoint = is_setpoint))
#                if parameter in self.averaged_data.arrays:
#                    average_data.arrays[parameter].ndarray = data_array[i].ndarray
#                else:
#                    average_data.arrays[parameter] = data_array[i]
##                    self.averaged_data.add_array(data_array[i])
#                i+=1
#        
##        self.average_plot.add(self.averaged_data.digitizer,figsize=(1200, 500))
#        return average_data
    
#    def plot_probability(self):
#               
#
#        for i in range(self.qubit_number):
#            name = self.X_parameter if self.X_parameter is not None else self.name
#            n = 1 if i == 1 else 2
#            if i >=2:
#                n = i+1
#            self.plot.append(QtPlot(window_title = 'raw plotting', remote = False)) 
#            self.plot[i].add(x = self.probability_data.arrays[name+'_set'],
#                     y = self.probability_data.arrays['Count_set'] if self.sweep_type is '1D' else self.probability_data.arrays[self.Y_parameter+'_set'],
#                     z = self.probability_data.arrays['digitizerqubit_%d'%(n)], figsize=(1200, 500))
#        return True
    def plot_probability(self, measurements = 'All'):
        #plot 2D data set for all measurements in the experiment
        if measurements == 'All':
            measurements = self.X_all_measurements
            
        self.plot = []
            
        for i in range(self.qubit_number):
            self.plot.append(QtPlot(window_title = 'raw plotting', remote = False))  
        
        if self.X_parameter_type == 'Out_Sequence':
            x = self.data_set.arrays[self.X_parameter+'_set']
            y = self.data_set.arrays['Count_set']
            z = self.data_set.arrays['probability_data']
            
            for i in range(self.qubit_number):

                self.plot[i].add(x = x,
                         y = y, 
                         z = z[:,:,i,0], figsize=(1200, 500))
        else:
            x = self.data_set.arrays['sweep_data']
            y = self.data_set.arrays['Count_set']
            z = self.data_set.arrays['probability_data']
            
            for measurement in measurements:
                pos = self.X_all_measurements.index(measurement)
                x_start  = self.X_sweep_points[pos]
                x_end =  self.X_sweep_points[pos+1]
 
                for i in range(self.qubit_number):
                    self.plot[i].add(x = x[0,:,x_start:x_end],
                             y = y, 
                             z = z[:,i,x_start:x_end], figsize=(1200, 500))    

                
    def plot_average_probability(self, measurements = 'All', xaxis = True):
        
        if measurements == 'All':
            measurements = self.X_all_measurements
        
        self.average_plot = []
        
        for i in range(self.qubit_number):
            self.average_plot.append(MatPlot(figsize = (8,5)))  
            
        if self.X_parameter_type == 'Out_Sequence':
            x1 = self.data_set.arrays[self.X_parameter+'_set'].mean(axis = 0)
#            x = x1[0,:]
            y = self.data_set.arrays['probability_data'].mean(axis = 0)
            for i in range(self.qubit_number):           
                self.average_plot[i].add(x1, y[:,i,0])   
        else:
            for measurement in measurements:
                pos = self.X_all_measurements.index(measurement)
                x_start  = self.X_sweep_points[pos]
                x_end =  self.X_sweep_points[pos+1]
                
                if xaxis  == True:
                    x1 = self.data_set.arrays['sweep_data'].mean(axis = 0)
                    x = x1[0,x_start:x_end]
                else:
                    x = np.zeros(shape = (x_end-x_start))
                    x[:] = list(range(x_start, x_end))                    
                y = self.data_set.arrays['probability_data'].mean(axis = 0)
    
                for i in range(self.qubit_number):  
                        self.average_plot[i].add(x, y[i,x_start:x_end])

    
#    def plot_average_probability(self, withaxis = True):
#        if len(self.average_plot) == 0:
#            for i in range(self.qubit_number):
#                self.average_plot.append(MatPlot(figsize = (8,5)))
#        self.calculate_average_data()
#
#        for i in range(self.qubit_number):
#            n = 1 if i == 1 else 2
#            if i >=2:
#                n = i+1      
#            if withaxis:    
#                self.average_plot[i].add(x=self.averaged_data.arrays[self.X_parameter+'_set'],
#                                 y=self.averaged_data.arrays['digitizerqubit_%d'%(n)],)# figsize=(1200, 500))
#            else:
#                y1  =self.averaged_data.arrays['digitizerqubit_%d'%(n)]
#                x1  = np.arange(len(y1))
#                self.average_plot[i].add(x=x1,y=y1,)# figsize=(1200, 500))    
        
#==============================================================================
#         if self.X_parameter_type is 'In_Sequence':
#             for i in range(self.qubit_number):
#                 n = 1 if i == 1 else 2
#                 if i >=2:
#                     n = i+1                
#                 self.average_plot[i].add(x=self.averaged_data.arrays[self.X_parameter+'_set'],
#                                  y=self.averaged_data.arrays['digitizerqubit_%d'%(n)],)# figsize=(1200, 500))
#         else:
#             for i in range(self.qubit_number):
#                 n = 1 if i == 1 else 2
#                 if i >=2:
#                     n = i+1
#                 self.average_plot[i].add(x=self.averaged_data.arrays[self.X_parameter+'_set'],
#                                  y=self.averaged_data.arrays['digitizerqubit_%d'%(n)],)#figsize=(1200, 500))
#==============================================================================
        
        return True
    
    def make_sequencers(self, **kw):
        
        sequencer_amount = len(self.seq_cfg)
        self.sequencer = []
        for i in range(sequencer_amount):
            name = 'Seq_%d'%(i+1)#self.seq_cfg[i]
            print('sequencer name',name)
            sequencer_i = Sequencer(name = name, qubits=self.qubits, awg=self.awg, awg2=self.awg2, pulsar=self.pulsar,
                          vsg=self.vsg, vsg2=self.vsg2,digitizer=self.digitizer)
            print('sequencer name 2nd check',name)
            self.sequencer.append(sequencer_i)
            print('sequencer list', self.sequencer)
            self.sequencer[i].sequence_cfg = self.seq_cfg[i]
            self.sequencer[i].sequence_cfg_type = self.seq_cfg_type[i]
            
            self.sequencer[i].sweep_loop1 = self.sweep_loop1[i]
            self.sequencer[i].sweep_loop2 = self.sweep_loop2[i]
            self.sequencer[i].set_sweep()
            print('sequencer type3', type(sequencer_i))
            
            self.sequencer[i].manip_elem = self.manip_elem[i]
            for manip in self.manip_elem[i]:
                self.sequencer[i].add_manip_elem(name = manip.name, manip_elem = manip)
                
            self.make_all_segment_list(seq_num = i)
            print('sequencer type4', type(sequencer_i))
        
        return self.sequencer
    
    
    
    def make_all_segment_list(self, seq_num):
        
        i = 0

        for segment_type in self.seq_cfg_type[seq_num]:
            print('segment_type', segment_type)

            if segment_type.startswith('init') and segment_type not in self.segment:#not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_initialize_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('manip') and segment_type not in self.segment:#not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_manipulation_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('read') and segment_type not in self.segment:#not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_readout_segment_list(segment_num = i, name = segment_type)

            i+=1

        return True
    
    
    """
    this function below organizes segments. e.g. self.initialize_segment = {'step1': element1, 'step2': 1D/2D list, 'step3': element3...}
    """
    
    def first_segment_into_sequence_and_elts(self, segment, segment_type, repetition, rep_idx = 0,idx_j = 0, idx_i = 0, seq_num = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
        
        sq = seq_num
#        print('rep_idx',rep_idx)
        for step in segment:
            print('step',step, 'idx_i', idx_i)
            
            if type(segment[step]) is not list:             ## smarter way for this
                element = segment[step]
                wfname = element.name
                name = wfname + '_%d_%d_%d'%(sq,j,i)
                repe = repetition[step]
                if i == 0:
                    self.elts.append(element)
                    if segment_type in self.sequencer[sq].sequence_cfg_type:
                        self.sequencer[sq].elts.append(element)
            else:
                jj = 0 if len(segment[step]) == 1 else j
                ii = 0 if len(segment[step][0]) == 1 else i
                element = segment[step][jj][ii]
                wfname=element.name
                name = wfname + '_%d_%d_%d'%(sq,j,i)
                
                repe = repetition[step][jj][ii]
                if ii != 0 or i==0:                                                         ## !!!!!!!!! here is a potential BUG!!!!!!
                    self.elts.append(element)
                    if segment_type in self.sequencer[sq].sequence_cfg_type:
                        self.sequencer[sq].elts.append(element)
            
            self.sequence.append(name = name, wfname = wfname, trigger_wait = False, repetitions = repe)
            if segment_type in self.sequencer[sq].sequence_cfg_type:
                self.sequencer[sq].sequence.append(name = name, wfname = wfname, trigger_wait = False, repetitions = repe)
        
        return True
    
    """
    problems below
    """
    
    def update_segment_into_sequence_and_elts(self, segment, segment_type, repetition, rep_idx = 0,idx_j = 0, idx_i = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
        unit_seq_length=0
        for seg_type in self.sequence_cfg_type:
            unit_seq_length += len(self.segment[seg_type])
        if segment == self.initialize_segment:
            former_element = 0
        elif segment == self.manipulation_segment:
            former_element =  len(self.initialize_segment)
        elif segment == self.readout_segment:
            former_element =  len(self.initialize_segment) + len(self.manipulation_segment)
        
        
        for step in segment:                                ## step = 'step1'... is a string
            
            if type(segment[step]) is list:
                jj = 0 if len(segment[step]) == 1 else j
                ii = 0 if len(segment[step][0]) == 1 else i
                element = segment[step][jj][ii]
                wfname=element.name
                name = wfname + '_%d_%d'%(j,i)
                repe = repetition[step][jj][ii]
                if jj != 0:
                    wfname_to_delete = segment[step][jj-1][ii].name
                    element_no = i*unit_seq_length + former_element + int(step[-1])+1
                    print('element_no',element_no)
                    if i == 0:
                        self.add_new_element_to_awg_list(element = element)
                    self.add_new_waveform_to_sequence(wfname = wfname, element_no = element_no, repetitions = repe)
                    self.delete_element_from_awg_list(wfname = wfname_to_delete)
            
        return True
    
    
    def add_segment(self, segment, segment_type, repetition, rep_idx = 0, idx_j = 0, idx_i = 0, seq_num = 0):
        
        j = idx_j
        
        if j == 0:
            self.first_segment_into_sequence_and_elts(segment, segment_type, repetition, idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
        else:
            self.first_segment_into_sequence_and_elts(segment, segment_type, repetition, idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
#            self.update_segment_into_sequence_and_elts(segment, repetition, idx_j = idx_j, idx_i = idx_i)
        
        return True
    
    def add_compensation(self, idx_j = 0, idx_i = 0, seq_num = 0):
        unit_length = 0
        for segment in self.sequencer[seq_num].sequence_cfg:
            unit_length += len(segment) 
        amplitude = [0,0]
        
        for i in range(unit_length):
            wfname = self.sequence.elements[-(i+1)]['wfname']
            for elt in self.elts:
                if elt.name == wfname:
                    element = elt
                    break
            tvals, wfs = element.ideal_waveforms()
            repe = self.sequence.elements[-(i+1)]['repetitions']
            amplitude[0] += np.sum(wfs[self.channel_VP[0]])*repe
            amplitude[1] += np.sum(wfs[self.channel_VP[1]])*repe
            
        comp_amp = [-amplitude[k]/3000000 for k in range(2)]
        
        if np.max(np.abs(comp_amp)) >= 0.5:
            raise ValueError('amp too large')
        
        compensation_element = Element('compensation_%d_%d'%(seq_num, idx_i), self.pulsar)

        compensation_element.add(SquarePulse(name = 'COMPEN1', channel = self.channel_VP[0], amplitude=comp_amp[0], length=1e-6),
                            name='compensation1',)
        compensation_element.add(SquarePulse(name = 'COMPEN2', channel = self.channel_VP[1], amplitude=comp_amp[1], length=1e-6),
                            name='compensation2',refpulse = 'compensation1', refpoint = 'start', start = 0)
        
        compensation_element.add(SquarePulse(name='comp_c1m2', channel=self.occupied_channel1, amplitude=2, length=1e-6),
                                           name='comp%d_c1m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        compensation_element.add(SquarePulse(name='comp_c5m2', channel=self.occupied_channel2, amplitude=2, length=1e-6),
                                           name='comp%d_c5m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        
        self.elts.append(compensation_element)
        self.sequencer[seq_num].elts.append(compensation_element)
        
        self.sequence.append(name = 'compensation_%d_%d'%(seq_num, idx_i), 
                             wfname = 'compensation_%d_%d'%(seq_num, idx_i), 
                             trigger_wait = False, repetitions = 3000)
        self.sequencer[seq_num].sequence.append(name = 'compensation_%d_%d'%(seq_num, idx_i), 
                                                wfname = 'compensation_%d_%d'%(seq_num, idx_i), 
                                                trigger_wait = False, repetitions = 3000)
        
        return True

   
    def generate_unit_sequence(self, seq_num = 0, rep_idx = 0, idx_i = 0, idx_j = 0):          # rep_idx = 10*i+j
        
        i = 0           ## not used in this version
        
        for segment_type in self.sequencer[seq_num].sequence_cfg_type:

            if segment_type.startswith('init'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('manip'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('read'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            self.add_segment(segment = segment, segment_type = segment_type, repetition = repetition, 
                             idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
            i+=1
            
        self.add_compensation(idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)

        return True
 
    def generate_1D_sequence(self, idx_j = 0):
        seq_num = 0
        
#        self.set_trigger()
        
        for sequencer in self.sequencer:
            D1 = sequencer.dimension_1
            for idx_i in range(D1):
                self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
            seq_num += 1
        self.sequence, self.elts = self.set_trigger(self.sequence, self.elts)
#        if idx_j == 0:
#            self.load_sequence()

        return self.sequence
    
    def generate_1D_sequence_v2(self, idx_j = 0):
       
        for sequencer in self.sequencer:
            sequencer.generate_1D_sequence(idx_j = idx_j)

        for idx_i in range(len(self.sequencer)):
            self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i)
        if idx_j == 0:
            self.load_sequence()
        
        return self.sequence
    
    
    def add_marker_into_first_readout(self, awg, element = None):
        
        first_read_step = self.segment['read']['step1']

        if type(first_read_step) is list:
            if type(first_read_step[0]) is list:
                first_read = first_read_step[0][0]
            else:
                first_read = first_read_step[0]
        else:
            first_read = first_read_step
        
        tvals, wfs = first_read.normalized_waveforms()
        name = '1st' + first_read.name
        print('readoutname:', name)
#        first_read.add(SquarePulse(name = 'read_marker', channel = self.digitier_readout_marker, amplitude=2, length=1e-6),
#                       name='read_marker', refpulse = 'read1', refpoint = 'start')
#        self.elts.append(first_read)
        
        channel = self.digitier_readout_marker
        wfs[channel] += 1
        i = (int(channel[2])-1)%4+1
        print('channel', 'ch%d'%i)
        wfs_vp = wfs['ch%d'%int(channel[2])][0]
        for ch in ['ch%d'%i, 'ch%d_marker1'%i, 'ch%d_marker2'%i]:
            if len(wfs[ch]) != 1000:
                wfs[ch] = np.full((1000,),1)
        wfs['ch%d_marker1'%i] = np.full((1000,),1)
        wfs['ch%d_marker2'%i] = np.full((1000,),1)
        wfs['ch%d'%i] = np.full((1000,),wfs_vp)
        print('ana', wfs['ch%d'%i], 'm1', wfs['ch%d_marker1'%i], 'm2', wfs['ch%d_marker2'%i])
        awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                  m2 = wfs['ch%d_marker2'%i], wfmname = name+'_ch%d'%i)
        
        element_no = 0
        for seg in self.sequencer[0].sequence_cfg_type:
            if seg=='read':#seg.startswith('read'):
                break
            else:
                element_no += len(self.segment[seg])
        
#        element_no = len(self.segment['init']) + len(self.segment['manip']) + 1 + 1
        element_no += 2
        awg.set_sqel_waveform(waveform_name = name+'_ch%d'%i, channel = i,
                              element_no = element_no)
        return first_read

    def add_marker_into_first_readout_v2(self, awg, element = None):
        
        return True
    def add_new_element_to_awg_list(self, awg, element):

        name = element.name

        tvals, wfs = element.normalized_waveforms()

        for i in range(1,5):
            awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                      m2 = wfs['ch%d_marker2'%i], wfmname = name+'_ch%d'%i)

        return True

    def add_new_waveform_to_sequence(self, wfname, element_no, repetitions):

        for i in range(1,5):
            self.awg.set_sqel_waveform(waveform_name = wfname+'_ch%d'%i, channel = i,
                                        element_no = element_no)
        
        self.awg.set_sqel_loopcnt(loopcount = repetitions, element_no = element_no)
        
        return True
    
    def delete_element_from_awg_list(self, wfname):
        
        for i in range(1,5):
            self.awg.write('WLISt:WAVeform:DELete "{}"'.format(wfname+'_ch%d'%i))
        
        return True
    
    def store_sequence(self, filename,):
        
        self.awg.send_awg_file(filename = filename, awg_file = self.awg_file)
        
        return True
    
    def restore_previous_sequence(self, filename = 'setup_0_.AWG'):
        
        self.awg.load_awg_file(filename = filename)
        
        self.awg2.load_awg_file(filename = filename)
#        self.awg.write('AWGCONTROL:SRESTORE "{}"'.format(filename))
        
        return True
    
    def save_sequence(self, awg, filename,disk):
        
        awg.write('AWGCONTROL:SSAVE "{}","{}"'.format(filename,'C:'))
        
        return True

    def set_trigger(self,sequence, elts):
        
        trigger_element = Element('trigger', self.pulsar)

        trigger_element.add(SquarePulse(name = 'TRG2', channel = 'ch8_marker2', amplitude=2, length=300e-9),
                            name='trigger2',)
        trigger_element.add(SquarePulse(name = 'TRG1', channel = 'ch4_marker2', amplitude=2, length=1426e-9),   #1376e-9
                            name='trigger1',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)
        
        trigger_element.add(SquarePulse(name = 'TRG_digi', channel = 'ch2_marker1', amplitude=2, length=800e-9),
                            name='TRG_digi',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)
        trigger_element.add(SquarePulse(name = 'SIG_digi', channel = 'ch8', amplitude=0.5, length=800e-9),
                            name='SIG_digi',refpulse = 'TRG_digi', refpoint = 'start', start = 0)
        
        
        extra_element = Element('extra', self.pulsar)
        extra_element.add(SquarePulse(name = 'EXT2', channel = 'ch8_marker2', amplitude=2, length=5e-9),
                            name='extra2',)
        extra_element.add(SquarePulse(name = 'EXT1', channel = 'ch4_marker2', amplitude=2, length=2e-6),
                            name='extra1',)

        elts.insert(0,trigger_element)
#        self.elts.append(extra_element)
        sequence.insert_element(name = 'trigger', wfname = 'trigger', pos = 0)
#        self.sequence.append(name ='extra', wfname = 'extra', trigger_wait = False)
        return sequence, elts

    def load_sequence(self, measurement = 'self',):
        
        print('load sequence')
#        elts = list(self.element.values())
#        self.awg.delete_all_waveforms_from_list()
#        self.awg2.delete_all_waveforms_from_list()
#        time.sleep(1)

        if measurement is 'self':
            sequence = self.sequence
            elts = self.elts
        else:
            sequencer = self.find_sequencer(measurement)
            sequence = sequencer.sequence
            elts = sequencer.elts
        
#        sequence, elts = self.set_trigger(sequence, elts)
#        elts = self.elts
#        sequence = self.sequence
        self.pulsar.program_awgs(sequence, *elts, AWGs = ['awg','awg2'],)       ## elts should be list(self.element.values)
        
        time.sleep(5)
        
        self.add_marker_into_first_readout(self.awg2)
#        self.awg2.trigger_level(0.5)
        self.awg2.set_sqel_trigger_wait(element_no = 1, state = 1)
        time.sleep(0.2)
        last_element_num = self.awg2.sequence_length()
        self.awg.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)
        self.awg2.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)

        return True
    
    def set_sweep_with_calibration(self, repetition = False, plot_average = False, **kw):
        
#==============================================================================
# unless you are specifically liveplotting I dont think this should be created.
#         self.plot_average = plot_average
#         
#         for i in range(self.qubit_number):
#             d = i if i == 1 else 2
#             self.plot.append(QtPlot(window_title = 'raw plotting qubit_%d'%d, remote = False))
# #            self.plot.append(MatPlot(figsize = (8,5)))
#             if plot_average:
# #                self.average_plot.append(QtPlot(window_title = 'average data qubit_%d'%d, remote = False))
#                 self.average_plot.append(MatPlot(figsize = (8,5)))
#==============================================================================
        
        for seq in range(len(self.sequencer)):
            self.sequencer[seq].set_sweep()
            self.make_all_segment_list(seq)
        
        if repetition is True:
            count = kw.pop('count', 1)

            Count = StandardParameter(name = 'Count', set_cmd = self.function)
            Sweep_Count = Count[1:2:1]
            
            Loop_Calibration = Loop(sweep_values = Sweep_Count).each(self.dig)
            
            
            if self.Loop is None:
                self.Loop = Loop(sweep_values = Sweep_Count).each(Loop_Calibration, self.dig)
            else:
                raise TypeError('calibration not set')
#                LOOP = self.Loop
#                self.Loop = Loop(sweep_values = Sweep_Count).each(LOOP)
                
        return True
    
    def calibrate_by_Rabi(self, qubit = 'qubit2'):
        q = qubit[-1]
        DS = self.averaged_data
        x = DS.arrays[self.X_parameter+'_set'].ndarray
        y = DS.arrays['digitizerqubit_'+q].ndarray
        
        try: 
            pars, pcov = curve_fit(Func_Sin, x, y,)
            y_new = Func_Sin(x,pars[0],pars[1],pars[2],pars[3])
            pt = MatPlot()
            pt.add(x,y_new)
        except RuntimeError:
            print('fitting not converging')
        
        return True
    
    def calibrate_by_Ramsey(self, qubit = 0):

        q = qubit
        VSG = self.vsg if q == 1 else self.vsg2

        pos = self.X_all_parameters.index('frequency_shift')
        x_start  = self.X_sweep_points[pos]
        x_end =  self.X_sweep_points[pos+1]
            
        x = self.data_set.arrays['sweep_data'][self.current_yvalue, q, x_start:x_end]
        y = self.data_set.arrays['probability_data'][self.current_yvalue, q, x_start:x_end]

        print('count: ', self.current_yvalue)
        print('x:',x)
        print('y:',y)
        
        try: 
            pars, pcov = curve_fit(Func_Gaussian, x, y, bounds = ((0, x[0]), (1,x[-1])))
#            plt.clf()
#            plt.plot(x,y)
#            plt.plot(x, Func_Gaussian(x, *pars), 'r-', label='fit')
#            plt.show()
            frequency_shift = pars[1]
            
        except RuntimeError:
            print('fitting not converging')
            frequency_shift = 0
        
        frequency = VSG.frequency()+frequency_shift*0.30
        print('frequency shift: ', frequency_shift)
        print('update fequency to: ', frequency)
        VSG.frequency(frequency)
        return True
    
    def draw_allXY(self, qubit = 'qubit2'):
        q = qubit[-1]
        
        DS = self.averaged_data
        
        pt = MatPlot()
        
        pt.add(x = DS.arrays[self.X_parameter+'_set'][11:], y = DS.arrays['digitizerqubit_'+q][11:])
        
        return True
    
    def run_experiment_with_calibration(self, calibration, experiment):
        
        calibration_sequencer = self.find_sequencer(calibration)
        
        measurement_sequencer = self.find_sequencer(experiment)
        
        self.load_sequence(calibration)
        
        calibration_data = self.calculate_average_data(measurement = 'calibration')
        
        self.close()
        
        self.load_sequence(experiment)
        self.close()
        
        return True

    def run_experiment(self,):
        
        print('run experiment')

        self.awg.all_channels_on()
#        self.awg.force_trigger()
        self.awg2.all_channels_on()
        
        self.vsg.status('On')
        self.vsg2.status('On')

        self.pulsar.start()
        
        time.sleep(2)
        
        if self.Loop is not None:
#            self.data_set = self.Loop.run(location = self.data_location, loc_record = {'name': self.name, 'label': self.label}, io = self.data_IO,)
#            self.data_set = self.Loop.get_data_set(location = self.data_location, 
#                                                   loc_record = {'name': self.name, 'label': self.label}, 
#                                                   io = self.data_IO, write_period = self.write_period)
#            
#
            self.data_set = self.Loop.get_data_set(location = self.data_location, 
                                                   loc_record = {'name': self.name, 'label': self.label}, 
                                                   io = self.data_IO, write_period = self.write_period, formatter = self.formatter)
#            
            self.data_set.add_metadata({'qubit_number': self.qubit_number})
            self.data_set.add_metadata({'X_all_measurements': self.X_all_measurements})
            self.data_set.add_metadata({'readnames': self.readnames})
            self.data_set.add_metadata({'X_parameter_type': self.X_parameter_type})
            self.data_set.add_metadata({'Y_parameter_type': self.Y_parameter_type})
            self.data_set.add_metadata({'X_sweep_points': self.X_sweep_points})
            self.data_set.add_metadata({'X_parameter':self.X_parameter})
            self.data_set.add_metadata({'Y_parameter':self.Y_parameter})
            
            self.data_set.arrays[self.Y_parameter+'_set'].ndarray = np.array(self.Y_sweep_array)



#            self.data_set = self.Loop.get_data_set(location = self.data_location, 
#                                                   loc_record = {'name': self.name, 'label': self.label}, 
#                                                   io = self.data_IO, write_period = self.write_period)

#            self.live_plotting()

#            self.plot_probability()
#            if self.plot_average:
#                self.plot_average_probability()
            
#            self.Loop.with_bg_task(task = self.live_plotting, bg_final_task = self.plot_save, min_delay = 1.5).run()
            try:
                self.Loop.run()
#                print('test')
            except:
                self.close()
#            self.Loop.with_bg_task(task = self.calibrate_by_Ramsey('qubit_2'),).run()
            self.awg.stop()
            self.awg2.stop()
            
#            self.vsg.status('Off')
#            self.vsg2.status('Off')
            
#            
#                    
                    
                    
            
#            
#            self.calculate_average_data()
            
            
#            self.plot_probability()
#            self.plot_average_probability()
        self.close()
        
        self.data_set.arrays[self.Y_parameter+'_set'].ndarray = np.array(self.Y_sweep_array)

        return self.data_set

    def close(self,):
        self.awg.stop()
        self.awg2.stop()
#        self.restore_previous_sequence()
        self.vsg.status('Off')
        self.vsg2.status('Off')
        self.awg.delete_all_waveforms_from_list()
        self.awg2.delete_all_waveforms_from_list()
        return True


    def run_all(self, name):                    ## the idea is to make a shortcut to run all the experiment one time

        self.initialize()

#        self.readout()

#        self.manipulation(name = manipulate)

        self._generate_sequence(name)

        self.awg.delete_all_waveforms_from_list()

        self.awg.stop()

        self.load_sequence(name)

        self.run_experiment(name)

        return True


