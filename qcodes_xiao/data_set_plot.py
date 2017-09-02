# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:38:33 2017

@author: X.X
"""
#%% import module
import numpy as np

import matplotlib.pyplot as plt

#import qcodes.instrument_drivers.Spectrum.M4i as M4i
#from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot
#%%
data = np.array([1,2,3])
data1 = np.array([[1,2,33,5],[5,232,7,3],[1,2,3,4]])

data_array1 = DataArray(preset_data = data, name = 'digitizer', is_setpoint = True)

data_array2 = DataArray(preset_data = data, name = 'digitizer2')

data_array3 = DataArray(preset_data = data, name = 'digitizer3')

#data_array4 = DataArray(parameter=digitizer_param, is_setpoint=True)

data_array5 = DataArray(preset_data = data, name = 'digitizer5')

data_array6 = DataArray(preset_data = data1, name = 'digitizer6')

#%%

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()
try_location = 'trytrytry'

#arrays = LP.containers()
arrays2 = []
arrays3 = [data_array1,]
arrays4 = [data_array1, data_array2, data_array3]

data_set = new_data(arrays=arrays3, location=try_location, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

#data_set.save_metadata()


#%% load data
NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()

test_location = '2017-08-28/14-38-39_finding_resonance_Freq_sweep'
new1_location = '2017-08-28/newnewnew'
new_location = '2017-09-02/16-18-06_finding_resonance_Freq_sweep'
data_set_2 = DataSet(location = test_location, io = NewIO,)
data_set_2.read()

data_set_3 = load_data(location = new_location, io = NewIO,)

sample_rate = int(np.floor(61035/1))
pretrigger = 16
readout_time = 1e-3

loop_num = 6
qubit_num = 1
repetition = 10

seg_size = int(((readout_time*sample_rate+pretrigger) // 16 + 1) * 16)
#%%
def convert_to_ordered_data(data_set,):
    
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter]
        dimension_1 = data_array.shape[0]
        arrayid = data_set.arrays[parameter].array_id
        if parameter.endswith('set'):
            if data_array.ndarray.ndim == 1:
                data_array1 = DataArray(preset_data = data_array.ndarray, name = parameter, array_id = arrayid, is_setpoint = True)
        elif not parameter.endswith('set'):
            
            data_num = int(data_set.arrays[parameter].shape[1]/2/(repetition+1) * repetition)
            
            data = np.ndarray(shape = (dimension_1, data_num))
            marker = np.ndarray(shape = (dimension_1, data_num))
            setpara = np.ndarray(shape = (dimension_1, data_num))
            
            for k in range(dimension_1):
                raw_data = data_array[k][::2]
                raw_marker = data_array[k][1::2]
                for seg in range(seg_size*loop_num):
                    if raw_marker[seg] > 0.1:           ##  a better threshold ???
                        break                
                data[k] = raw_data[seg:data_num+seg]
                marker[k] = raw_marker[seg:data_num+seg]
                setpara[k] = np.linspace(0, data_num-1, data_num)
                
            data_array2 = DataArray(preset_data = setpara, name = 'frequency', array_id = 'frequency_set', is_setpoint = True)
            data_array3 = DataArray(preset_data = data, name = parameter, array_id = arrayid, is_setpoint = False)
#            data_array4 = DataArray(preset_data = data, name = parameter, array_id = arrayid, is_setpoint = False)
    
    data_set_new = DataSet(location = new_location, io = NewIO, formatter = formatter)
    data_set_new.add_array(data_array1)
    data_set_new.add_array(data_array2)
    data_set_new.add_array(data_array3)
#    data_set_new.add_array(data_array4)
    return data_set_new

#%%

def convert_to_01_state(data_set, threshold):
    
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter]
        dimension_1 = data_array.shape[0]
        arrayid = data_set.arrays[parameter].array_id
        if parameter[-3:] == 'set':     ## or data_set.arrays[parameter].is_setpoint
            if len(data_array.shape) == 1:
                data_array1 = DataArray(preset_data = data_array.ndarray, name = parameter, array_id = arrayid, is_setpoint = True)

        elif parameter[-3:] != 'set':
            seg_num = int(data_set.arrays[parameter].shape[1]/seg_size)
            data = np.ndarray(shape = (dimension_1, seg_num))
            setpara = np.ndarray(shape = (dimension_1, seg_num))
            
            for k in range(dimension_1):
                print('parameter', parameter)
                for j in range(seg_num):
                    setpara[k][j] = j
                    for i in range(seg_size):
                        if data_array.ndarray[k][j*seg_size+i] >= threshold:
                            data[k][j] = 1
                            break
                    if i == seg_size-1:
                        data[k][j] = 0
            
            data_array2 = DataArray(preset_data = setpara, name = 'frequency', array_id = 'frequency_set', is_setpoint = True)
            data_array3 = DataArray(preset_data = data, name = parameter, array_id = arrayid, is_setpoint = False)            
    data_set_new = DataSet(location = new_location, io = NewIO, formatter = formatter)
    data_set_new.add_array(data_array1)
    data_set_new.add_array(data_array2)
    data_set_new.add_array(data_array3)
    
    return data_set_new
#%%
def convert_to_probability(data_set, threshold):
    
    data_set = convert_to_01_state(data_set, threshold)
    
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter]
        dimension_1 = data_array.shape[0]
        arrayid = data_set.arrays[parameter].array_id
        if parameter[-3:] == 'set':     ## or data_set.arrays[parameter].is_setpoint
            if len(data_array.shape) == 1:
                data_array1 = DataArray(preset_data = data_array.ndarray, name = parameter, array_id = arrayid, is_setpoint = True)
    
        elif parameter[-3:] != 'set':
            seg_num = int(data_set.arrays[parameter].shape[1])

            data = np.ndarray(shape = (dimension_1, loop_num))
            setpara = np.ndarray(shape = (dimension_1, loop_num))
            
            for k in range(dimension_1):
#                data_k = []
#                setpara_k = []
                state = np.ndarray(shape = (loop_num, int(seg_num/loop_num)))
                for i in range(seg_num):
                    loop = i%loop_num
                    sweep = i//loop_num
                    state[loop][sweep] = data_array.ndarray[k][i]
                
                for j in range(loop_num):
                    setpara[k][j] = j
                    probability = np.average(state[j])
                    data[k][j] = probability
                
            data_array2 = DataArray(preset_data = setpara, name = 'frequency', array_id = 'frequency_set', is_setpoint = True)
            data_array3 = DataArray(preset_data = data, name = parameter, array_id = arrayid, is_setpoint = False)
          
    data_set_new = DataSet(location = new_location, io = NewIO, formatter = formatter)
    data_set_new.add_array(data_array1)
    data_set_new.add_array(data_array2)
    data_set_new.add_array(data_array3)              
    
    return data_set_new
    
    

#data_set_100 = DataSet(location = new_location, formatter = formatter, io = NewIO,)

#%% plot

Plot = MatPlot()

x = [1,2,3]

y = [22,3,555]

plt.plot(x, y) 