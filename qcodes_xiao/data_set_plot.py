# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:38:33 2017

@author: X.X
"""
#%% import module
import numpy as np

import matplotlib.pyplot as plt

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
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


class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, 
                posttrigger_size, label=None, unit=None, instrument=digitizer,
                **kwargs):
       
        super().__init__(name=name, shape=(2*memsize,), instrument=instrument, **kwargs)
        
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        global digitizer
        
        
    def get(self):
#        res = digitizer.single_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        res = digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
#        res = digitizer.single_software_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        print(res.shape)
        return res
        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)




#%%
data = np.array([1,2,3])
data1 = np.array([[1,2,33,5],[5,232,7,3],[1,2,3,4]])

data_array1 = DataArray(preset_data = data, name = 'digitizer', is_setpoint = True)

data_array2 = DataArray(preset_data = data, name = 'digitizer2')

data_array3 = DataArray(preset_data = data, name = 'digitizer3')

data_array4 = DataArray(parameter=digitizer_param, is_setpoint=True)

data_array5 = DataArray(preset_data = data, name = 'digitizer5')

data_array6 = DataArray(preset_data = data1, name = 'digitizer6')

#%%

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()
test_location = '2017-08-18/20-40-19_T1_Vread_sweep'

#arrays = LP.containers()
arrays2 = []
arrays3 = [data_array1,]
arrays4 = [data_array1, data_array2, data_array3]

data_set = new_data(arrays=arrays3, location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

data_set.save_metadata()


#%% load data

test_location = '2017-08-18/20-40-19_T1_Vread_sweep'

data_set_2 = DataSet(location = test_location, io = NewIO,)
data_set_2.read()

data_set_3 = load_data(location = test_location, io = NewIO,)

#%% plot

Plot = MatPlot()

x = [1,2,3]

y = [22,3,555]

plt.plot(x, y) 