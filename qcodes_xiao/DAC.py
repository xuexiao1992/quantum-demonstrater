# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:46:55 2017

@author: X.X
"""
from qcodes.loops import Loop, ActiveLoop
from qcodes.instrument.sweep_values import SweepFixedValues
import stationF006

from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray

from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot

from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability, set_digitizer, seperate_data, average_probability
#%%

#station = stationF006.initialize()
G = station.gates
keithley = station.keithley

T = G.T
LP = G.LP

AMP = keithley.amplitude



#%%
Sweep_Value1 = T[0:-75:1]
Sweep_Value2 = LP[-320:-400:1]

LOOP = Loop(sweep_values = Sweep_Value2).loop(sweep_values = Sweep_Value1).each(AMP)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()

OldIO = DiskIO(base_location = 'D:\\文献\\QuTech\\QTlab\\xiaotest\\testIO')

## get_data_set should contain parameter like io, location, formatter and others
data = LOOP.get_data_set(location=None, loc_record = {'name':'DAC', 'label':'V_sweep'}, 
                       io = NewIO,)
print('loop.data_set: %s' % LOOP.data_set)