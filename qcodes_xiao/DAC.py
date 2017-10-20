# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:46:55 2017

@author: X.X
"""
import numpy as np
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

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm


from data_set_plot import digitizer_param, convert_to_ordered_data, convert_to_01_state, convert_to_probability, set_digitizer, seperate_data, average_probability
#%%

#station = stationF006.initialize()
G = station.gates
keithley = station.keithley

digitizer = station.digitizer

T = G.T
LP = G.LP

AMP = keithley.amplitude
#%%

pretrigger=16
mV_range=1000
sample_rate = int(np.floor(61035/1))

digitizer.sample_rate(sample_rate)

sample_rate = digitizer.sample_rate()

readout_time = 10

qubit_num = 1

seg_size = ((readout_time*sample_rate+pretrigger) // 16 + 1) * 16

sweep_num = 1#len(sweep_loop1['para1']) if 'para1' in sweep_loop1 else 1
#    import data_set_plot
#    data_set_plot.loop_num = sweep_num

repetition = 1

memsize = int((repetition)*sweep_num*qubit_num*seg_size)
posttrigger_size = seg_size-pretrigger
    
#digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)
#digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)
    
digitizer.enable_channels(pyspcm.CHANNEL1)
    
#    digitizer.enable_channels(pyspcm.CHANNEL1)
digitizer.data_memory_size(memsize)
    
digitizer.segment_size(seg_size)
    
digitizer.posttrigger_memory_size(posttrigger_size)
    
digitizer.timeout(60000)
    
digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)

#trig_mode = pyspcm.SPC_TM_POS | pyspcm.SPC_TM_REARM
trig_mode = pyspcm.SPC_TMASK_SOFTWARE

digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 800, level1 = 900)

#digitizer, DIG = set_digitizer(digitizer = digitizer, sweep_num = 1, qubit_num = 1, repetition = 1, threshold = 1, X_sweep_array = [], saveraw = True)

DIG = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, 
                      posttrigger_size=posttrigger_size, digitizer = digitizer)


#%%
Sweep_Value1 = T[-17:-21:0.1]
#Sweep_Value1 = T[0:-75:1]
Sweep_Value2 = LP[-320:-400:1]

#LOOP = Loop(sweep_values = Sweep_Value2).loop(sweep_values = Sweep_Value1).each(AMP)

LOOP = Loop(sweep_values = Sweep_Value1).each(DIG)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\BillCoish_experiment')
formatter = HDF5FormatMetadata()

## get_data_set should contain parameter like io, location, formatter and others
data = LOOP.get_data_set(location=None, loc_record = {'name':'DAC', 'label':'V_sweep'}, 
                       io = NewIO,)
print('loop.data_set: %s' % LOOP.data_set)

#pt = MatPlot()
#pt.add(x = data.gates_T_set, y = data.gates_LP_set, z = data.keithley_amplitude)
#T(-17.792019531548021)