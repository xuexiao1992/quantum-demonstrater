# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:46:55 2017

@author: X.X
"""
import numpy as np
from qcodes import combine
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
keithley.nplc(1)
digitizer = station.digitizer

T = G.T
LP = G.LP
SQD3 = G.SQD3
SQD1 = G.SQD1
RP = G.RP

AMP = keithley.amplitude
#%%

pretrigger=16
mV_range=1000
sample_rate = int(np.floor(61035/1))

digitizer.sample_rate(sample_rate)

sample_rate = digitizer.sample_rate()

readout_time = 0.5

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
def Counts(x):
    return True

Count = StandardParameter(name = 'Count', set_cmd = Counts)

#%%
#
#Sweep_Value1 = T[-16:-24:0.1]
#Tvals = np.linspace(-14,-24, 81);
#LPvals = np.linspace(-355,-350, 81)

#vsd = 40

#LP()
#Out[246]: -356.17608911268803
#
#T()
#Out[247]: -14.618142977035177
'''
Tvals = np.linspace(-13,-19, 81);
LPvals = np.linspace(-355.6,-353.6, 81)
Sweep_Value2 = Count[0:10:1]

array = np.zeros([81,2])
array[:,0] = Tvals
array[:,1] = LPvals
combined = combine(T, LP, name = "T_and_LP")
LOOP = Loop(sweep_values = Sweep_Value2).loop(combined.sweep(array), delay = 0.1).each(DIG)
'''
#
#keithley.nplc(10)
#LOOP = Loop(sweep_values = Sweep_Value2).loop(combined.sweep(array), delay = 0.1).each(AMP)
#
'''
for normal experiemnt:
    SQD1: -37
    SQD3: -243
    T: -15
    LP: -351

'''
##
#Sweep_Value1 = T[-5:-25:0.5]
#Sweep_Value2 = LP[-340:-360:0.5]
Sweep_Value1 = T[0:-75:1]
Sweep_Value2 = LP[-320:-400:1]

#Sweep_Value1 = T[20:-100:1]
#Sweep_Value2 = LP[-100:-400:1]

#Sweep_Value1 = T[-20:-80:1]
#Sweep_Value1 = T[-80:-20:1]
#Sweep_Value2 = LP[-110:-260:1]


#Sweep_Value1 = T[-40:-50:0.5]
#Sweep_Value2 = LP[-210:-230:0.5]

Sweep_Value1 = T[-10:-30:0.2]
Sweep_Value2 = LP[-340:-360:0.2]

#Sweep_Value1 = RP[-1025:-950:1]
#Sweep_Value2 = T[-25:-10:1]
##
#Sweep_Value1 = SQD1[0:-100:1]
#Sweep_Value2 = SQD3[-200:-300:1]
#
##
LOOP = Loop(sweep_values = Sweep_Value2).loop(sweep_values = Sweep_Value1).each(AMP)

#LOOP = Loop(sweep_values = Sweep_Value1).each(DIG)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')


## get_data_set should contain parameter like io, location, formatter and others
data = LOOP.get_data_set(location=None, loc_record = {'name':'DAC', 'label':'V_sweep'}, 
                       io = NewIO,)
print('loop.data_set: %s' % LOOP.data_set)
'''
pt = MatPlot()
pt.add(x = data.gates_T_set, y = data.gates_LP_set, z = data.keithley_amplitude)

pt = MatPlot()
pt.add(x = data.gates_SQD1_set, y = data.gates_SQD3_set, z = data.keithley_amplitude)

pt = MatPlot()
pt.add(x = data.gates_T, y =data.keithley_amplitude)


#T(-17.792019531548021)
formatter = HDF5FormatMetadata()
data.formatter = formatter
data.write()



G.B(-315)
G.LD(-240)
G.LP(-351)
G.LPF(0)
G.LS(26)
G.RD(-1000)
G.RP(-1000)
G.RPF(0)
G.RQPC(0)
G.RS(-400)
G.SQD1(-27)
G.SQD2(0)
G.SQD3(-251)
G.T(-13.2)
G.VI1(0)
G.VI2(120)
G.acQD(77)
G.acres(160)
'''