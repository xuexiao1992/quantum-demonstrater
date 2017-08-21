# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:36:09 2017

@author: X.X
"""
#%%
import logging
import numpy as np
import ctypes as ct

import qcodes as qc
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
import stationF006
from pycqed.measurement.waveform_control.pulse import SquarePulse, CosPulse, LinearPulse
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element


#%%     digitizer parameter

digitizer = stationF006.digitizer

pretrigger=16
mV_range=1000
rate = int(np.floor(250000000/1))
#seg_size = int(np.floor((rate * (10e-6))/16)*16 + pretrigger )
seg_size = 1040
memsize = 5*seg_size

posttrigger_size = 1024
#digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)

#digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)

digitizer.enable_channels(pyspcm.CHANNEL1 | pyspcm.CHANNEL2)

digitizer.data_memory_size(memsize)

digitizer.segment_size(seg_size)

digitizer.posttrigger_memory_size(posttrigger_size)

digitizer.timeout(60000)

digitizer.sample_rate(250000000)

digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)

#trig_mode = pyspcm.SPC_TMASK_SOFTWARE

#trig_mode = pyspcm.SPC_TM_POS

trig_mode = pyspcm.SPC_TM_POS | pyspcm.SPC_TM_REARM

digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 800, level1 = 900)

#%% digitizer parameter
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


dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size )
