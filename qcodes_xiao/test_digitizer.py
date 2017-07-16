# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
from qcodes.data.io import DiskIO

#from test_awg2 import station
import temp


logging.info('LD400: load digitizer driver')
digitizer = M4i.M4i(name='digitizer', server_name=None)
if digitizer==None:
    print('Digitizer driver not laoded')
else:
    print('Digitizer driver loaded')
print('')

    
logging.info('all drivers have been loaded')
#digitizer = temp.station.digitizer

#
#digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)

#digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)

digitizer.enable_channels(pyspcm.CHANNEL1)

digitizer.data_memory_size(4128)

digitizer.segment_size(2064)

digitizer.posttrigger_memory_size(2048)

digitizer.timeout(60000)

digitizer.sample_rate(250000000)

digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)

#trig_mode = pyspcm.SPC_TMASK_SOFTWARE

trig_mode = pyspcm.SPC_TM_POS

#trig_mode = pyspcm.SPC_TM_REARM

digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 2500, level1 = None)


class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, 
                posttrigger_size, label=None, unit=None, instrument=None,
                **kwargs):
       
        super().__init__(name=name, shape=(1*memsize,), instrument=instrument, **kwargs)
        
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


pretrigger=16
mV_range=1000
rate = int(np.floor(250000000/1))
#seg_size = int(np.floor((rate * (10e-6))/16)*16 + pretrigger )
seg_size = 2064
memsize = 2*seg_size

posttrigger_size = 2048
dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size )
#c = dig.get()


aaa=4

def Pfunction(a):
    global aaa
    
    aaa = a + 5
    
    return aaa

aa=5

def Ffunction():
    global aa
    a=0
    a = a +5
    b =3
    aa += a*b
    return aa


P = StandardParameter(name = 'Para1', set_cmd = Pfunction)
F = StandardParameter(name = 'Fixed1', get_cmd = Ffunction)

Sweep_Value = P[1:5:1]

LP = Loop(sweep_values = Sweep_Value).each(dig)

print('loop.data_set: %s' % LP.data_set)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()


data = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, 
                       io = NewIO, )

print('loop.data_set: %s' % LP.data_set)


LP.run()





