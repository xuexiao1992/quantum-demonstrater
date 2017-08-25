# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% import
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
#from test_awg2 import station
import stationF006


from pycqed.measurement.waveform_control.pulse import SquarePulse, CosPulse, LinearPulse
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#%%
station = stationF006.initialize(server_name=None)
awg = station.awg
digitizer = station.digitizer

def make5014pulsar(awg, awg2):
    AWG = awg
    awg = awg.name
    awg2 = awg2.name
    pulsar = Pulsar(name = 'PuLsAr', default_AWG = awg, master_AWG = awg)
#    pulsar = ps.Pulsar()
#    pulsar.AWG = AWG
    marker1highs = [2, 2, 2.7, 2, 2, 2, 2.7, 2]
    for i in range(8):
        pulsar.define_channel(id='ch{}'.format(i%4 + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=1, low=-1,
                              offset=0.0, delay=0, active=True, AWG = awg if i<4 else awg2)
        pulsar.define_channel(id='ch{}_marker1'.format(i%4 + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True, AWG = awg if i<4 else awg2)
        pulsar.define_channel(id='ch{}_marker2'.format(i%4 + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True, AWG = awg if i<4 else awg2)
    return pulsar


#%% program sequence
pulsar = make5014pulsar(awg = station.components['awg'], awg2 = station.components['awg2'])

initialize = Element(name = 'initialize',pulsar = pulsar)

manipulation = Element(name = 'manipulation', pulsar = pulsar)

readout = Element(name = 'readout',pulsar = pulsar)

initialize.add(SquarePulse(name='init', channel='ch1', amplitude=1, length=100e-6),
               name = 'init')

manipulation.add(SquarePulse(name='manip1',channel = 'ch1',amplitude=0.4, length = 0.5e-6),
                 name = 'manip1')

manipulation.add(CosPulse(name = 'manip2', channel = 'ch1', frequency = 3e6, amplitude=0.3, length = 200e-9),
                 name = 'manip2',refpulse = 'manip1',refpoint = 'start',start = 200e-9)

readout.add(LinearPulse(name='read1', channel='ch1', start_value=0.2,end_value = 0.4, length=50e-6),
            name = 'read1')
readout.add(LinearPulse(name='read1', channel='ch1', start_value=0.4,end_value = 0.5, length=20e-6),
            name = 'read2',refpulse = 'read1', refpoint = 'end')
readout.add(SquarePulse(name = 'trig1', channel = 'ch1_marker1',amplitude = 2, length = 2e-6),
            name = 'trig1',refpulse = 'read1',refpoint = 'start', start = 20e-6)
readout.add(SquarePulse(name = 'trig2', channel = 'ch1_marker1',amplitude = 2, length = 2e-6),
            name = 'trig2',refpulse = 'read2', refpoint = 'start', start = 10e-6)

seq = Sequence(name ='seq',)

seq.append(name ='initialize', wfname = 'initialize', trigger_wait=False)
seq.append(name = 'manipulation', wfname = 'manipulation', trigger_wait = False)
seq.append(name = 'readout', wfname = 'readout', trigger_wait = False)

elts = [initialize, manipulation, readout]

awg.delete_all_waveforms_from_list()

awg.stop()

pulsar.program_awgs(seq,*elts,AWGs=['awg'])
awg.ch1_amp(1)
awg.run()
#%%

pretrigger=16
mV_range=1000
rate = int(np.floor(250000000/1))
#seg_size = int(np.floor((rate * (10e-6))/16)*16 + pretrigger )
seg_size = 1040
memsize = 5*seg_size

posttrigger_size = 1024

#
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
    a = a + 5
    b =3
    aa += a*b
    return aa

def awgfunc(a):
    
    awg.stop()
    awg.run()
    global aaa
    
    aaa = a + 5
    
    return aaa

P = StandardParameter(name = 'Para1', set_cmd = Pfunction)
F = StandardParameter(name = 'Fixed1', get_cmd = Ffunction)

awgpara = StandardParameter(name = 'AWGpara', set_cmd = awgfunc)

Sweep_Value = P[1:5:1]
Sweep_Value2 = awgpara[1:5:1]
LP = Loop(sweep_values = Sweep_Value2).each(dig)

print('loop.data_set: %s' % LP.data_set)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()

#data = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

loop_size = len(Sweep_Value2)
data_arrays = []
loop_array = DataArray(parameter=Sweep_Value2.parameter, is_setpoint=True)
loop_array.nest(size=loop_size)
data_arrays = [loop_array]
actions = [dig]

##      store data by user
data=dig.get()
data1 = DataArray(preset_data = data, name = 'digitizer', array_id = 'digitizer', is_setpoint = True)
data2 = DataArray(preset_data = data, name = 'digitizer2')
data3 = DataArray(preset_data = data, name = 'digitizer3')

data1.ndarray

##
arrays = LP.containers()
arrays2 = []
arrays3 = [data1,]
arrays4 = [data1, data2,data3]
data_set_2 = new_data(arrays=arrays3,location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

data_set_2.save_metadata()

test_location = '2017-08-18/20-40-19_T1_Vread_sweep'

data_set_3 = DataSet(location = test_location, io = NewIO,)
data_set_3.read()
AWGpara_array = data_set_3.arrays['AWGpara_set'].ndarray
index0_array = data_set_3.arrays['index0_set'].ndarray
digitizer_array = data_set_3.arrays['digitizer_digitizer'].ndarray

#
#print('loop.data_set: %s' % LP.data_set)
#
#data = LP.run()
#
