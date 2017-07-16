# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:21:14 2017

@author: think
"""

#import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum.M4i import M4i
import pprint
import temp
import numpy as np
import qcodes.instrument_drivers.tektronix.AWG5014 as AWG
import pyspcm
from qcodes.loops import Loop, ActiveLoop
from qcodes.instrument.sweep_values import SweepFixedValues
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control.pulse import SquarePulse, CosPulse 
#import M4i
#import qcodes.instrument_drivers.Spectrum.M4i as M4i
#from users.boterjm.Drivers.Spectrum.M4i import M4i
#import users.boterjm.Drivers.Spectrum.pyspcm as pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.io import DiskIO

#


server_name = None
station = temp.initialize(server_name=server_name)
#temp.initialize(server_name=server_name)

def make5014pulsar(awg):
    pulsar = ps.Pulsar()
    pulsar.AWG = awg

    marker1highs = [2, 2, 2.7, 2]
    marker2highs = [2, 2, 2.7, 2]
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              # max safe IQ voltage
                              high=0.5, low=-0.5,
                              offset=0.0, delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2.0, low=0, offset=0.,
                              delay=0, active=True)

        pulsar.AWG_sequence_cfg = {
            'SAMPLING_RATE': 1e9,
            'CLOCK_SOURCE': 1,  # Internal | External
            'REFERENCE_SOURCE': 1,  # Internal | External
            'EXTERNAL_REFERENCE_TYPE': 1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,  # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE': 1,  # External | Internal
            'TRIGGER_INPUT_IMPEDANCE': 1,  # 50 ohm | 1 kohm
            'TRIGGER_INPUT_SLOPE': 1,  # Positive | Negative
            'TRIGGER_INPUT_POLARITY': 1,  # Positive | Negative
            'TRIGGER_INPUT_THRESHOLD': 0.6,  # V
            'EVENT_INPUT_IMPEDANCE': 2,  # 50 ohm | 1 kohm
            'EVENT_INPUT_POLARITY': 1,  # Positive | Negative
            'EVENT_INPUT_THRESHOLD': 1.4,  # V
            'JUMP_TIMING': 1,  # Sync | Async
            'RUN_MODE': 4,  # Continuous | Triggered | Gated | Sequence
            'RUN_STATE': 0,  # On | Off
        }

    return pulsar

pulsar = make5014pulsar(station.components['awg'])
station.pulsar = pulsar

# Generating an example sequence
initialize = Element('initialize', pulsar=pulsar)
# we copied the channel definition from out global pulsar
print('Channel definitions: ')
pprint.pprint(initialize._channels)
print()

plungerchannel = 3
microwavechannel = 2
VLP_channel = 4
base_amplitude = 0.25  # some safe value for the sample
initialize_amplitude = 0.3
readout_amplitude = 0.5


initialize.add(SquarePulse(channel = 'ch%d' % plungerchannel, name = 'initialize_1', amplitude = initialize_amplitude,
                           length = 2e-6), name = 'initialize_1')

initialize.add(SquarePulse(channel = 'ch%d' % plungerchannel, name = 'initialize_2', 
                           amplitude = 0.5*initialize_amplitude, length =0.5e-6,),
                           name= 'initialize_2', refpulse = 'initialize_1', refpoint = 'center')

initialize.add(CosPulse(channel = 'ch%d' % plungerchannel, name = 'initialize_3', frequency = 1e6,
                           amplitude = 0.5*initialize_amplitude, length =1.5e-6,),
                           name= 'initialize_3', refpulse = 'initialize_1', refpoint = 'end')



manipulation = Element('manipulation', pulsar=pulsar)

dd = [4, 0, 1, 2, 3, 4, 0]
manipulation.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=0.35, length=1e-6), 
                 name='m0',)
manipulation.add(SquarePulse(channel = 'ch%d' %microwavechannel, amplitude=0.1, length=1e-6), 
                 name='m0_m', )

for i, d in enumerate(dd):
    manipulation.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=base_amplitude * d / np.max(dd), length=1e-6),
                   name='m%d' % (i + 1), refpulse='m%d' % i, refpoint='end')
    manipulation.add(SquarePulse(channel = 'ch%d' %microwavechannel, amplitude=2*base_amplitude * d / np.max(dd), length=1e-6),
                   name='m%d_m' % (i + 1), refpulse='m%d_m' % i, refpoint='end')

manipulation.print_overview()

readout = Element('readout', pulsar=pulsar)
readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=readout_amplitude, length=5e-6), name = 'readout_0')

for i in range(3):
    readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=readout_amplitude/(i+1), length=5e-6), 
                name = 'readout_%d' %(i+1), refpulse ='readout_%d' % i, refpoint = 'end')


  
readout.add(SquarePulse(channel = 'ch%d_marker1'%plungerchannel, amplitude=base_amplitude/(i+1), length=1e-6), 
            name = 'readout_marker1', refpulse ='readout_0', refpoint = 'start')
readout.add(SquarePulse(channel = 'ch%d_marker2'%plungerchannel, amplitude=base_amplitude/(i+1), length=1e-6), 
            name = 'readout_marker2', refpulse ='readout_0', refpoint = 'start')
    
readout.print_overview()
print('Element overview:')
initialize.print_overview()
print()

awg = station.awg
print('a')
elts = [initialize, manipulation, readout]
#elts = [initialize]

print('b')
myseq = Sequence('ASequence')
print('c')
myseq.append(name='initialize', wfname='initialize', repetitions = 2, trigger_wait=False,)
myseq.append(name='manipulation', wfname='manipulation', trigger_wait=False,)
myseq.append(name='readout', wfname='readout', trigger_wait=False,)

awg.delete_all_waveforms_from_list()
print('e')
awg.stop()
print('f')
awg_file = pulsar.program_awg(myseq, *elts)

v = awg.write('SOUR1:ROSC:SOUR INT')

awg.ch2_state.set(1)
awg.ch3_state.set(1)
awg.force_trigger()
awg.run()












"""

digitizer = station.digitizer
#
#digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL1 | pyspcm.CHANNEL2 | pyspcm.CHANNEL3)

digitizer.enable_channels(pyspcm.CHANNEL1)

digitizer.data_memory_size(2048)

digitizer.posttrigger_memory_size(1024)

digitizer.timeout(60000)

digitizer.sample_rate(200000)


class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, 
                posttrigger_size, label=None, unit=None, instrument=None,
                **kwargs):
       
        super().__init__(name=name, shape=(memsize,), instrument=instrument, **kwargs)
        
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        
    def get(self):
        
        res = digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
        return res
        
    def __getitem__(self, keys):
        return SweepFixedValues(self, keys)
    
    
    
    
pretrigger=16
mV_range=1000
memsize = 2048
rate = int(np.floor(250000000/1))
seg_size = int(np.floor((rate * (10e-6))/16)*16 + pretrigger )
posttrigger_size = 1024
dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size )


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

#TEST = StandardParameter(name = 'testsweep', set_cmd = )

Sweep_Value = P[1:5:1]

LP = Loop(sweep_values = Sweep_Value).each(dig)

print('loop.data_set: %s' % LP.data_set)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()


data = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, 
                       io = NewIO, )

print('loop.data_set: %s' % LP.data_set)


LP.run()
"""