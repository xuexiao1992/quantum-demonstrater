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
                              high=1, low=-1,
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

## Generating an example sequence
#initialize = Element('initialize', pulsar=pulsar)
## we copied the channel definition from out global pulsar
#print('Channel definitions: ')
#pprint.pprint(initialize._channels)
#print()

plungerchannel = 3
microwavechannel = 2
VLP_channel = 4
base_amplitude = 0.25  # some safe value for the sample
initialize_amplitude = 0.3
readout_amplitude = 0.5




awg = station.awg
print('a')
#elts = [initialize, manipulation, readout]
elts = []

myseq = Sequence('ASequence')

def initialize(num):
    
    initialize = Element('initialize%d' % num, pulsar=pulsar)


    initialize.add(SquarePulse(channel = 'ch%d' % plungerchannel, name = 'initialize_1', amplitude = initialize_amplitude,
                               length = 0.5e-6), name = 'initialize_1')

    initialize.add(SquarePulse(channel = 'ch%d' % plungerchannel, name = 'initialize_2', 
                               amplitude = 0.5*initialize_amplitude, length =0.5e-6,),
                               name= 'initialize_2', refpulse = 'initialize_1', refpoint = 'center')
    
#    initialize.add(CosPulse(channel = 'ch%d' % plungerchannel, name = 'initialize_3', frequency = 1e6,
#                            amplitude = 0.5*initialize_amplitude, length =150e-6,),
#                               name= 'initialize_3', refpulse = 'initialize_1', refpoint = 'end')
    elts.append(initialize)
    myseq.append(name='initialize%d' % num, wfname='initialize%d' % num, repetitions = 350, trigger_wait=False,)
    return initialize

def initialize_add(num):
    initialize_add = Element('initialize_add%d' % num, pulsar=pulsar)
    initialize_add.add(SquarePulse(channel = 'ch%d' % plungerchannel, name = 'initialize_1', amplitude = initialize_amplitude,
                               length = 1e-6), name = 'initialize_1')
    elts.append(initialize_add)
    myseq.append(name='initialize_add%d' % num, wfname='initialize_add%d' % num, repetitions = 100, trigger_wait=False,)
    return initialize_add

def manipulation(num):
    
    manipulation = Element('manipulation%d' % num, pulsar=pulsar)

    dd = [4, 0, 1, 2, 3, 4, 0]
    manipulation.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=0.35, length=30e-6), 
                     name='m0',)
    manipulation.add(SquarePulse(channel = 'ch%d' %microwavechannel, amplitude=0.1, length=1e-6), 
                     name='m0_m', )

    for i, d in enumerate(dd):
        manipulation.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=base_amplitude * d / np.max(dd), length=10e-6),
                         name='m%d' % (i + 1), refpulse='m%d' % i, refpoint='end')
        manipulation.add(SquarePulse(channel = 'ch%d' %microwavechannel, amplitude=2*base_amplitude * d / np.max(dd), length=5e-6),
                         name='m%d_m' % (i + 1), refpulse='m%d_m' % i, refpoint='end')
    elts.append(manipulation)
    myseq.append(name='manipulation%d'%num, wfname='manipulation%d'%num, trigger_wait=False,)
    return manipulation



#manipulation.print_overview()


def readout(num):
    
    readout = Element('readout%d' % num, pulsar=pulsar)
    readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=readout_amplitude, length=1e-6), name = 'readout_0')
    
    for i in range(3):
        readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=readout_amplitude/(i+1), length=0.5e-6), 
                    name = 'readout_%d' %(i+1), refpulse ='readout_%d' % i, refpoint = 'end')
        
    readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=0*readout_amplitude, length=3e-6), 
                name = 'readout_4', refpulse = 'readout_3', refpoint = 'end')
    
    readout.add(CosPulse(channel = 'ch%d' % plungerchannel, name = 'readout_5', frequency = 1e6,
                            amplitude = 0.5*initialize_amplitude, length =1.5e-6,),
                   name= 'readout_5', start = -2e-6, refpulse = 'readout_4', refpoint = 'end')
    
    readout.add(SquarePulse(channel = 'ch%d' % plungerchannel, amplitude=0*readout_amplitude, length=5e-6), 
                name = 'readout_6', refpulse = 'readout_5', refpoint = 'end')


    readout.add(SquarePulse(channel = 'ch%d_marker1'%plungerchannel, amplitude=base_amplitude, length=0.1e-6), 
                name = 'readout_marker1', refpulse ='readout_0', refpoint = 'start')
    readout.add(SquarePulse(channel = 'ch%d_marker2'%plungerchannel, amplitude=base_amplitude, length=0.1e-6), 
                name = 'readout_marker2', refpulse ='readout_0', refpoint = 'start')
    elts.append(readout)
    myseq.append(name='readout%d'%num, wfname='readout%d'%num, reprtitions = 800, trigger_wait=False,)
    return readout
    
#readout.print_overview()
#print('Element overview:')
#initialize.print_overview()
#print()

#myseq.append(name='initialize', wfname='initialize', repetitions = 2, trigger_wait=False,)
#myseq.append(name='manipulation', wfname='manipulation', trigger_wait=False,)
#myseq.append(name='readout', wfname='readout', trigger_wait=False,)


#
#for exp in range(5):
#    for elem in range(5):
#        num = exp*5+elem 
#        initialize(num)
#        manipulation(num)
#        readout(num)
#
#for count in range(20):
#    exp=count*15
#    initialize(exp+1)
#    initialize_add(exp+2)
#    initialize(exp+3)
#    initialize_add(exp+4)
#    initialize(exp+5)
#    manipulation(exp+6)
#    readout(exp+7)
#    initialize(exp+8)
#    initialize_add(exp+9)
#    initialize(exp+10)
#    manipulation(exp+11)
#    readout(exp+12)
#    initialize(exp+13)
#    manipulation(exp+14)
#    readout(exp+15)
#
#a = initialize(1)
#b = initialize_add(2)
#c = initialize(3)
#d = manipulation(4)
e = readout(5)



awg.delete_all_waveforms_from_list()
print('e')
awg.stop()
print('f')
awg_file = pulsar.program_awg(myseq, *elts)
print('r')
v = awg.write('SOUR1:ROSC:SOUR INT')
print('erer')
awg.ch2_state.set(1)
awg.ch3_state.set(1)
awg.force_trigger()
awg.run()
#
#awg.get_sqel_waveform(channel = 2, element_no =1)
#
#awg.set_sqel_waveform(waveform_name = 'ini11_ch3',channel = 2, element_no =1)
#
#awg.send_waveform_to_list(w = wfs['ch3'], m1 = wfs['ch3_marker1'], m2 = wfs['ch3_marker2'], wfmname = 'ini11_ch3')
#awg.send_waveform_to_list(w = wfs['ch3'], m1 = wfs['ch3_marker1'], m2 = wfs['ch3_marker2'], wfmname = '"ini11_ch3"\n')
#awg.send_waveform_to_list()






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