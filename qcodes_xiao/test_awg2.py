# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:21:01 2017

@author: think
"""

#%% load modules
import numpy as np
import time
import numpy as np
import pprint
import imp
from imp import reload

import qtt
import qcodes
from qcodes.plots.qcmatplotlib import MatPlot
import pycqed
#import stationV2
import stationF006

from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control.sequence import Sequence

from pycqed.measurement.waveform_control import element
import pycqed.measurement.waveform_control
reload(pycqed.measurement.waveform_control)
import pycqed.measurement.waveform_control.pulsar
reload(pycqed.measurement.waveform_control.pulsar)
from pycqed.measurement.waveform_control import pulsar as ps

from qcodes.instrument.parameter import ArrayParameter
from qcodes.instrument.sweep_values import SweepFixedValues

#%% set directory for data saving
#datadir = r'W:\data'
#qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
#qcodes.DataSet.default_formatter = qcodes.data.hdf5_format.HDF5Format()

#station = qcodes.station.Station()

server_name = None
station = stationF006.initialize(server_name=server_name)
#temp.initialize(server_name=server_name)

def make5014pulsar(awg, awg2):
    pulsar = ps.Pulsar()
    pulsar.AWG = awg
    pulsar2 = ps.Pulsar()
    pulsar2.AWG = awg2

    marker1highs = [2, 2, 2.7, 2, 2, 2, 2.7, 2]
#    marker2highs = [2, 2, 2.7, 2]
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              # max safe IQ voltage
                              high=.7, low=-.7,
                              offset=0.0, delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
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
    for i in range(4):
        pulsar2.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 5), type='analog',
                              high=.7, low=-.7,
                              offset=0.0, delay=0, active=True)
        pulsar2.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 5),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True)
        pulsar2.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 5),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True)

        pulsar2.AWG_sequence_cfg = {
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

    return pulsar, pulsar2

pulsar, pulsar2 = make5014pulsar(awg = station.components['awg'], awg2 = station.components['awg2'])
station.pulsar = pulsar
station.pulsar2 = pulsar2





# Generating an example sequence
test_element = element.Element('test_element', pulsar=pulsar)
# we copied the channel definition from out global pulsar
print('Channel definitions: ')
pprint.pprint(test_element._channels)
print()


#%% define some bogus pulses.

basechannel = 3
VT_channel = 2
VLP_channel = 4
base_amplitude = 0.25  # some safe value for the sample

sin_pulse = pulse.CosPulse(channel='ch%d' % basechannel, name='A cosine pulse on RF')
#sq_pulse = pulse.SquarePulse(channels=['ch%d' %VT_channel, 'ch%d' %VLP_channel] , name='A square pulse on MW pmod')

sq_pulse = pulse.SquarePulse(channel='ch%d' % basechannel, name='A square pulse on MW pmod1')
sq_pulse_marker = pulse.SquarePulse(channel='ch%d_marker1' % basechannel, name='A square pulse on MW pmod2')
sq_pulse1 = pulse.SquarePulse(channel='ch%d' % VT_channel, name='A square pulse on MW pmod3')
sin_pulse_marker = pulse.CosPulse(channel='ch%d_marker2' % basechannel, name='A cosine pulse on RF1')

#special_pulse = pulse.CosPulse(channel='ch%d' % basechannel, name='special pulse')
#special_pulse.amplitude = base_amplitude
#special_pulse.length = 2e-6
#special_pulse.frequency = 10e6
#special_pulse.phase = 0

# create a few of those
test_element.add(pulse.cp(sin_pulse, frequency=1e6, amplitude=base_amplitude / np.sqrt(2), length=1e-6),
                 name='first pulse')
test_element.add(pulse.cp(sq_pulse1, amplitude=.5 * base_amplitude, length=2e-6),
                 name='second pulse marker', refpulse='first pulse', refpoint='start')
test_element.add(pulse.cp(sq_pulse, amplitude=0.3, length=1e-6),
                name='second pulse', refpulse='first pulse', refpoint='end')
test_element.add(pulse.cp(sq_pulse1, amplitude=0.2, length=1e-6),
                name='second pulse1', refpulse='first pulse', refpoint='end')


test_element.add(pulse.cp(sin_pulse, frequency=2e6, amplitude=0.25 / np.sqrt(2), length=1e-6),
                 name='third pulse', refpulse='second pulse', refpoint='end')

my_element = element.Element('my_element', pulsar=pulsar)
dd = [0, 0, 1, 2, 3, 4, 0]
my_element.add(pulse.cp(sq_pulse, amplitude=0.35, length=1e-6), name='d0', )
my_element.add(pulse.cp(sin_pulse_marker, amplitude=0.1, length=1e-6), name='d0_marker2', )

for i, d in enumerate(dd):
    my_element.add(pulse.cp(sq_pulse, amplitude=base_amplitude * d / np.max(dd), length=1e-6),
                   name='d%d' % (i + 1), refpulse='d%d' % i, refpoint='end')
    my_element.add(pulse.cp(sq_pulse_marker, amplitude=2, length=1e-6),
                   name='d%d_marker' % (i + 1), refpulse='d%d' % i, refpoint='end')

my_element.print_overview()

your_element = element.Element('your_element', pulsar=pulsar)
your_element.add(pulse.cp(sq_pulse, amplitude=0.25, length=1e-6), name = 'your0')

for i in range(3):
    your_element.add(pulse.cp(sq_pulse, amplitude=base_amplitude/(i+1), length=1e-6), name = 'your%d' %(i+1), refpulse ='your%d' % i, refpoint = 'end')
    your_element.add(pulse.cp(sq_pulse_marker, amplitude=base_amplitude/(i+1), length=1e-6), name = 'your%d_marker' %(i+1), refpulse ='your%d' % i, refpoint = 'end')
    
your_element.print_overview()
print('Element overview:')
test_element.print_overview()
print()



awg = station.awg
awg2 = station.awg2
print('a')
elts = [test_element, my_element, your_element]
print('b')
myseq = Sequence('ASequence')
print('c')
myseq.append(name='test_element', wfname='test_element', trigger_wait=False,)
myseq.append(name='my_element', wfname='my_element', trigger_wait=False,)
myseq.append(name='your_element', wfname='your_element', trigger_wait=False,)
myseq.append(name='test_element1', wfname='test_element', trigger_wait=False,)

#print('d')
#awg.delete_all_waveforms_from_list()
#awg2.delete_all_waveforms_from_list()
#
#print('e')
#awg.stop()
#awg2.stop()
#print('f')
##pulsar.program_awg(myseq, *elts)
#pulsar2.program_awg(myseq, *elts)
#
#v = awg.write('SOUR1:ROSC:SOUR INT')
#w = awg2.write('SOUR1:ROSC:SOUR INT')
#
#
#
#awg.ch3_state.set(1)
#awg.force_trigger()
#awg.run()
#
#awg2.ch3_state.set(1)
#awg2.force_trigger()
#awg2.run()





def ideal_waveforms(element = my_element):
        wfs = {}
        tvals = np.arange(element.samples())/element.clock

        for c in element._channels:
            wfs[c] = np.zeros(element.samples()) + element._channels[c]['offset']
        # we first compute the ideal function values
        for p in element.pulses:
            psamples = element.pulse_samples(p)
            print('psamples', psamples)
            if not element.global_time:
                pulse_tvals = tvals.copy()[:psamples]
                print('pulse_tvals', pulse_tvals)
                pulsewfs = element.pulses[p].get_wfs(pulse_tvals)
                print('pulsewfs', pulsewfs)
            else:
                chan_tvals = {}
                for c in element.pulses[p].channels:
                    idx0 = element.pulse_start_sample(p, c)
                    idx1 = element.pulse_end_sample(p, c) + 1
                    c_tvals = np.round(tvals.copy()[idx0:idx1] +
                                       element.channel_delay(c) +
                                       element.time_offset,
                                       ps.SIGNIFICANT_DIGITS)
                    chan_tvals[c] = c_tvals
                if p == 'd0':
                    print('chan_tvals', chan_tvals) 

                pulsewfs = element.pulses[p].get_wfs(chan_tvals)
                if p == 'd0':
                    print('pulsewfs', pulsewfs) 
                    
            if p == 'd0':
                print('wfs', wfs)
                
            for c in element.pulses[p].channels:
                idx0 = element.pulse_start_sample(p, c)
                idx1 = element.pulse_end_sample(p, c) + 1
                if p == 'd0':
                    print('wfs2', wfs)
                wfs[c][idx0:idx1] += pulsewfs[c]
                if p == 'd0':
                    print('wfs3', wfs)
                    print('pulsewfs', pulsewfs)

        return tvals, wfs


class test:
    def __init__(self,a = 119):
        self.a = a
        self.b = 0
        
    def __call__(self):
        self.b += 15
        return 0