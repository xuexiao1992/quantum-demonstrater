# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 15:50:17 2017

@author: X.X
"""

#%%
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
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element




import sys
sys.path.append('C:\\Users\\LocalAdmin\\Documents\\backup_pycqed\\PycQED_py3\\pycqed\\measurement\\waveform_control')
import pulsar as ps
import element as ele
#%%


server_name = None
station = stationF006.initialize(server_name=server_name)


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
                              high=.7, low=-.7,
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


#%%

pulsar = make5014pulsar(awg = station.components['awg'], awg2 = station.components['awg2'])

# Generating an example sequence
test_element = Element('test_element', pulsar=pulsar)
# we copied the channel definition from out global pulsar
print()



#%%

awg = station.awg
awg2 = station.awg2
awg.clock_freq(1e9)
awg2.clock_freq(1e9)

basechannel = 1
VT_channel = 5
VLP_channel = 8
base_amplitude = 0.25  # some safe value for the sample

sin_pulse = pulse.CosPulse(channel='ch%d' % basechannel, name='A cosine pulse on RF')
sq_pulse = pulse.SquarePulse(channel='ch%d' % basechannel, name='A square pulse on MW pmod1')
sq_pulse_marker = pulse.SquarePulse(channel='ch%d_marker1' % basechannel, name='A square pulse on MW pmod2')
sq_pulse1 = pulse.SquarePulse(channel='ch%d' % VT_channel, name='A square pulse on MW pmod3')
sin_pulse_marker = pulse.CosPulse(channel='ch%d' % VLP_channel, name='A cosine pulse on RF1')


test_element.add(pulse.cp(sin_pulse, frequency=1e6, amplitude=0.5, length=5e-6),
                 name='first pulse')

test_element.add(pulse.cp(sq_pulse1, amplitude=0.2, length=5e-6),
                 name='second pulse marker', refpulse='first pulse', refpoint='start')

test_element.add(pulse.cp(sq_pulse, amplitude=0.3, length=1e-6),
                name='second pulse', refpulse='first pulse', refpoint='end')

test_element.add(pulse.cp(sq_pulse1, amplitude=0.6, length=1e-6),
                name='second pulse1', refpulse='first pulse', refpoint='end')

test_element.add(pulse.cp(sin_pulse_marker, frequency=3e6, amplitude=0.2, length=1e-6),
                name='second pulse2', refpulse='first pulse', refpoint='end')

test_element.add(pulse.SquarePulse(name = 'TRG', channel = 'ch4_marker1', amplitude=2, length=3000e-9),
                 name='trigger', refpulse = 'first pulse', refpoint = 'start')


test_element.add(pulse.cp(sin_pulse, frequency=2e6, amplitude=0.25 / np.sqrt(2), length=1e-6),
                 name='third pulse', refpulse='second pulse', refpoint='end')
test_element.add(pulse.cp(sq_pulse1, amplitude=0.6, length=1e-6),
                name='second pulse4', refpulse='second pulse', refpoint='end')


test_element2 = Element('test2', pulsar = pulsar)
test_element2.add(pulse.cp(sin_pulse, frequency=1e6, amplitude=0.3, length=2e-6),
                 name='first pulse')
test_element2.add(pulse.cp(sq_pulse1, amplitude=0.3, length=2e-6),
                 name='second', refpulse = 'first pulse', refpoint = 'start')
test_element2.add(pulse.SquarePulse(name = 'marker1',amplitude=2, length = 2e-6, channel = 'ch3_marker2'),
                  name = 'marker1', refpulse = 'first pulse', refpoint = 'start')
test_element2.add(pulse.SquarePulse(name = 'marker2',amplitude=2,length = 2e-6, channel = 'ch6_marker1'),
                  name = 'marker2', refpulse = 'first pulse', refpoint = 'start')


test_element3 = Element('test3', pulsar = pulsar)
test_element3.add(pulse.cp(sq_pulse1, amplitude=0.1, length=1e-6),
                 name='firstpulse')
test_element3.add(pulse.cp(sq_pulse_marker, amplitude=0.8, length=1e-6),
                  name='firstpulse1',refpulse = 'firstpulse', refpoint = 'start')
test_element3.add(pulse.SquarePulse(name = 'sq',amplitude = 0.3,length = 1e-6, channel = 'ch2',),
                  name = 'sq',refpulse = 'firstpulse',refpoint = 'start')
trigger_element = Element('trigger', pulsar)

trigger_element.add(pulse.SquarePulse(name = 'TRG2', channel = 'ch8_marker2', amplitude=2, length=70e-9),
                 name='trigger2', )
trigger_element.add(pulse.SquarePulse(name = 'TRG1', channel = 'ch4_marker2', amplitude=2, length=1570e-9),
                 name='trigger1',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)

test_element4 = Element('test4', pulsar)
test_element4.add(pulse.SquarePulse(name = 'TRG1', channel = 'ch4_marker2', amplitude=0, length=1570e-9),
                 name='tri',)

elts = [trigger_element, test_element, test_element2, test_element3]

myseq = Sequence('ASequence')
print('c')

myseq.append(name='trigger', wfname='trigger', trigger_wait=False,)

myseq.append(name='test_element', wfname='test_element', trigger_wait=False,)
myseq.append(name='test2_1', wfname='test2', trigger_wait=False,)
myseq.append(name='test3_1', wfname='test3', trigger_wait=False, )

myseq.append(name='test2_2', wfname='test2', trigger_wait=False,)
myseq.append(name='test3_2', wfname='test3', trigger_wait=False, goto_target = 'test2_1')

awg.delete_all_waveforms_from_list()
awg2.delete_all_waveforms_from_list()

awg.write('SOUR1:ROSC:SOUR EXT')
awg2.write('SOUR1:ROSC:SOUR INT')
#awg.clock_source('EXT')

print('e')
awg.stop()
awg2.stop()
print('f')
#pulsar.program_awg(myseq, *elts)
#awg.run_mode('CONT')
pulsar.program_awgs(myseq, *elts, AWGs = ['awg','awg2'], allow_first_nonzero = False)
awg2.set_sqel_trigger_wait(element_no = 1, state = 1)
awg.set_sqel_goto_target_index(element_no = 6, goto_to_index_no = 2)
awg2.set_sqel_goto_target_index(element_no = 6, goto_to_index_no = 2)


#awg2.run_mode('TRIG')
awg2.trigger_level(0.5)
#
#channels = pulsar.channels.keys()
#AWGs = pulsar.used_AWGs()
#AWG_wfs={}
#
#for el in elts:
#            tvals, waveforms = el.normalized_waveforms()
#            for cname in waveforms:
#                if cname not in channels:
#                    continue
#                if not pulsar.channels[cname]['active']:
#                    continue
#                cAWG = pulsar.channels[cname]['AWG']
#                cid = pulsar.channels[cname]['id']
#                if cAWG not in AWGs:
#                    continue
#                if cAWG not in AWG_wfs:
#                    AWG_wfs[cAWG] = {}
#                if el.name not in AWG_wfs[cAWG]:
#                    AWG_wfs[cAWG][el.name] = {}
#                AWG_wfs[cAWG][el.name][cid] = waveforms[cname]
#
#AWG_obj = pulsar.AWG_obj(AWG = 'awg')
#
#pulsar._program_AWG5014(AWG_obj,myseq,AWG_wfs['awg'])
#v = awg.write('SOUR1:ROSC:SOUR INT')
#w = awg2.write('SOUR1:ROSC:SOUR INT')


#awg2.ch3_state.set(1)
#awg2.force_trigger()
#awg2.run()
#
#
#awg.ch3_state.set(1)
#awg.force_trigger()
#awg.run()

