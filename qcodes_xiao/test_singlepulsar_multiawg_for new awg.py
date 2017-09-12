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
station = stationF006.initialize()
awg = station.awg


#%%


def make5014pulsar(awg):
    awg = awg.name
    pulsar = Pulsar()#name = 'PuLsAr',)# default_AWG = awg, master_AWG = awg)
#    pulsar = ps.Pulsar()
    AWG = station.awg
    pulsar.AWG = AWG
    marker1highs = [2, 2, 2.7, 2, 2, 2, 2.7, 2]
    for i in range(4):
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=.7, low=-.7,
                              offset=0.0, delay=0, active=True,)# AWG = awg)
        
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True,)
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True,)
        
    return pulsar


#%%

pulsar = make5014pulsar(awg = station.components['awg'])

# Generating an example sequence
test_element = Element('test_element', pulsar=pulsar)
# we copied the channel definition from out global pulsar
print()



#%%

awg = station.awg
awg.clock_freq(1e9)

basechannel = 1
VT_channel = 2
VLP_channel = 4
base_amplitude = 0.25  # some safe value for the sample

sin_pulse = pulse.CosPulse(channel='ch%d' % basechannel, name='A cosine pulse on RF')
sq_pulse = pulse.SquarePulse(channel='ch%d' % basechannel, name='A square pulse on MW pmod1')
sq_pulse_marker = pulse.SquarePulse(channel='ch%d' % basechannel, name='A square pulse on MW pmod2')
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

test_element.add(pulse.SquarePulse(name = 'TRG', channel = 'ch4', amplitude=2, length=3000e-9),
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
test_element2.add(pulse.SquarePulse(name = 'marker1',amplitude=2, length = 2e-6, channel = 'ch3'),
                  name = 'marker1', refpulse = 'first pulse', refpoint = 'start')
test_element2.add(pulse.SquarePulse(name = 'marker2',amplitude=2,length = 2e-6, channel = 'ch4'),
                  name = 'marker2', refpulse = 'first pulse', refpoint = 'start')


test_element3 = Element('test3', pulsar = pulsar)
test_element3.add(pulse.cp(sq_pulse1, amplitude=0.1, length=1e-6),
                 name='firstpulse')
test_element3.add(pulse.cp(sq_pulse_marker, amplitude=0.8, length=1e-6),
                  name='firstpulse1',refpulse = 'firstpulse', refpoint = 'start')
test_element3.add(pulse.SquarePulse(name = 'sq',amplitude = 0.3,length = 1e-6, channel = 'ch2',),
                  name = 'sq',refpulse = 'firstpulse',refpoint = 'start')
trigger_element = Element('trigger', pulsar = pulsar)

trigger_element.add(pulse.SquarePulse(name = 'TRG2', channel = 'ch3', amplitude=2, length=70e-9),
                 name='trigger2', )
trigger_element.add(pulse.SquarePulse(name = 'TRG1', channel = 'ch4', amplitude=2, length=1570e-9),
                 name='trigger1',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)

test_element4 = Element('test4', pulsar = pulsar)
test_element4.add(pulse.SquarePulse(name = 'TRG1', channel = 'ch4', amplitude=0, length=1570e-9),
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

for i in range(5):
    
    myseq.append(name='test2_%d'%(i+5), wfname='test2', trigger_wait=False,)
    myseq.append(name='test3_%d'%(i+5), wfname='test3', trigger_wait=False, )
    
awg.delete_all_waveforms_from_list()

#awg.write('SOUR1:ROSC:SOUR INT')
#awg.clock_source('EXT')

print('e')
awg.stop()
print('f')
awg.sequence_length(len(myseq.elements))
pulsar.program_awg(myseq, *elts)

#%%

def add_new_element_to_awg_list(awg, element):

    name = element.name

    tvals, wfs = element.normalized_waveforms()

    for i in range(1,5):
        awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                  m2 = wfs['ch%d_marker2'%i], wfmname = name+'_ch%d'%i)

    return True

def add_new_waveform_to_sequence(awg, wfname, element_no, repetitions):

    for i in range(1,5):
        awg.set_sqel_waveform(waveform_name = wfname+'_ch%d'%i, channel = i,
                              element_no = element_no)
        
        awg.set_sqel_loopcnt(loopcount = repetitions, element_no = element_no)
        
    return True


#awg.run_mode('CONT')
#pulsar.program_awgs(myseq, *elts, AWGs = ['awg'], allow_first_nonzero = False)
#awg.set_sqel_goto_target_index(element_no = 6, goto_to_index_no = 2)

#awg2.ch3_state.set(1)
#awg2.force_trigger()
#awg2.run()
#
#
#awg.ch3_state.set(1)
#awg.force_trigger()
#awg.run()

