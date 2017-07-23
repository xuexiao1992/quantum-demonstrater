# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:57:10 2017
@author: steph
"""

#import stations and all libs there in.
import temp
station = temp.initialize()



#%%
import numpy as np 
from pycqed.measurement.waveform_control.pulse import SquarePulse, CosPulse
import pycqed.measurement.waveform_control.pulsar as ps
from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.sequence import Sequence
from qubit import Qubit
from manipulation import Manipulation
from experiment import Experiment
from calibration import Calibration
import pprint


def make5014pulsar(awg):
    pulsar = ps.Pulsar()
    pulsar.AWG = awg
    
    #set max volatage of the markers.
    marker1highs = [2, 2, 2, 2]
    marker2highs = [2, 2, 2, 2]
    
    #init the 4 channels of the AWG.
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
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
                              high=marker2highs[i], low=0, offset=0.,
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

#%% MAKE test segment.

awg = temp.awg





initialize = Element('initialize', pulsar=pulsar)
# we copied the channel definition from out global pulsar
print('Channel definitions: ')
pprint.pprint(initialize._channels)
print()

plungerchannel = 4
Ichannel = 2
Qchannel = 3
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




Qubit_1 = Qubit(name = 'Qubit_1')

Qubit_2 = Qubit(name = 'Qubit_2')

Qubit_1.define_gate(gate_name = 'LP1', gate_number = 1, microwave = 1, channel_I = 'ch2', channel_Q = 'ch3', channel_PM = 'ch2_marker1')

#Qubit_1.define_gate(gate_name = 'RP1', gate_number = 2, microwave = 1, channel_I = 'RP1I', channel_Q = 'RP1Q')
#
Qubit_1.define_gate(gate_name = 'Plunger1', gate_number = 3, gate_function = 'plunger', channel_VP = 'ch4')
#
Qubit_2.define_gate(gate_name = 'LP2', gate_number = 4, microwave = 1, channel_I = 'LP2I', channel_Q = 'LP2Q', channel_PM = 'ch1_marker1')
#
Qubit_2.define_gate(gate_name = 'Plunger2', gate_number = 5, gate_function = 'plunger', channel_VP = 'ch1')


#cali = Calibration(name = 'calibration20170619', qubits_name = ['Qubit_1', 'Qubit_2'], qubit = Qubit_1, awg = station.components['awg'], pulsar = pulsar)
#
#cali.Sweep_1D(parameter = 'length', start = 1e-6, stop = 1.5e-6, points = 41)
#
#cali.run_Ramsey()
#

Manip_1 = Manipulation(name = 'Manip_1', qubits_name = ['Qubit_1', 'Qubit_2'], pulsar = pulsar)

print('Channel definitions: ')
pprint.pprint(Manip_1._channels)
print()



Manip_1.add_single_qubit_gate(name = 'X1_Q1', qubit = Qubit_1, axis = [0,1,0], degree = 180, refphase = 0)

Manip_1.add_single_qubit_gate(name = 'Y1_Q1', refgate = 'X1_Q1', waiting_time = 100e-9, qubit = Qubit_1, axis = [0,1,0], degree = 90 ,refphase = 0)

Manip_1.add_Y(name = 'Y2_Q1', refgate = 'Y1_Q1', waiting_time = 100e-9, qubit = Qubit_1)
##
###Manip_1.add_Z(name = 'Z1_Q1', refgate = 'X2_Q1', qubit = Qubit_1, degree = np.pi/4)
###
Manip_1.add_X(name = 'X2_Q1', refgate = 'Y2_Q1', waiting_time = 100e-9, qubit = Qubit_1)
#
Manip_1.add_X(name = 'X3_Q1', refgate = 'X2_Q1', waiting_time = 100e-9,qubit = Qubit_1)
#
Manip_1.add_CNot(name = 'CP1_Q1Q2', refgate = 'X3_Q1', waiting_time = 3e-6, control_qubit = Qubit_2, target_qubit = Qubit_1,)
#Manip_1.add_CRotation(name = 'CR1_Q1Q2',refgate = 'CP1_Q1Q2', waiting_time = 3e-6, control_qubit = Qubit_2, target_qubit = Qubit_1,)

Manip_2 = Manipulation(name = 'Manip_2', qubits_name = ['Qubit_1', 'Qubit_2'], pulsar = pulsar)

#Manip_2.add_CRotation(name = 'CP1_Q1Q2', control_qubit = Qubit_2, target_qubit = Qubit_1,)
Manip_2.add_CNot(name = 'CP1_Q1Q2', control_qubit = Qubit_2, target_qubit = Qubit_1,)

#Manip_1.add_Y(name = 'Y3_Q1', refgate = 'X3_Q1', qubit = Qubit_1)



#elts = [initialize, Manip_1]

elts = [initialize, Manip_1, Manip_2]

myseq = Sequence('ASequence')

myseq.append(name='initialize', wfname='initialize', trigger_wait=False,)
#
myseq.append(name='Manip_1', wfname='Manip_1', trigger_wait=False,)
myseq.append(name = 'Manip_2', wfname = 'Manip_2', trigger_wait = False)

awg.delete_all_waveforms_from_list()
print('e')
awg.stop()
print('f')
pulsar.program_awg(myseq, *elts)
print('g')
v = awg.write('SOUR1:ROSC:SOUR INT')
print('h')
awg.ch2_state.set(1)
awg.ch3_state.set(1)
awg.ch4_state.set(1)
awg.force_trigger()
awg.run()


