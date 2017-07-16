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
import pycqed.measurement.waveform_control.pulse as pulse
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.waveform_control.element as elem
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

Qubit_1 = Qubit(name = 'Qubit_1')

#Qubit_2 = Qubit(name = 'Qubit_2')

Qubit_1.define_gate(gate_name = 'LP1', gate_number = 1, microwave = 1, channel_I = 'ch4', channel_Q = 'ch2')

#Qubit_1.define_gate(gate_name = 'RP1', gate_number = 2, microwave = 1, channel_I = 'RP1I', channel_Q = 'RP1Q')
#
Qubit_1.define_gate(gate_name = 'Plunger1', gate_number = 3, gate_function = 'plunger', channel_VP = 'ch3')
#
#Qubit_2.define_gate(gate_name = 'LP2', gate_number = 4, microwave = 1, channel_I = 'LP2I', channel_Q = 'LP2Q')
#
#Qubit_2.define_gate(gate_name = 'Plunger2', gate_number = 5, gate_function = 'plunger', channel_VP = 'P2DC')


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



Manip_1.add_single_qubit_gate(name = 'X1_Q1', qubit = Qubit_1,)

#Manip_1.add_single_qubit_gate(name = 'X1_Q2', refgate = 'X1_Q1', qubit = Qubit_2)

Manip_1.add_X(name = 'X2_Q1', refgate = 'X1_Q1', qubit = Qubit_1)

#Manip_1.add_Z(name = 'Z1_Q1', refgate = 'X2_Q1', qubit = Qubit_1, degree = np.pi/4)
#
#Manip_1.add_X(name = 'X3_Q1', refgate = 'X2_Q1', qubit = Qubit_1)
elts = [Manip_1]

myseq = Sequence('ASequence')

myseq.append(name='Manip_1', wfname='Manip_1', trigger_wait=False,)

awg.delete_all_waveforms_from_list()
print('e')
awg.stop()
print('f')
pulsar.program_awg(myseq, *elts)
print('g')
v = awg.write('SOUR1:ROSC:SOUR INT')
print('h')
awg.ch2_state.set(1)
awg.force_trigger()
awg.run()


