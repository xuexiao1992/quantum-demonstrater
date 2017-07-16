# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:24:26 2017

@author: think
"""

import math
import numpy as np
from scipy import constants as C

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
from pycqed.measurement.waveform_control.sequence import Sequence

from qubit import Qubit
from manipulation import Manipulation
from Gates import Single_Qubit_Gate
from experiment import Experiment

Qubit_1 = Qubit(name = 'Qubit_1')

Qubit_2 = Qubit(name = 'Qubit_2')

Qubit_1.define_gate(gate_name = 'LP1', gate_number = 1, microwave = 1, channel_I = 'LP1I', channel_Q = 'LP1Q')

Qubit_1.define_gate(gate_name = 'RP1', gate_number = 2, microwave = 1, channel_I = 'RP1I', channel_Q = 'RP1Q')

Qubit_1.define_gate(gate_name = 'Plunger1', gate_number = 3, gate_function = 'plunger', channel_VP = 'P1DC')

Qubit_2.define_gate(gate_name = 'LP2', gate_number = 4, microwave = 1, channel_I = 'LP2I', channel_Q = 'LP2Q')

Qubit_2.define_gate(gate_name = 'Plunger2', gate_number = 5, gate_function = 'plunger', channel_VP = 'P2DC')


Manip_1 = Manipulation(name = 'Manip_1', qubits_name = ['Qubit_1', 'Qubit_2'])

Manip_1.add_single_qubit_gate(name = 'X1_Q1', qubit = Qubit_1)

Manip_1.add_single_qubit_gate(name = 'X1_Q2', refgate = 'X1_Q1', qubit = Qubit_2)

Manip_1.add_X(name = 'X2_Q1', refgate = 'X1_Q2', qubit = Qubit_1)

Manip_1.add_Z(name = 'Z1_Q1', refgate = 'X2_Q1', qubit = Qubit_1, degree = np.pi/4)

Manip_1.add_X(name = 'X3_Q1', refgate = 'X2_Q1', qubit = Qubit_1)

