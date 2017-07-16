# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:54:13 2017

@author: think
"""

import struct
import logging
import warnings

import numpy as np
import array as arr

from time import sleep, localtime
from io import BytesIO

from qcodes import VisaInstrument, validators as vals
from pyvisa.errors import VisaIOError

import temp
awg = temp.awg
AWG_sequence_cfg = {
            'SAMPLING_RATE': awg.get('clock_freq'),
            'CLOCK_SOURCE': (1 if awg.ask('AWGControl:CLOCk:' +
                                           'SOURce?').startswith('INT')
                             else 2),  # Internal | External
            'REFERENCE_SOURCE': (1 if awg.ask('SOURce1:ROSCillator:' +
                                               'SOURce?').startswith('INT')
                                 else 2),  # Internal | External
            'EXTERNAL_REFERENCE_TYPE':   1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,
            # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE':   1 if
            awg.get('trigger_source').startswith('EXT') else 2,
            # External | Internal
            'TRIGGER_INPUT_IMPEDANCE': (1 if awg.get('trigger_impedance') == 50. else 2),  # 50 ohm | 1 kohm
            'TRIGGER_INPUT_SLOPE': (1 if awg.get('trigger_slope').startswith(
                                    'POS') else 2),  # Positive | Negative
            'TRIGGER_INPUT_POLARITY': (1 if awg.ask('TRIGger:' +
                                                     'POLarity?').startswith(
                                       'POS') else 2),  # Positive | Negative
            'TRIGGER_INPUT_THRESHOLD':  awg.get('trigger_level'),  # V
            'EVENT_INPUT_IMPEDANCE':   (1 if awg.get('event_impedance') ==
                                        50. else 2),  # 50 ohm | 1 kohm
            'EVENT_INPUT_POLARITY':  (1 if awg.get('event_polarity').startswith(
                                      'POS') else 2),  # Positive | Negative
            'EVENT_INPUT_THRESHOLD':   awg.get('event_level'),  # V
            'JUMP_TIMING':   (1 if
                              awg.get('event_jump_timing').startswith('SYNC')
                              else 2),  # Sync | Async
            'RUN_MODE':   4,  # Continuous | Triggered | Gated | Sequence
            'RUN_STATE':  0,  # On | Off
            }


