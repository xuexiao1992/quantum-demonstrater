# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:23:48 2018

@author: TUD278306
"""

G = station.gates
T = G.T
LP = G.LP


AMP = keithley.amplitude

T_factor = 1
LP_factor = 1

experiment.saveraw = True
experiment.readout_time = 0.0012
experiment.threshold = 0.040
experiment.seq_repetition = 100

