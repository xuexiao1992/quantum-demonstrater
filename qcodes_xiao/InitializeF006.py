# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:23:48 2018

@author: TUD278306
"""

import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from experiment_version2 import Experiment
#from experiment_version2_1 import Experiment as Experiment_v2_1
from experiment_version2_2 import Experiment as Experiment_v2_2
from calibration_version2 import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability, seperate_data, organise

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY, Ramsey_all, AllXY_all, CPhase_Calibrate, Charge_Noise, DCZ, Sychpulses1, Sychpulses2, Rabi_all, Wait, MeasureTminus, Ramsey_00_11_basis, Rabi_detuning, Ramsey_withnoise
#from manipulation_library import RB, Rabi_detuning, RB_all,RB_Marcus, Ramsey_withnoise, MultiCPhase_Calibrate, RBinterleavedCZ
#from digitizer_setting import digitizer_param

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter, Parameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot

from plot_functions import plot1D, plot2D, fitcos, plot1Ddata, plot2Ddata

from mpldatacursor import datacursor

import sys
sys.path.append('C:\\Users\\LocalAdmin\\Documents\\GitHub\\PycQED_py3\\pycqed\\measurement\\waveform_control')

import pulsar as ps
import element as ele

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import os
import time
import threading
import multiprocessing

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['font.size'] = 10

#%% Pulsar
def set_5014pulsar(awg, awg2):
    
    awg = awg.name
    awg2 = awg2.name
#    del pulsar
    pulsar = Pulsar(name = 'PuLsAr', default_AWG = awg, master_AWG = awg)

    marker1highs = [2, 2, 2, 2, 2, 2, 2, 2]
    for i in range(8):
        pulsar.define_channel(id='ch{}'.format(i%4 + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=2, low=-2,
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

#%% Initialize
station = stationF006.initialize()
print('station finished')
pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)

#%% Create shortcuts for easier parameter access
vsg = station.vsg
vsg2 = station.vsg2

time.sleep(0.2)

awg = station.awg
awg2 = station.awg2

digitizer = station.digitizer

qubit_1 = station.qubit_1
qubit_2 = station.qubit_2
qubits = [qubit_1, qubit_2]

G = station.gates
T = G.T
LP = G.LP

T_factor = 1
LP_factor = 1

keithley = station.keithley
AMP = keithley.amplitude

#%%
Count = Parameter(name = 'Count', set_cmd = Counts)