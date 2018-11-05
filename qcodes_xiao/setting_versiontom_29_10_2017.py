# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
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

#%% make experiment

def set_step(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

    return step


def set_manip(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    parameter1 = kw.pop('parameter1', 0)
    parameter2 = kw.pop('parameter2', 0)
    manip = kw.pop('manip_elem', 0)

    step = {'time' : time}
    step.update(kw)

    step['manip_elem'] = manip
    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

        step['parameter1'] = parameter1
        step['parameter2'] = parameter2

    return step

def save_object(obj, obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj
'''
save_object(dig_data, obj_name = 'test_dig')
test = load_object('test_dig')
'''
#%%  Sweep

def sweep_array(start, stop, points):

    sweep_array = np.linspace(start, stop, points)

    return list(sweep_array)

#%%
def make_manipulation_cfg():

    manipulation_cfg = {
            'gate1': ['X','Y'],
            'gate2': ['Y','X'],
            'gate3': ['CPhase'],
            'gate4': ['Z', 'X']
            }

    return manipulation_cfg

#%% make pulsar

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

#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def Counts(x):
    return True
#pars, pcov = curve_fit(Func_Sin, x, y,)

#%%
##
#station = stationF006.initialize()
#print('station finished')
#pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)

vsg = station.vsg
vsg2 = station.vsg2

#Count = StandardParameter(name = 'Count', set_cmd = Counts)

Count = Parameter(name = 'Count', set_cmd = Counts)

time.sleep(0.2)
awg = station.awg
awg2 = station.awg2
#    awg.clock_freq(1e9)
#    awg2.clock_freq(1e9)
    
digitizer = station.digitizer
#    awg.ch3_amp

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

#experiment.saveraw = saveraw
#experiment.readout_time = readout_time
#experiment.threshold = threshold
#experiment.seq_repetition = seq_repetition

init_cfg = {
#        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.004, LP_factor*30*0.5*-0.001]),
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.000, LP_factor*30*0.5*0.000]),
        'step2' : set_step(time = 4e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.005]),
#        'step2' : set_step(time = 5e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.006]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.013, LP_factor*30*0.5*-0.000]),
        'step4' : set_step(time = 5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step5' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.0012, LP_factor*30*0.5*-0.0002]),
        'step6' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.0005]),
#        'step6' : set_step(time = 0.5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.002]),  # from Tom
        }

manip_cfg = {
        'step1' : set_manip(time = 2e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.010,LP_factor*30*0.5*0.020],)
#        'step1' : set_manip(time = 2e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.002,LP_factor*30*0.5*0.016],)
#        'step1' : set_manip(time = 3e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.015,LP_factor*30*0.5*0.060],)
        }


read0_cfg = {
        'step1' : set_step(time = 0.40e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.0005]),
#        'step1' : set_step(time = 0.02e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.00, LP_factor*30*0.5*0.025]),
#        'step1' : set_step(time = 2e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.010, LP_factor*30*0.5*-0.020]),
        }


read_cfg = {
#        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.000]),
        'step1' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
#        'step3' : set_step(time = 0.888e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        }

init2_cfg = {
#        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.005]),
        'step1' : set_step(time = 2.5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.0012, LP_factor*30*0.5*-0.0002]),
        'step3' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.0005]),
        }

manip2_cfg = {
        'step1' : set_manip(time = 1.0e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.010, LP_factor*30*0.5*0.020],)
#        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.002,LP_factor*30*0.5*0.016],)
#        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.015,LP_factor*30*0.5*0.060],)
        }


read2_cfg = {
        'step1' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
#        'step3' : set_step(time = 0.888e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        }

#read3_cfg = {
#        'step1' : set_step(time = 0.262e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step3' : set_step(time = 0.95e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step4' : set_step(time = 0.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*0.00]),
#        }

readBill_cfg = {
        'step1' : set_step(time = 0.263e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 2.45e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step4' : set_step(time = 0.5e-3, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.000]),
        }

sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence_cfg_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

sequence1_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence1_cfg_type = ['init', 'manip01','read', 'init2', 'manip2', 'read2']

sequence11_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence11_cfg_type = ['init', 'manip11','read', 'init2', 'manip2', 'read2']

sequence23_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence23_cfg_type = ['init', 'manip21','read', 'init2', 'manip2', 'read2']

sequence21_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence21_cfg_type = ['init', 'manip21','read', 'init2', 'manip2', 'read2']

sequence31_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence31_cfg_type = ['init', 'manip31','read', 'init2', 'manip2', 'read2']

sequence41_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence41_cfg_type = ['init', 'manip41','read', 'init2', 'manip2', 'read2']

sequence51_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence51_cfg_type = ['init', 'manip51','read', 'init2', 'manip2', 'read2']

sequence61_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence61_cfg_type = ['init', 'manip61','read', 'init2', 'manip2', 'read2']

sequence71_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence71_cfg_type = ['init', 'manip71','read', 'init2', 'manip2', 'read2']

sequence81_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence81_cfg_type = ['init', 'manip81','read', 'init2', 'manip2', 'read2']

sequence91_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence91_cfg_type = ['init', 'manip91','read', 'init2', 'manip2', 'read2']

sequence101_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence101_cfg_type = ['init', 'manip101','read', 'init2', 'manip2', 'read2']


sequence00_cfg = [init_cfg, manip_cfg, read0_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence00_cfg_type = ['init', 'manip00', 'read0', 'read', 'init2', 'manip2', 'read2']

sequence000_cfg = [init_cfg, manip_cfg, read0_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence000_cfg_type = ['init', 'manip000', 'read0', 'read', 'init2', 'manip2', 'read2']

sequence2_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence2_cfg_type = ['init', 'manip','read',]

sequence3_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence3_cfg_type = ['init', 'manip2','read',]

sequence3_cfg2 = [init_cfg, manip2_cfg, read0_cfg, read_cfg,]
sequence3_cfg2_type = ['init', 'manip', 'read0', 'read',]

#sequence_cfg2 = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
#sequence_cfg2_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']


manip3_cfg = deepcopy(manip2_cfg)
manip4_cfg = deepcopy(manip2_cfg)
manip5_cfg = deepcopy(manip2_cfg)
manip6_cfg = deepcopy(manip2_cfg)
manip7_cfg = deepcopy(manip2_cfg)
manip8_cfg = deepcopy(manip2_cfg)
manip9_cfg = deepcopy(manip2_cfg)
manip10_cfg = deepcopy(manip2_cfg)
manip11_cfg = deepcopy(manip2_cfg)
manip12_cfg = deepcopy(manip2_cfg)
manip13_cfg = deepcopy(manip2_cfg)
manip14_cfg = deepcopy(manip2_cfg)
manip15_cfg = deepcopy(manip2_cfg)
manip16_cfg = deepcopy(manip2_cfg)

manip000_cfg = deepcopy(manip2_cfg)


sequenceBill_cfg = [init_cfg, manip000_cfg, 
                    manip2_cfg, readBill_cfg, 
                    manip3_cfg, readBill_cfg, 
                    manip4_cfg, readBill_cfg,
                    manip5_cfg, readBill_cfg,
                    manip6_cfg, readBill_cfg,
                    manip7_cfg, readBill_cfg,
                    manip8_cfg, readBill_cfg,
                    manip9_cfg, readBill_cfg,
                    manip10_cfg, readBill_cfg,
                    manip11_cfg, readBill_cfg,
                    manip12_cfg, readBill_cfg,
                    manip13_cfg, readBill_cfg,
                    manip14_cfg, readBill_cfg,
                    manip15_cfg, readBill_cfg,
                    manip16_cfg, readBill_cfg,]

sequenceBill_cfg_type = ['init', 'manip', 
                         'manip2', 'read', 
                         'manip3', 'read3', 
                         'manip4', 'read4',
                         'manip5', 'read5',
                         'manip6', 'read6',
                         'manip7', 'read7',
                         'manip8', 'read8',
                         'manip9', 'read9',
                         'manip10', 'read10',
                         'manip11', 'read11',
                         'manip12', 'read12',
                         'manip13', 'read13',
                         'manip14', 'read14',
                         'manip15', 'read15',
                         'manip16', 'read16',]

#sequenceBill1_cfg = [init_cfg, manip_cfg, read_cfg, 
#                     init2_cfg, manip2_cfg, read2_cfg, 
#                     read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg]    
#sequenceBill1_cfg_type = ['init', 'manip21','read', 'init2', 'manip22', 'read2', 
#                          'read3', 'read4', 'read5', 'read6', 'read7', 'read8', 'read9', 'read10']

sequenceBill2_cfg = [init_cfg, manip2_cfg, readBill_cfg,]
sequenceBill2_cfg_type = ['init', 'manip2','read',]

T_shift = 30*0.5*-0.002

init_cfg_T1 = {
        'step1' : set_step(time = 1.0e-3, qubits = qubits, voltages = [T_shift+30*0.5*0.004, 30*0.5*-0.001]),
        'step2' : set_step(time = 1.0e-3, qubits = qubits, voltages = [T_shift+30*0.5*-0.004, 30*0.5*0]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_shift+30*0.5*-0.015, 30*0.5*0]),
        'step4' : set_step(time = 2e-3, qubits = qubits, voltages = [T_shift+30*0.5*0, 30*0.5*0]),
        'step5' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_shift+30*0.5*0.002, 30*0.5*0.000]),
#        'step6' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.00, LP_factor*30*0.5*0.01]),
        }

manip_cfg_T1 = {
        'step1' : set_manip(time = 1.2e-6, qubits = qubits, voltages = [T_shift+30*0.5*-0.004, 30*0.5*0.016],)
        }

read0_cfg_T1 = {
#        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_shift+30*0.5*-0.004, 30*0.5*0.016],)
        'step1' : set_step(time = 1e-6, qubits = qubits, voltages = [30*0.5*0.0, 30*0.5*0.0]),
        }

read_cfg_T1 = {
#        'step1' : set_step(time = 1.2e-6, qubits = qubits, voltages = [30*0.5*0.0, 30*0.5*0.0]),
        'step1' : set_step(time = 0.352e-3, qubits = qubits, voltages = [T_shift+30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_shift+30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [T_shift+30*0.5*0, 30*0.5*0]),
        }

sequenceT1_cfg = [init_cfg_T1, manip_cfg_T1, read_cfg_T1,]
sequenceT1_cfg_type = ['init', 'manip','read',]

sequenceT1_cfg2 = [init_cfg_T1, manip_cfg_T1, read0_cfg_T1, read_cfg_T1,]
sequenceT1_cfg2_type = ['init', 'manip', 'read0', 'read',]

#%%

#digitizer, dig = set_digitizer(experiment.digitizer)
#
#vsg.frequency(18.3700e9)
#vsg2.frequency(19.6749e9)
#
#vsg.power(17.75)
#vsg2.power(6.0)


#qubit_1.Pi_pulse_length = 270e-9
#qubit_2.Pi_pulse_length = 100e-9
#
#qubit_1.halfPi_pulse_length = 135e-9
#qubit_2.halfPi_pulse_length = 50e-9

qubit_2.CRot_pulse_length = 340e-9

qubit_2.Pi_pulse_length = 250e-9
qubit_2.halfPi_pulse_length = 125e-9

qubit_1.Pi_pulse_length = 250e-9
qubit_1.halfPi_pulse_length = 125e-9

def reset_experiment():
    experiment.reset()
    
    experiment.qubit_number = 1
    experiment.threshold = 0.020
    experiment.seq_repetition = 100
    
    print('experiment reset')

    return True
#%%        Set Experiment

experiment = Experiment(name = 'RB_experiment', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                        awg = awg, awg2 = awg2, pulsar = pulsar, 
                        vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

calibration = Calibration(name = 'ramsey_calibration', label = 'Ramsey_scan', qubits = [qubit_1, qubit_2], 
                         awg = awg, awg2 = awg2, pulsar = pulsar,
                         vsg = vsg, vsg2 = vsg2, digitizer = digitizer)
'''
experiment = Experiment_v2_2(name = 'RB_experiment2', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                             awg = awg, awg2 = awg2, pulsar = pulsar, 
                             vsg = vsg, vsg2 = vsg2, digitizer = digitizer)
'''

print('experiment initialized')

#self.digitizer_trigger_channel = 'ch5_marker1'
#self.digitier_readout_marker = 'ch6_marker1'

experiment.qubit_number = 2
experiment.readnames = ['Qubit2', 'Qubit1']

#experiment.threshold = 0.0195
experiment.threshold = 0.038

experiment.seq_repetition = 100
experiment.saveraw = False


experiment.readout_time = 0.0022

'''

experiment2.qubit_number = 2
experiment2.readnames = ['Qubit2', 'Qubit1']

#experiment2.threshold = 0.024
#experiment2.threshold = 0.02

experiment2.seq_repetition = 100
experiment2.saveraw = False


experiment2.readout_time = 0.0008

'''

#calibration.qubit_number = 2
#calibration.threshold = 0.013
#calibration.seq_repetition = 100

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 300e-9)
ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, qubit = 'qubit_1', duration_time = 100e-9, waiting_time = 300e-9)

ramsey3 = Ramsey(name = 'Ramsey3', pulsar = pulsar,qubit = 'qubit_1', waiting_time = 300e-9)
ramsey4 = Ramsey(name = 'Ramsey4', pulsar = pulsar,qubit = 'qubit_2', waiting_time = 300e-9)

allxy = AllXY(name = 'AllXY', pulsar = pulsar, qubit = 'qubit_2')



#rb2 = RB(name = 'RB2', pulsar = pulsar, qubit = 'qubit_1')

#rb12 = RB_all(name = 'RB12', pulsar = pulsar)


allxy2 = AllXY(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1')

rabi12 = Rabi_all(name = 'Rabi12', pulsar = pulsar)

rabi123 = Rabi_all(name = 'Rabi123', pulsar = pulsar)

ramsey12 = Ramsey_all(name = 'Ramsey12', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.10, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 260e-9, detune_q1= False)

ramsey_2 = Ramsey_all(name = 'Ramsey_2', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.10, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 260e-9, detune_q1= False)

#ramsey12_2 = Ramsey_all(name = 'Ramsey12_2', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.1, amplitude = 1, 
#                        duration_time = 125e-9, waiting_time = 260e-9,detune_q1= False)

allxy12 = AllXY_all(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1',)

finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)

rabi = Rabi(name = 'Rabi', pulsar = pulsar)

wait = Wait(name = 'Wait', pulsar = pulsar)
#
crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.038*T_factor, amplitude2 = 30*0.5*0.012*LP_factor,
            amplitudepi = 1, frequency_shift = 0.057e9, duration_time = 220e-9)

rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)

rabi3 = Rabi(name = 'Rabi3', pulsar = pulsar, amplitude = 1, qubit = 'qubit_2',)

rabi_off = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1', frequency_shift = -30e6)

crot_freq_bare = 0.06454e9
crot_freq = 0.0516114e9

#phase_1 = 80
#phase_2 = 218
#AMP_C = 30*0.5*-0.0283
#AMP_T = 30*0.5*0.02
#
#crot =  MultiCPhase_Calibrate(name = 'CRot ', pulsar = pulsar, Pi_amplitude = 0, 
#                          detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1', cphase_number = 1)
#%%     T1 Q2
'''
experiment.saveraw = True
experiment.threshold = 0.0218
qubit_2.Pi_pulse_length = 270e-9

experiment.qubit_number = 1
experiment.seq_repetition = 100

#experiment.saveraw = True
experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequenceT1_cfg2, sequenceT1_cfg2_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 31))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.05e9, 0.05e9, 51), element = 'Rabi')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi')

experiment.add_X_parameter('Rabi_Scan', parameter = 'time', sweep_array = sweep_array(1e-6, 25e-3, 60), element = 'read0_step1')


experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5),)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%     MW crosstalk compensate test
'''
vsg.frequency(18.4e9)
vsg2.frequency(19659140000)
#
vsg.power(17)
vsg2.power(5)
'''
#vsg2.power(13)
#vsg.power(16)

'''
#vsg.power()
#vsg.power(3)
#vsg2.power(2.2)
experiment.saveraw = True

experiment.qubit_number = 1
experiment.seq_repetition = 100

#rabi = Rabi(name = 'Rabi', pulsar = pulsar, amplitude = 0.4, duration_time = 250e-9)
#rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, duration_time = 100e-9, qubit = 'qubit_1',)
rabi12 = Rabi_all(name = 'Rabi', pulsar = pulsar, duration_time = 100e-9)

rabi123 = Rabi_all(name = 'Rabi123', pulsar = pulsar, amplitude = 1, third_tone = 1, phase_1 = 169) #duration_time = 80e-9)

#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi2,], sequence3_cfg, sequence3_cfg_type)
experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg, sequence3_cfg_type)

#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi12,], sequence3_cfg, sequence3_cfg_type)

#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 31))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi')
experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'phase_1', sweep_array = sweep_array(0, 360, 61), element = 'Rabi')
#
#experiment.add_measurement('Rabi_Scan_3tones', ['Rabi123'], [rabi123,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Rabi_Scan_3tones', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi123')

#experiment.add_Y_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.5e-6, 51), element = 'Rabi')
#experiment.add_Y_parameter('Rabi_Scan', parameter = 'phase_1', sweep_array = sweep_array(0, 360, 61), element = 'Rabi')
#experiment.add_Y_parameter('Rabi_Scan_3tones', parameter = 'phase_1', sweep_array = sweep_array(0, 360, 61), element = 'Rabi123')
#experiment.add_Y_parameter('Rabi_Scan_3tones', parameter = 'frequency_shift', sweep_array = sweep_array(-0.015e9, 0.015e9, 31), element = 'Rabi123')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3),)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%%
'''
#vsg2.frequency(19659140000)
#vsg2.frequency(19661189747)
vsg2.frequency(19658789747)

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 300e-9, off_resonance = True)

ramsey_0 = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 0, off_resonance = True)

experiment.saveraw = True

experiment.qubit_number = 1
experiment.seq_repetition = 100
experiment.calibration_qubit = 'qubit_2'

experiment.add_measurement('Ramsey_Calibration', ['RC'], [ramsey,], sequence2_cfg, sequence2_cfg_type)
experiment.add_X_parameter('Ramsey_Calibration', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'RC')

experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey_0,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey')

experiment.add_X_parameter('Ramsey_Scan', parameter = 'phase_2', sweep_array = sweep_array(0, 360, 61), element = 'Ramsey')
#experiment.add_Y_parameter('Ramsey_Scan', parameter = vsg.frequency, sweep_array = sweep_array(1, 3, 3),)
experiment.add_Y_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 600e-9, 31), element = 'Ramsey', with_calibration = True)

#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)
#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = False)


experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%% calibrate readout Q2
'''
#experiment.readout_time = 0.0016

experiment.readout_time = 0.003

experiment.saveraw = True

experiment.qubit_number = 1
experiment.seq_repetition = 100

experiment.threshold = 0.040
experiment.calibration_qubit = 'qubit_2'

#qubit_2.Pi_pulse_length = 240e-9

rabi = Rabi(name = 'Rabi', pulsar = pulsar, amplitude = 1)

#rabi2 = Rabi(name = 'Rabi', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)

finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)
#ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 110e-9, waiting_time = 300e-9)


#experiment.add_measurement('Ramsey_Scan', ['Ramsey4'], [ramsey4,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')


experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg, sequence3_cfg_type)

#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi2,], sequence3_cfg, sequence3_cfg_type)

#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg2, sequence3_cfg2_type)


#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi12,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Finding_resonance'], [finding_resonance,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.60e9, 19.72e9, 11))
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.2e9, 20.0e9, 81))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey')
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.power, sweep_array = sweep_array(0, 10, 41))

#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(1e-9, 3e-6, 101), element = 'Rabi')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'phase_2', sweep_array = sweep_array(0, 360, 41), element = 'Ramsey')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.5e-6, 51), element = 'Ramsey')

#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 2), with_calibration = False)
#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 3), with_calibration = True)
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5),)
#experiment.add_Y_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.475e9, 19.490e9, 31))
#experiment.set_sweep(repetition = True, plot_average = False, count = 3)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
'''
% For regular measurement
ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,0,0,:])
for i in range(1,10):
    for j in range(10,20):
        pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,i,j,:])

% For finding resonance scan
ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,0,0,0,0,:])
for i in range(20,30):
    for j in range(10,30):
        pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,i,0,0,j,:])

plot1D(experiment.data_set, measurements = ['Rabi_Scan'], fitfunction = fitcos)


import matplotlib.pyplot as plt
plt.hist(yy.reshape(3100*48,))


'''
#%%
'''
experiment.saveraw = True

experiment.qubit_number = 1
experiment.calibration_qubit = 'qubit_2'

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = qubit_2.halfPi_pulse_length, waiting_time = 300e-9)

ramsey_echo = Ramsey(name = 'Ramsey_Echo', pulsar = pulsar, duration_time = qubit_2.halfPi_pulse_length, waiting_time = 0e-9, echo = True)


experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey')

#experiment.add_measurement('Echo_Scan', ['Ramsey_Echo'], [ramsey_echo,], sequence3_cfg2, sequence3_cfg2_type)
#experiment.add_X_parameter('Echo_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 25e-6, 51), element = 'Ramsey_Echo')

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = False)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

'''
#%%     CRot
'''
experiment.qubit_number = 1
#qubit_2.Pi_pulse_length = 240e-9
#experiment.threshold = 0.022
experiment.saveraw = True
experiment.readout_time = 0.0016


experiment.add_measurement('CRot_Scan', ['CRot'], [crot], sequence3_cfg, sequence3_cfg_type)

#experiment.add_X_parameter('CRot_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.02e9, 0.09e9, 51), element = 'CRot')
experiment.add_X_parameter('CRot_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.07e9, 31), element = 'CRot')
#experiment.add_X_parameter('CRot_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1e-6, 41), element = 'CRot')

#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.385e9, 18.40e9, 16))
#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.power, sweep_array = sweep_array(17.25, 17.35, 51))
experiment.add_Y_parameter('CRot_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5),)
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''
#%%     Q1 rabi and adiabatic
'''
finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar, qubit = 'qubit_1')

experiment.saveraw = True
experiment.qubit_number = 2
experiment.seq_repetition = 100
#experiment.threshold = 0.018
#qubit_2.Pi_pulse_length = 280e-9
qubit_1.Pi_pulse_length = 250e-9
#experiment.readout_time = 0.0008
rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)

#18328513000
experiment.add_measurement('Rabi_Scan', ['Finding_resovsnance','CRot'], [finding_resonance, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)


experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.32e9, 18.37e9, 51))
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.1e9, 18.7e9, 31))

#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.07e9, 31), element = 'CRot')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Rabi2')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.004, 30*0.5*-0.016, 61), element = 'init_step3')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.30e9, 18.50e9, 11))
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('sweep parameter set')

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
'''
#%% Bill Coish exp 2
'''
experiment.qubit_number = 1
#experiment.threshold = 0.021
experiment.seq_repetition = 100

experiment.calibration_qubit = 'qubit_2'

#experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 21), element = 'Ramsey')

experiment.saveraw = True
experiment.add_measurement('Rabi_Scan', ['Rabi3'], [rabi3,], sequenceBill2_cfg, sequenceBill2_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', 
                           sweep_array = [0,qubit_2.halfPi_pulse_length,qubit_2.Pi_pulse_length], element = 'Rabi3')

print('sweep parameter set')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 101, 100))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''

#%%     Bill Coish exp 3
'''
experiment.qubit_number = 15
experiment.seq_repetition = 100

experiment.saveraw = True

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot',], [ramsey12, crot,], sequenceBill1_cfg, sequenceBill1_cfg_type)

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey2')

experiment.add_measurement('Bill_Coish', 
                           ['Rabi2', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot', 'CRot'], 
                           [rabi2, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot, crot], 
                           sequenceBill_cfg, sequenceBill_cfg_type)

experiment.add_X_parameter('Bill_Coish', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi2')
#experiment.add_X_parameter('Bill_Coish', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 11), element = 'Rabi2')

print('sweep parameter set')

#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 101, 100))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)


print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''
#%%     Randomized_Behcnmarking Qubit2
'''
from manipulation_library import RB

rb = RB(name = 'RB', pulsar = pulsar)

#clifford_sets = generate_randomized_clifford_sequence()
experiment.qubit_number = 1
experiment.seq_repetition = 100
experiment.calibration_qubit = 'qubit_2'
#experiment.threshold = 0.018


experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey4,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey')

experiment.add_measurement('RB', ['RB'], [rb,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 27, 28), element = 'RB')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')
experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 29, 30), element = 'RB', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

time.sleep(15)
experiment.run_experiment()
'''

#%%     Randomized_Behcnmarking Qubit1
'''
from manipulation_library import RB, RB_all

rb2 = RB(name = 'RB2', pulsar = pulsar, qubit = 'qubit_1')

experiment.qubit_number = 2
experiment.seq_repetition = 100
#experiment.calibration_qubit = 'qubit_1'
experiment.calibration_qubit = 'all'

#experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey2')
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB_Q1', ['RB2', 'CRot'], [rb2, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB_Q1', parameter = 'clifford_number', sweep_array = sweep_array(0, 27, 28), element = 'RB2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')
experiment.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 9, 10), element = 'RB2', with_calibration = True)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
#time.sleep(15)
#experiment.run_experiment()
#ds1 = experiment.data_set

#time.sleep(60)
'''


#%%        Randomized_Benchmarking Q1 & Q2
'''
from manipulation_library import RB_all
#from RB_library_version2 import RB_all, RB_all_test

rb12 = RB_all(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.15)

#rb12_2 = RB_all(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.1)

#rb12 = RB_all_test(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.0)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('RB', ['RB12', 'CRot'], [rb12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 31, 32), element = 'RB12')
#experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 14, 15), element = 'RB12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')

experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 24, 25), 
#                           element = 'RB12', with_calibration = True, pre_stored = True)

experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 23, 24), element = 'RB12', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
try:
    experiment.generate_sequence()
except:
    experiment.generate_1D_sequence()
#experiment.generate_sequence()
experiment.load_sequence()
print('sequence loaded')


time.sleep(10)

#experiment.run_experiment()


try:
    experiment.run_experiment()
except:
    experiment.close()
'''
#%%        Randomized_Benchmarking Q1 & Q2
'''
#from manipulation_library import RB_all
from RB_library_version2 import RB_all, RB_all_test

rb12 = RB_all(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.10)

#rb12_2 = RB_all(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.1)

#rb12 = RB_all_test(name = 'RB12', pulsar = pulsar, off_resonance_amplitude = 1.0)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

clifford_num = 24

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB', ['RB12', 'CRot'], [rb12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, clifford_num-1, clifford_num), 
                           element = 'RB12')
#experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 4, 5), element = 'RB12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')

experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 24, 25), 
#                           element = 'RB12', with_calibration = True, pre_stored = True)

'''
#2
'''
experiment.add_measurement('Ramsey_Scan2', ['Ramsey12','CRot'], [ramsey12, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11),
                           element = 'Ramsey12', extending_X = False)

experiment.add_measurement('RB2', ['RB12', 'CRot'], [rb12, crot], sequence31_cfg, sequence31_cfg_type)
experiment.add_X_parameter('RB2', parameter = 'clifford_number', sweep_array = sweep_array(0, clifford_num-1, clifford_num), 
                           element = 'RB12', extending_X = False)

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence41_cfg, sequence41_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12',
                           extending_X = False)
'''
#3
'''

experiment.add_measurement('Ramsey_Scan3', ['Ramsey12','CRot'], [ramsey12, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Ramsey_Scan3', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11),
                           element = 'Ramsey12', extending_X = False)

experiment.add_measurement('RB3', ['RB12', 'CRot'], [rb12, crot], sequence31_cfg, sequence31_cfg_type)
experiment.add_X_parameter('RB3', parameter = 'clifford_number', sweep_array = sweep_array(0, clifford_num-1, clifford_num), 
                           element = 'RB12', extending_X = False)

experiment.add_measurement('Rabi_Scan3', ['Rabi12','CRot'], [rabi12, crot], sequence41_cfg, sequence41_cfg_type)
experiment.add_X_parameter('Rabi_Scan3', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), 
                           element = 'Rabi12', extending_X = False)
'''
#4
'''
experiment.add_measurement('Ramsey_Scan4', ['Ramsey12','CRot'], [ramsey12, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Ramsey_Scan4', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11),
                           element = 'Ramsey12', extending_X = False)

experiment.add_measurement('RB4', ['RB12', 'CRot'], [rb12, crot], sequence31_cfg, sequence31_cfg_type)
experiment.add_X_parameter('RB4', parameter = 'clifford_number', sweep_array = sweep_array(0, clifford_num-1, clifford_num), 
                           element = 'RB12', extending_X = False)

experiment.add_measurement('Rabi_Scan4', ['Rabi12','CRot'], [rabi12, crot], sequence41_cfg, sequence41_cfg_type)
experiment.add_X_parameter('Rabi_Scan4', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12',
                           extending_X = False)
'''
#5
'''
experiment.add_measurement('Ramsey_Scan5', ['Ramsey12','CRot'], [ramsey12, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Ramsey_Scan5', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11),
                           element = 'Ramsey12', extending_X = False)

experiment.add_measurement('RB5', ['RB12', 'CRot'], [rb12, crot], sequence31_cfg, sequence31_cfg_type)
experiment.add_X_parameter('RB5', parameter = 'clifford_number', sweep_array = sweep_array(0, clifford_num-1, clifford_num), 
                           element = 'RB12', extending_X = False)

experiment.add_measurement('Rabi_Scan5', ['Rabi12','CRot'], [rabi12, crot], sequence41_cfg, sequence41_cfg_type)
experiment.add_X_parameter('Rabi_Scan5', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12',
                           extending_X = False)
'''
'''
#sweep = sweep_array(0, 639, 640)
seq_num = 128

experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, seq_num-1, seq_num), element = 'RB12', with_calibration = True)
experiment.add_Y_parameter('RB2', parameter = 'sequence_number', sweep_array = sweep_array(seq_num, 2*seq_num-1, seq_num), element = 'RB12', with_calibration = True)
experiment.add_Y_parameter('RB3', parameter = 'sequence_number', sweep_array = sweep_array(2*seq_num, 3*seq_num-1, seq_num), element = 'RB12', with_calibration = True)
experiment.add_Y_parameter('RB4', parameter = 'sequence_number', sweep_array = sweep_array(3*seq_num, 4*seq_num-1, seq_num), element = 'RB12', with_calibration = True)
experiment.add_Y_parameter('RB5', parameter = 'sequence_number', sweep_array = sweep_array(4*seq_num, 5*seq_num-1, seq_num), element = 'RB12', with_calibration = True)


#experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 127, 128), element = 'RB12', with_calibration = True)
#experiment.add_Y_parameter('RB2', parameter = 'sequence_number', sweep_array = sweep_array(128, 255, 128), element = 'RB12', with_calibration = True)
#experiment.add_Y_parameter('RB3', parameter = 'sequence_number', sweep_array = sweep_array(256, 383, 128), element = 'RB12', with_calibration = True)
#experiment.add_Y_parameter('RB4', parameter = 'sequence_number', sweep_array = sweep_array(384, 511, 128), element = 'RB12', with_calibration = True)
#experiment.add_Y_parameter('RB5', parameter = 'sequence_number', sweep_array = sweep_array(512, 639, 128), element = 'RB12', with_calibration = True)


experiment.set_sweep(repetition = False, plot_average = False, count = 1, make_new_element = False)
print('loading sequence')


#experiment.generate_sequence()
'''
'''
try:
    experiment.generate_sequence()
except:
    experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
'''
time.sleep(15)

try:
    experiment.run_experiment()
except:
    experiment.close()    
'''
#ds2 = experiment2.data_set
'''
plot2D(ds, measurements = ['Ramsey_Scan', 'RB', 'Rabi_Scan'])
'''
#%%        Randomized_Benchmarking Q1 & Q2 interleaved with CZ
'''
from RB_library_version2 import RB_all, RB_all_test
#from manipulation_library import RBinterleavedCZ

phase_1 = 191.19
phase_2 = 98.87

AMP_C = 30*0.5*-0.0255
AMP_T = 30*0.5*0.012

duration = 80e-9

#RBI = RBinterleavedCZ(name = 'RBI', pulsar = pulsar, amplitude_control = AMP_C, amplitude_target = AMP_T, phase_1 = phase_1, phase_2 = phase_2 , 
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.15, control_qubit = 'qubit_1', control = 0)

RBI = RB_all(name = 'RBI', pulsar = pulsar, amplitude_control = AMP_C, amplitude_target = AMP_T, phase_1 = phase_1, phase_2 = phase_2 , 
             detuning_time = duration, off_resonance_amplitude = 1.10,)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


#experiment.add_measurement('RB', ['RBI', 'CRot'], [RBI, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_measurement('RB', ['RBI', 'CRot'], [RBI, crot], sequence00_cfg, sequence00_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 5, 6), element = 'RBI')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')


#experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence000_cfg, sequence000_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)
experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 3, 4), element = 'RBI', with_calibration = True)
#experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 127, 128), element = 'RBI', with_calibration = True)


experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
#experiment.load_sequence()
#print('sequence loaded')
'''
'''
time.sleep(15)

try:
    experiment.run_experiment()
except:
    experiment.close()
'''

'''
ds = experiment.data_set
pt = MatPlot()

pt.add(x = ds.sweep_data[0,0,11:-2],y=ds.probability_data[0,0,11:-2])


'''

'''
for i in range(115,128):
    for j in range(len(ds.probability_data[i][0]))
        ds.probability_data[i][0][j] = 0
'''
'''
for i in range(115,128):
    ds.sweep_data[i][0] = ds.sweep_data[0][0]
'''
#%%     Randomized_Benchmarking Marcus
'''
from manipulation_library import RB_Marcus

phase_1 = 141.1
phase_2 = 308

AMP_C = 30*0.5*-0.0269
AMP_T = 30*0.5*0.02
duration = 80e-9


rb_marcus = RB_Marcus(name = 'RB_M', pulsar = pulsar, detuning_time = duration, phase_1 = phase_1, phase_2 = phase_2, 
                      Pi_amplitude = 0, amplitude_control = AMP_C, amplitude_target = AMP_T)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_measurement('RB_Q1', ['RB_M', 'CRot'], [rb_marcus, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_measurement('RB_Q1', ['RB_M', 'CRot'], [rb_marcus, crot], sequence00_cfg, sequence00_cfg_type)
experiment.add_X_parameter('RB_Q1', parameter = 'clifford_number', sweep_array = sweep_array(0, 17, 18), element = 'RB_M')

#experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence000_cfg, sequence000_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 23, 24), element = 'RB_M', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

time.sleep(15)
try:
    experiment.run_experiment()
except:
    experiment.close()
'''

#%%     full C2 RB

'''
from RB_library import RB_Martinis

rb_martinis = RB_Martinis(name = 'RB_M', pulsar = pulsar, detuning_time = 80e-9, phase_1 = 178, phase_2 = 302, Pi_amplitude = 1,)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB_Q1', ['RB_M', 'CRot'], [rb_martinis, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB_Q1', parameter = 'clifford_number', sweep_array = sweep_array(0, 11, 12), element = 'RB_M')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 39, 40), element = 'RB_M', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(15)
experiment.run_experiment()
'''

#%%     charge noise bob joynt
'''
from manipulation_library import Charge_Noise_Bob2, Charge_Noise_Bob3, Grover

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.readout_time = 0.0026
experiment.threshold = 0.032

phase_1 = 219 #for target = 'qubit_1'
phase_2 = 115 #for target = 'qubit_2'

AMP_C = 30*0.5*-0.02505
AMP_T = 30*0.5*0.011
duration = 80e-9

off_resonance_amplitude = 1.1

# 00-11 subspace
charge_noise_bob = Charge_Noise_Bob3(name = 'Charge_Noise', pulsar = pulsar, detuning_time = duration, phase_1 = phase_1, phase_2 = phase_2,
                                     amplitude_control = AMP_C, amplitude_target = AMP_T, decoupled_cphase = False,
                                     off_resonance_amplitude = off_resonance_amplitude, add_dephase = False, DFS = 0)

# 01-10 subspace
charge_noise_bob2 = Charge_Noise_Bob3(name = 'Charge_Noise_2', pulsar = pulsar, detuning_time = duration, phase_1 = phase_1, phase_2 = phase_2,
                                      amplitude_control = AMP_C, amplitude_target = AMP_T, decoupled_cphase = False,
                                      off_resonance_amplitude = off_resonance_amplitude, add_dephase = False, DFS = 1)


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [charge_noise_bob, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 36), element = 'Charge_Noise')
#
#experiment.add_measurement('CN2', ['Charge_Noise_2', 'CRot'], [charge_noise_bob2, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('CN2', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 61), element = 'Charge_Noise_2')



#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)
experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 3), with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%% charge noise bob joynt with added noise....
'''
from manipulation_library import Charge_Noise_Bob2, Charge_Noise_Bob3, Grover, Charge_Noise_Bob_withaddednoise

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True


phase_1 = 2.5
phase_2 = 301.5
AMP_C = 30*0.5*-0.03025
AMP_T = 30*0.5*0.075


charge_noise_bob = Charge_Noise_Bob_withaddednoise(name = 'Charge_Noise', pulsar = pulsar, detuning_time = 80e-9, 
                                    phase_1 = phase_1, phase_2 = phase_2 , off_resonance_amplitude = 1.05,
                                    add_dephase = False, decoupled_qubit = 'qubit_1',
                                    amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 0,
                                    decoupled_cphase = False, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.3e6 , sigma4=0.0e6)

charge_noise_bob2 = Charge_Noise_Bob_withaddednoise(name = 'Charge_Noise_2', pulsar = pulsar, detuning_time = 80e-9, 
                                     phase_1 = phase_1, phase_2 = phase_2 , off_resonance_amplitude = 1.05,
                                     add_dephase = False, decoupled_qubit = 'qubit_1',
                                     amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 1,
                                     decoupled_cphase = False, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.3e6 , sigma4=0.0e6)


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')
##
#experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [charge_noise_bob, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 61), element = 'Charge_Noise')

experiment.add_measurement('CN2', ['Charge_Noise_2', 'CRot'], [charge_noise_bob2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('CN2', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 61), element = 'Charge_Noise_2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#
#experiment.add_measurement('Rabi_Scan3', ['Rabi123','CRot'], [rabi123, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('Rabi_Scan3', parameter = 'duration_time', sweep_array = sweep_array(0, 30e-6, 2), element = 'Rabi123')

#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)
experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 20), with_calibration = True)


#experiment.add_Y_parameter('CN', parameter = 'dummy', sweep_array = sweep_array(0, 10, 10), element = 'Charge_Noise', with_calibration = True)
#experiment.add_Y_parameter('CN2', parameter = 'dummy', sweep_array = sweep_array(0, 10, 10), element = 'Charge_Noise_2', with_calibration = True)
#
#experiment.add_Y_parameter('CN', parameter = 'dummy', sweep_array = sweep_array(0, 10, 5), element = 'Charge_Noise_2', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%     CPhase
'''
experiment.readout_time = 0.0015
experiment.threshold = 0.027
experiment.seq_repetition = 50

# for normal CPhase

phase_1 = 258 #for target = 'qubit_1'
phase_2 = 155 #for target = 'qubit_2'

AMP_C = 30*0.5*-0.0247
AMP_T = 30*0.5*0.011
duration = 80e-9

# for opposite CPhase
#AMP_C = 30*0.5*0.041
#AMP_T = 30*0.5*-0.05
#duration = 80e-9

#phase1_r = 7

control = 'qubit_1'

#cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')
#
#cphase2 = CPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')


cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, 
                          detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T,
                          detuning_time = duration, phase = 0, off_resonance_amplitude = 1.10, control_qubit = control)

cphase2 = CPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, 
                           detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T,
                           detuning_time = duration, phase = 0, off_resonance_amplitude = 1.10, control_qubit = control)


experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('CPhase_Calibration', ['CPhase','CRot'], [cphase, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase')
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(100e-9, 3e-6, 30), element = 'CPhase')
experiment.add_X_parameter('CPhase_Calibration', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase')

experiment.add_measurement('CPhase_Calibration2', ['CPhase2','CRot'], [cphase2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('CPhase_Calibration2', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase2')
#experiment.add_X_parameter('CPhase_Calibration2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase2')
#experiment.add_X_parameter('CPhase_Calibration2', parameter = 'detuning_time', sweep_array = sweep_array(100e-9, 3e-6, 30), element = 'CPhase2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('CPhase_Calibration', parameter = Count, sweep_array = sweep_array(1, 5, 10), with_calibration = True)
#experiment.add_Y_parameter('CPhase_Calibration', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
#time.sleep(0.5)
'''
'''
plot1D(experiment.data_set, measurements = ['CPhase_Calibration','CPhase_Calibration2'], sameaxis = True, fitfunction = fitcos)
plot1D(experiment.data_set, measurements = ['CPhase_Calibration','CPhase_Calibration2'], sameaxis = True)
pt = MatPlot()
pt.add(x = experiment.data_set.sweep_data[0,1,11:32],y=experiment.data_set.probability_data[:,1,11:32].mean(axis=0))
pt.add(x = experiment.data_set.sweep_data[0,1,11:32],y=experiment.data_set.probability_data[:,1,32:53].mean(axis=0))
'''
#%% MultiCphase
'''
phase_1 = 84
phase_2 = 72
#AMP_C = 30*0.5*-0.02549
#AMP_T = 30*0.5*0.04
AMP_C = 30*0.5*-0.0365
AMP_T = 30*0.5*0.02
phase_1 = 0
phase_2 = 0

#cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')
#
#cphase2 = CPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')


cphase =  MultiCPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, 
                          detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T,
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude =0, control_qubit = 'qubit_1', cphase_number = 1,phase_1 = phase_1, phase_2 = phase_2)

cphase2 =  MultiCPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, 
                           detuning_amplitude = AMP_C, detuning_amplitude2 =AMP_T,
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 0, control_qubit = 'qubit_1', cphase_number =1,phase_1 = phase_1, phase_2 = phase_2)

cphase3 =  MultiCPhase_Calibrate(name = 'CPhase3', pulsar = pulsar, Pi_amplitude = 1, 
                           detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T,
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 0, control_qubit = 'qubit_1', cphase_number =1,phase_1 = phase_1, phase_2 = phase_2)


experiment.calibration_qubit = 'all'

experiment.calibration_qubit == 'qubit_2'

experiment.add_measurement('Ramsey_Scan', ['Ramsey4','CRot'], [ramsey4, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('CPhase_Calibration', ['CPhase','CRot'], [cphase, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase')
experiment.add_X_parameter('CPhase_Calibration', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase')

experiment.add_measurement('CPhase_Calibration2', ['CPhase2','CRot'], [cphase2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('CPhase_Calibration2', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase2')

##
#experiment.add_measurement('CPhase_Calibration3', ['CPhase3','CRot'], [cphase3, crot], sequence31_cfg, sequence31_cfg_type)
#experiment.add_X_parameter('CPhase_Calibration3', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase3')

#experiment.add_X_parameter('CPhase_Calibration2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('CPhase_Calibration', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''
#%%     measure T-
#
'''
MeasureTminus = MeasureTminus(name = 'MeasureTminus', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.025, detuning_amplitude2 = 30*0.5*0.115,detuning = 0, 
                          detuning_time = 1e-6, phase = 0, off_resonance_amplitude = 1.0)
#measureTminus = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.023, 
#                          detuning_time = 71.5e-9, phase = 0, off_resonance_amplitude = 1.2)

experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('MeasureTminus', ['MeasureTminus','CRot'], [MeasureTminus, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('MeasureTminus', parameter = 'detuning', sweep_array = sweep_array(30*0.5*-0.010, 30*0.5*0.00, 30), element = 'MeasureTminus')
#
#experiment.add_measurement('CPhase_Calibration', ['CPhase','CRot'], [cphase, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 50), element = 'CPhase')


print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''
#%% test
'''
experiment.add_measurement('Ramsey_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'frequency_shift', sweep_array = sweep_array(-0.0001e9, 0.0001e9, 2), element = 'Rabi12')


print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 2)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''
#%%        DCZ
'''
from DD_library import DCZ_ledge, DCZ_ramp

experiment.saveraw = True
#experiment.threshold = 0.015

phase1 = 90
phase2 = 90

ramp_time = 100e-9
ledge_C = 30*0.5*-0.00
ledge_T = 30*0.5*0.015

AMP_C = 30*0.5*-0.0266
AMP_T = 30*0.5*0.0118

#AMP_C = 30*0.5*0.0308
#AMP_T = 30*0.5*-0.03

#AMP_C = 30*0.5*-0.01
#AMP_T = 30*0.5*0.10

#AMP_C = 30*0.5*0.0
#AMP_T = 30*0.5*0.0

AMP_C_r = 30*0.5*0.035      # for opposite CPhase
AMP_T_r = 30*0.5*-0.020

#dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T, 
#          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 0.85, control_qubit = 'qubit_1')
#
#dcz2 = DCZ(name = 'DCZ2', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude =  AMP_C, detuning_amplitude2 = AMP_T, 
#           detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 0.85, control_qubit = 'qubit_1')

#dcz = DCZ_ledge(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T, 
#dcz = DCZ_ramp(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T, ramp_time = ramp_time,
dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T, 
          detuning_amplitude3 = AMP_C_r, detuning_amplitude4 = AMP_T_r,
          ledge_amplitude1 = ledge_C, ledge_amplitude2 = ledge_T,
          detuning_time = 2.5e-6, phase = 0, off_resonance_amplitude = 1.10, control_qubit = 'qubit_2')

experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_measurement('DCZ', ['DCZ','CRot'], [dcz, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_measurement('DCZ', ['DCZ','CRot'], [dcz, crot], sequence00_cfg, sequence00_cfg_type)
experiment.add_X_parameter('DCZ', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 600e-9, 30), element = 'DCZ')
#experiment.add_X_parameter('DCZ', parameter = 'detuning_time', sweep_array = sweep_array(100e-9, 3.0e-6, 30), element = 'DCZ')
#experiment.add_X_parameter('DCZ', parameter = 'detuning_amplitude', sweep_array = sweep_array(30*0.5*-0.0265, 30*0.5*-0.0285, 21), element = 'DCZ')
#experiment.add_X_parameter('DCZ', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ')

#experiment.add_measurement('DCZ2', ['DCZ2','CRot'], [dcz2, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('DCZ2', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ2')
#experiment.add_X_parameter('DCZ2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 200e-9, 40), element = 'DCZ2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
#experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence000_cfg, sequence000_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_Y_parameter('DCZ', parameter = 'detuning_amplitude2', sweep_array = sweep_array(30*0.5*-0.02, 30*0.5*0.08, 11), element = 'DCZ',  with_calibration = True)
experiment.add_Y_parameter('DCZ', parameter = Count, sweep_array = sweep_array(1, 5, 3), with_calibration = True)
#experiment.add_Y_parameter('DCZ', parameter = 'detuning_amplitude2', sweep_array = sweep_array(30*0.5*0.01, 30*0.5*0.05, 41), element = 'DCZ', with_calibration = True)


print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''
'''
plot1D(experiment.data_set, measurements = ['DCZ','DCZ2'], sameaxis = True)
pt = MatPlot()
pt.add(x = experiment.data_set.sweep_data[0,1,11:42],y=experiment.data_set.probability_data[:,1,11:42].mean(axis=0))
pt.add(x = experiment.data_set.sweep_data[0,1,11:42],y=experiment.data_set.probability_data[:,1,42:73].mean(axis=0))
'''
#%% DCZ vary the phase
'''
#dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.027, detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2)
dcz2 = DCZ(name = 'DCZ2', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.027, detuning_amplitude2 = 30*0.5*0.020, detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2)
dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.027, detuning_amplitude2 = 30*0.5*0.020, detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2)

experiment.calibration_qubit = 'all'


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')



experiment.add_measurement('DCZ', ['DCZ','CRot'], [dcz, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('DCZ', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 1000e-9, 30), element = 'DCZ')
experiment.add_X_parameter('DCZ', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ')

experiment.add_measurement('DCZ2', ['DCZ2','CRot'], [dcz2, crot], sequence31_cfg, sequence31_cfg_type)
experiment.add_X_parameter('DCZ2', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ2')


experiment.add_measurement('Ramsey_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'frequency_shift', sweep_array = sweep_array(-0.0001e9, 0.0001e9, 2), element = 'Rabi12')

experiment.add_measurement('Wait', ['Wait','CRot'], [wait, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Wait', parameter = 'frequency_shift', sweep_array = sweep_array(-0.0001e9, 0.0001e9, 2), element = 'Wait')


#experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 2), element = 'AllXY12')
#

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''

#%%        Hahn echo
'''
from manipulation_library import Hahn

target_qubit = 'qubit_1'

Hahn = Hahn(name = 'Hahn', pulsar = pulsar, Pi_amplitude = 0, wait_time = 70e-9, off_resonance_amplitude = 1.15, target_qubit = target_qubit)

experiment.calibration_qubit = target_qubit

if target_qubit == 'qubit_1':
    experiment.add_measurement('Ramsey_Scan', ['Ramsey3','CRot'], [ramsey3, crot], sequence_cfg, sequence_cfg_type)
    experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey3')
elif target_qubit == 'qubit_2':
    experiment.add_measurement('Ramsey_Scan', ['Ramsey4','CRot'], [ramsey4, crot], sequence_cfg, sequence_cfg_type)
    experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')

experiment.add_measurement('Hahn', ['Hahn','CRot'], [Hahn, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Hahn', parameter = 'wait_time', sweep_array = sweep_array(1e-9, 60e-6, 30), element = 'Hahn')
#experiment.add_X_parameter('Hahn', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'Hahn1')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('Hahn', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%% Ramsey 00 and 11 basis
'''
ramsey_00_11_basis = Ramsey_00_11_basis(name = 'Ramsey_00_11_basis', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.027, detuning_amplitude2 = 30*0.5*0.020, detuning_time = 80e-9, wait_time = 100e-9, phase1 = 90, phase2 = -90, off_resonance_amplitude = 1.2)

experiment.calibration_qubit = 'all'


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Ramsey_00_11_basis', ['Ramsey_00_11_basis','CRot'], [ramsey_00_11_basis, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Ramsey_00_11_basis', parameter = 'wait_time', sweep_array = sweep_array(0, 1000e-9, 21), element = 'Ramsey_00_11_basis')

experiment.add_measurement('Ramsey_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'frequency_shift', sweep_array = sweep_array(-0.0001e9, 0.0001e9, 2), element = 'Rabi12')

experiment.add_measurement('Wait', ['Wait','CRot'], [wait, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Wait', parameter = 'frequency_shift', sweep_array = sweep_array(-0.0001e9, 0.0001e9, 2), element = 'Wait')


#experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 2), element = 'AllXY12')
#

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''
#%%     charge noise
"""
charge_noise = Charge_Noise(name = 'Charge_Noise', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.024, detuning_time = 50e-9, phase = 0)
experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('Charge_Noise', ['Charge_Noise','CRot'], [charge_noise, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Charge_Noise', parameter = 'detuning_time', sweep_array = sweep_array(0, 60e-9, 31), element = 'Charge_Noise')
#experiment.add_Y_parameter('Charge_Noise', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.010, 30*0.5*-0.004, 7), element = 'Charge_Noise')

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
"""

#%%        Two Qubit Gate
'''
#experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
##
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#experiment.add_Y_parameter('Ramsey_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 51))

experiment.calibration_qubit = 'all'
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 21))
#

experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY12')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))


print('sweep parameter set')
experiment.set_sweep(repetition = True, plot_average = False, count = 5)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''


#%%     Ramsey and AllXY Q1
'''
#experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.388e9, 18.398e9, 51))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.06e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.02e9, 0.02e9, 41), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

experiment.calibration_qubit = 'qubit_1'
#
experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9,11), element = 'Ramsey2')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#

experiment.add_measurement('AllXY_calibration', ['AllXY2', 'CRot'], [allxy2, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY2')
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 21), element = 'AllXY2')

#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = True)

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = False, with_calibration = False, plot_average = False, count = 1)
#
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''
#%%        AllXY calibration   qubit_2
'''
experiment.qubit_number = 1
#experiment.threshold = 0.011
experiment.seq_repetition = 100

experiment.calibration_qubit = 'qubit_2'

experiment.add_measurement('Ramsey_Scan', ['Ramsey4'], [ramsey4,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')

experiment.add_measurement('AllXY_calibration', ['AllXY'], [allxy,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY')
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 21), element = 'AllXY')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 6), with_calibration = True)
#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 15, 15), with_calibration = True)

print('sweep parameter set')

#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 15)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
#time.sleep(0.5)
#awg.ch3_amp(2.4)
#awg.ch4_amp(2.4)
#
#
#
#awg2.set_sqel_trigger_wait(element_no = 1, state = 0)
#for i in range(21):
#    awg2.set_sqel_trigger_wait(element_no = 2+10*i, state = 1)

'''

#%% Calibration of CROT readout step 1
'''
experiment.qubit_number = 1

experiment.add_measurement('2D_Rabi_Scan', ['CRot'], [crot], sequence2_cfg, sequence2_cfg_type)
experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.070e9, 0.090e9,21), element = 'CRot')
#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.8e-6, 31), element = 'CRot')

print('sweep parameter set')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 3, 3))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%% Calibration of CROT readout step 2
'''
experiment.qubit_number = 2
experiment.seq_repetition = 100

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.060e9, 0.080e9, 21), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'CRot')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 5, 3), with_calibration = False)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%% Measure Q1 frequency 
'''
experiment.qubit_number = 2
experiment.saveraw = True
#experiment.threshold = 0.017

experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.25e9, 18.4e9, 76))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.03e9, 0.03e9, 41), element = 'Rabi2')
experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.07e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.065e9, 31), element = 'CRot')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.5e-6, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.002, 30*0.5*-0.01, 41), element = 'init_step3')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.30e9, 18.50e9, 11))
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('sweep parameter set')
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
'''

#%% Measure Q1 ramsey frequency
'''
experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9,21), element = 'Ramsey2')

experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.6e-6, 41), element = 'Ramsey2')


experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 10))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))


print('sweep parameter set')

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
'''
#%% Measure Q1 as a function of detuing
'''
vsg.power(10.3)
rabi2_det = Rabi_detuning(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)
experiment.qubit_number = 2
experiment.seq_repetition = 200
experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2_det, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(30*0.5*-0.025, 30*0.5*0.000, 26), element = 'Rabi2')

experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.46e9, 61))
#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 3, 4))


print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%% Measure Q2 as a function of detuing
'''
rabi2_det = Rabi_detuning(name = 'Rabi2', pulsar = pulsar, amplitude = 30*0.5*-0.025, amplitude2 = 30*0.5*0.012, qubit = 'qubit_2',)

experiment.seq_repetition = 100
experiment.qubit_number = 2
experiment.readout_time = 0.0026
experiment.threshold = 0.035
experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2_det, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(30*0.5*-0.025, 30*0.5*-0.045, 21), element = 'Rabi2')
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(30*0.5*-0.030, 30*0.5*-0.045, 31), element = 'Rabi2')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.735e9, 19.785e9, 31))
experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.72e9, 19.76e9, 41))
#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 3, 4))

#vsg2.frequency(19.6724e9)
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%% Measure Q2 frequency
'''
experiment.seq_repetition = 100
experiment.saveraw = True
experiment.add_measurement('Rabi_Scan', ['Rabi3','CRot'], [rabi3, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi3')
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.5e9, 19.65e9, 76))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi3')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 21), element = 'Rabi3')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.565e9, 19.575e9, 21))

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 11, 3))

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
'''
#%% Measure Q2 ramsey
'''
experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.add_measurement('Ramsey_Scan', ['Ramsey4','CRot'], [ramsey4, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.004e9, 0.004e9,21), element = 'Ramsey4')

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3), with_calibration = False)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))


print('sweep parameter set')

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
'''
#%% Simultaneous pulse measure Q1 and Q2 frequency Rabi

#from DD_library import Rabi_all_with_detuning
#rabi12 = Rabi_all_with_detuning(name = 'Rabi12', pulsar = pulsar, off_resonance_amplitude = 1.15,
#                                max_duration = 3e-6)

experiment.calibration_qubit = 'all'
#experiment.calibration_qubit = 'qubit_2'

experiment.saveraw = True
experiment.readout_time = 0.003
experiment.threshold = 0.040
experiment.seq_repetition = 100

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence3_cfg, sequence3_cfg_type)   # only read Q2
experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.010e9, 0.010e9, 31), element = 'Rabi12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1.0e-6, 51), element = 'Rabi12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'time', sweep_array = sweep_array(2e-3, 8e-3, 31), element = 'init_step4')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.006, 30*0.5*-0.012, 61), element = 'init_step3')
#
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.070e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.67e9, 19.68e9, 20))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'T_amplitude', sweep_array = sweep_array(0, 30*0.5*0.008, 31), element = 'Rabi12')

#experiment.add_measurement('Ramsey_Scan2', ['Ramsey12_2','CRot'], [ramsey12, crot], sequence11_cfg, sequence11_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan2', parameter = 'waiting_time', sweep_array = sweep_array(50e-9, 2e-6, 40), element = 'Ramsey12_2')

#experiment.add_measurement('Rabi_Scan2', ['Rabi12_2','CRot'], [rabi12, crot], sequence23_cfg, sequence23_cfg_type)
#experiment.add_X_parameter('Rabi_Scan2', parameter = 'duration_time', sweep_array = sweep_array(0, 3.0e-6, 151), element = 'Rabi12_2')
#
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = False)
#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.642e9, 19.672e9, 31))

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

#%% Simultaneous pulse measure Q1 & Q2 frequency ramsey
'''
experiment.readout_time = 0.0025
experiment.threshold = 0.025
experiment.seq_repetition = 50
experiment.saveraw = True
#experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Cali', ['Ramsey_2','CRot'], [ramsey_2, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Ramsey_Cali', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey_2')

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey12')
experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1.5e-6, 50), element = 'Ramsey12')
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'phase', sweep_array = sweep_array(0, 360, 31), element = 'Ramsey12')
#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 20), with_calibration = True)
#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%
'''
from DD_library import Hahn_all

hahn_all = Hahn_all(name = 'hahn_all', pulsar = pulsar, off_resonance_amplitude = 1.15)

experiment.calibration_qubit = 'all'
#experiment.calibration_qubit = 'qubit_2'

experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Echo_Scan', ['hahn_all','CRot'], [hahn_all, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Echo_Scan', parameter = 'waiting_time', sweep_array = sweep_array(5e-9, 15e-6, 51), element = 'hahn_all')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi12')

#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 3), with_calibration = False)
experiment.add_Y_parameter('Echo_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = True)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.642e9, 19.672e9, 31))

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%Simultaneous pulse measure Q2 frequency ramsey with noise
'''
experiment.seq_repetition = 50

experiment.readout_time = 0.0025
experiment.threshold = 0.03

amplitude = 1.1;
off_resonance_amplitude = 1.1;

experiment.calibration_qubit = 'all'
ramsey_withnoise = Ramsey_withnoise(name = 'Ramsey_withnoise', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.2, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 300e-9,detune_q1= True, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.2e6 , sigma4=0.0e6)



experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')



experiment.add_measurement('Ramsey_Scan2', ['ramsey_withnoise','CRot'], [ramsey_withnoise, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey12')
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.6e-6, 81), element = 'ramsey_withnoise')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))
#experiment.add_Y_parameter('Ramsey_Scan', parameter = 'dummy', sweep_array = sweep_array(0, 10, 10), element = 'ramsey_withnoise', with_calibration = False)
experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 15), with_calibration = True)


print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%%
'''
experiment.calibration_qubit = 'all'

ramsey375 = Ramsey_all(name = 'Ramsey375', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 0, amplitude = 1, 
                      duration_time = 375e-9, waiting_time = 300e-9, detune_q1 = True)

ramsey125 = Ramsey_all(name = 'Ramsey125', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 0, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 300e-9, detune_q1 = True)


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey12')


experiment.add_measurement('Ramsey125_Scan', ['Ramsey125','CRot'], [ramsey125, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Ramsey125_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.2e-6, 41), element = 'Ramsey125')

experiment.add_measurement('Ramsey375_Scan', ['Ramsey375','CRot'], [ramsey375, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('Ramsey375_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.2e-6, 41), element = 'Ramsey375')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 5)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%% Simultaneous allXY with calibration
'''
experiment.readout_time = 0.0015
experiment.threshold = 0.027
experiment.seq_repetition = 50

experiment.calibration_qubit = 'all'
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 2e-6, 21), element = 'Ramsey12')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 21))
#

experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY12')
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 21), element = 'AllXY12')

#experiment.add_measurement('Ramsey_Scan2', ['Ramsey12','CRot'], [ramsey12, crot], sequence31_cfg, sequence31_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan2', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12',
#                           extending_X = False)

#experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence31_cfg, sequence31_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi12',
#                           extending_X = False)

#experiment.add_measurement('AllXY_calibration2', ['AllXY12', 'CRot'], [allxy12, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('AllXY_calibration2', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY12',
#                           extending_X = False)

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 20), with_calibration = True)
#experiment.add_Y_parameter('Ramsey_Scan', parameter = 'Count', sweep_array = sweep_array(1, 10, 6), element = 'Ramsey12', 
#                           with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count =5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''
#%%     Bootstrap Tomography
'''
from DD_library import Bootstrap


bootstrap = Bootstrap(name = 'Bootstrap', pulsar = pulsar, qubit = 'qubit_1', phase_error = 0, error_gate = 'None')


experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Bootstrap_Tomography', ['Bootstrap', 'CRot'], [bootstrap, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Bootstrap_Tomography', parameter = 'gate', sweep_array = sweep_array(1, 12, 12), element = 'Bootstrap')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 20, 20), with_calibration = True)
experiment.add_Y_parameter('Bootstrap_Tomography', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 21), 
                           element = 'Bootstrap', with_calibration = True)
#experiment.add_Y_parameter('Bootstrap_Tomography', parameter = 'phase_error', sweep_array = sweep_array(-30, 30, 21), 
#                           element = 'Bootstrap', with_calibration = True)


print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''

#%% Sync pulses step 1
'''
sychpulses1 = Sychpulses1(name = 'Sychpulses1', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.031, detuning_time = 72e-9, phase = 0, off_resonance_amplitude = 1.3)
experiment.calibration_qubit = 'all'

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('Sychpulses1', ['Sychpulses1','CRot'], [sychpulses1, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Sychpulses1', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 50e-9, 20), element = 'Sychpulses1')

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = False, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''

#%% Sync pulses step 2
'''
sychpulses2 = Sychpulses2(name = 'Sychpulses2', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.031, detuning_time = 72e-9, phase = 0, off_resonance_amplitude = 1.3)
experiment.calibration_qubit = 'all'

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('Sychpulses2', ['Sychpulses2','CRot'], [sychpulses2, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Sychpulses2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 22e-9, 11), element = 'Sychpulses2')

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = False, plot_average = False, count = 20)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
'''


#%%        T1
'''
T_shift = 30*0.5*-0.002
LP_shift = 0

init_t1_cfg = {
        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.004, LP_shift + 30*0.5*-0.001]),
        'step2' : set_step(time = 4e-3, qubits = qubits, voltages = [T_shift + 30*0.5*-0.004, LP_shift + 30*0.5*-0.005]),
#        'step3' : set_step(time = 0.01e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.0096, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 0.01e-3, qubits = qubits, voltages = [T_shift + 30*0.5*-0.004, LP_shift + 30*0.5*-0.005]),
        'step4' : set_step(time = 5e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.008, LP_shift + 30*0.5*0.008]),
#        'step5' : set_step(time = 1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*0.000]),
#        'step6' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.00, LP_factor*30*0.5*0.01]),
        }


read0_t1_cfg = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_shift + 30*0.5*0.002, LP_shift + 30*0.5*0.000],)
#        'step1' : set_step(time = 1e-6, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.002]),
        }


read_t1_cfg = {
        'step1' : set_step(time = 0.502e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        }


sequence_t1_cfg = [init_t1_cfg, read0_t1_cfg, read_t1_cfg]
sequence_t1_cfg_type = ['init', 'read0', 'read']


experiment.saveraw = True
experiment.threshold = 0.020
experiment.qubit_number = 1
experiment.seq_repetition = 50

experiment.add_measurement('T1_Scan', ['Ramsey12',], [ramsey12,], sequence_t1_cfg, sequence_t1_cfg_type)

experiment.add_X_parameter('T1_Scan', parameter = 'time', sweep_array = sweep_array(1e-6, 10e-3, 50), element = 'read0_step1')

experiment.add_Y_parameter('T1_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3),)

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

'''

#%%     PSB
'''
T_0 = 30*0.5*-0.011
LP_0 = 30*0.5*-0.000

T_start = 30*0.5*0.010
T_end = 30*0.5*-0.010

LP_start = 30*0.5*-0.020
LP_end = 30*0.5*0.020

T_steps = 20
LP_steps = 40

init_psb_cfg = {
        'step1' : set_step(time = 0.6e-3, qubits = qubits, voltages = [30*0.5*-0.010, 30*0.5*-0.001]),
        'step2' : set_step(time = 0.6e-3, qubits = qubits, voltages = [30*0.5*0.010, 30*0.5*-0.001]),
        }

#manip_cfg = {
#        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.004,LP_factor*30*0.5*0.016],)
#        }

read_psb_cfg = {
        'step1' : set_step(time = 0.255e-3, qubits = qubits, voltages = [T_0, LP_0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_0, LP_0]),
        'step3' : set_step(time = 0.5e-3, qubits = qubits, voltages = [T_0, LP_0]),
        }

sequence_psb_cfg = [init_psb_cfg, read_psb_cfg,]         ## the NAME here in this list is not important , only the order matters
sequence_psb_cfg_type = ['init', 'read']

experiment.saveraw = True
experiment.threshold = 0.018
experiment.qubit_number = 1
experiment.seq_repetition = 50

experiment.add_measurement('PSB_Scan', ['Ramsey12',], [ramsey12,], sequence_psb_cfg, sequence_psb_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey12')

#experiment.add_X_parameter('PSB_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5),)

experiment.add_X_parameter('PSB_Scan', parameter = 'voltage_1', 
                           sweep_array = sweep_array(T_start, T_end, T_steps), element = 'read_step1')
experiment.add_X_parameter('PSB_Scan', parameter = 'voltage_1', 
                           sweep_array = sweep_array(T_start, T_end, T_steps), element = 'read_step2')
experiment.add_X_parameter('PSB_Scan', parameter = 'voltage_1', 
                           sweep_array = sweep_array(T_start, T_end, T_steps), element = 'read_step3')


experiment.add_Y_parameter('PSB_Scan', parameter = 'voltage_2', 
                           sweep_array = sweep_array(LP_start, LP_end, LP_steps), element = 'read_step1')
experiment.add_Y_parameter('PSB_Scan', parameter = 'voltage_2', 
                           sweep_array = sweep_array(LP_start, LP_end, LP_steps), element = 'read_step2')
experiment.add_Y_parameter('PSB_Scan', parameter = 'voltage_2', 
                           sweep_array = sweep_array(LP_start, LP_end, LP_steps), element = 'read_step3')

#experiment.add_Y_parameter('PSB_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5),)

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

'''
'''
ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,0,0,:])
for i in range(1):
    for j in range(10,50):
        pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,i,j,:])
'''

'''
ds = experiment.data_set
T_gate = ds.sweep_data[0,0,:]
LP_gate = ds.voltage_2_set #sweep_array(30*0.5*0.012, 30*0.5*0.024, 10)
voltage = ds.raw_data[:,0,:,:,:].mean(axis = (2,3))
pt = MatPlot()
pt.add(x = T_gate,y = LP_gate, z = voltage)
'''
#%%
"""
calibration.run_experiment()

calibration.plot_average_probability()

DS = calibration.averaged_data

x = DS.frequency_shift_set.ndarray
y = DS.digitizerqubit_1.ndarray

pars, pcov = curve_fit(Func_Gaussian, x, y,)
frequency_shift = pars[1]

frequency = vsg2.frequency()+frequency_shift
vsg2.frequency(frequency)
#
#experiment.plot_average_probability()
#experiment.plot_save()

calibration.close()
reset_experiment()
"""
#%%        Active calibration

#experiment.add_measurement('Active_Calibration', ['Ramsey'], [ramsey], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Active_Calibration',)

#%%
"""
print('sweep parameter set')

experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
print('sequence loaded')

#experiment.load_sequence()
time.sleep(1)
#data_set = experiment.run_experiment()

experiment.convert_to_probability()
experiment.calculate_average_data()
experiment.plot_average_probability()

"""

#%% Charge noise Bob Joynt (Jelmer)
'''
from manipulation_library_Jelmer import ChargeNoiseBob_Jelmer

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

phase_1 = 9.6
phase_2 = 275.9

AMP_C = 30*0.5*-0.0304
AMP_T = 30*0.5*0.075

detuning_time = 80E-9;

amplitude = 1;
off_resonance_amplitude = 0.95;

decoupled_cphase = False;
decoupled_qubit = 'qubit_1';
add_dephase = False;
sigma1 = 0.6E6;
sigma2 = 0.6E6;

#init_state = '01+10';
init_state = '00+11';
#init_state = '00+01+10+11';
#init_state = '00+01';
#init_state = '10+11';
#init_state = '00+10';
#init_state = '01+11';

CN_manip = ChargeNoiseBob_Jelmer(name = 'Charge_Noise', pulsar = pulsar, phase_1 = phase_1, phase_2 = phase_2,
                                           amplitude_control = AMP_C, amplitude_target = AMP_T, detuning_time = detuning_time,
                                           off_resonance_amplitude = off_resonance_amplitude, decoupled_cphase = decoupled_cphase,
                                           decoupled_qubit = decoupled_qubit, add_dephase = add_dephase, sigma1 = sigma1, sigma2 = sigma2,
                                           init_state = init_state)

# Ramsey frequency update
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

# Charge noise experiment
experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [CN_manip, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 41), element = 'Charge_Noise')
#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')

# Rabi calibration measurement
experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')


experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 20), with_calibration = True)
#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%% Charge noise Bob Joynt (Jelmer v2)
'''
from manipulation_library_Jelmer import ChargeNoiseBob_Jelmer2

#experiment.name = 'NoiseExp'

experiment.qubit_number = 2
experiment.seq_repetition = 75
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.readout_time = 0.0015
experiment.threshold = 0.027

AMP_C = 30*0.5*-0.025
AMP_T = 30*0.5*0.011
detuning_time = 80E-9;
phase_1 = 236 #for target = 'qubit_1'
phase_2 = 155 #for target = 'qubit_2'

decoupled_cphase = False;
decoupled_qubit = 'qubit_1';

amplitude = 1.1;
off_resonance_amplitude = 1.1;

DFS = False;
add_dephase = True;
sigma = 0.2E6;
DD = 'None';

#experiment.label = 'DFS_01MHz_C'

CN_manip = ChargeNoiseBob_Jelmer2(name = 'Charge_Noise', pulsar = pulsar, phase_1 = phase_1, phase_2 = phase_2, amplitude_control = AMP_C,
                                           amplitude_target = AMP_T, detuning_time = detuning_time, off_resonance_amplitude = off_resonance_amplitude,
                                           decoupled_cphase = decoupled_cphase, decoupled_qubit = decoupled_qubit, DFS = DFS,
                                           add_dephase = add_dephase, sigma = sigma, DD = DD)

# Ramsey frequency update
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

# Charge noise experiment
experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [CN_manip, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 0.8e-6, 81), element = 'Charge_Noise')
#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')

# Rabi calibration measurement
experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')


experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 100), with_calibration = True)
#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''