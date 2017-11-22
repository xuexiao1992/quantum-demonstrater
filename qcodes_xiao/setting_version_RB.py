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
from calibration_version2 import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability, seperate_data, organise

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY, Ramsey_all, AllXY_all, CPhase_Calibrate, Charge_Noise, DCZ, Sychpulses1, Sychpulses2, Rabi_all, Wait, MeasureTminus, Ramsey_00_11_basis, RB, Rabi_detuning, RB_all,RB_Marcus
#from digitizer_setting import digitizer_param

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
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

from plot_functions import plot1D, plot2D

from mpldatacursor import datacursor
import sys
sys.path.append('C:\\Users\\LocalAdmin\\Documents\\GitHub\\PycQED_py3\\pycqed\\measurement\\waveform_control')
import pulsar as ps
import element as ele
import time

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
####
#pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)

Count = StandardParameter(name = 'Count', set_cmd = Counts)


    
init_cfg = {
        'step1' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*-0.001]),
        'step2' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*-0.004, 30*0.5*0]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*-0.011, 30*0.5*0]),
        'step4' : set_step(time = 4e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step5' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.000]),
        }

manip_cfg = {
        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }

read_cfg = {
        'step1' : set_step(time = 0.262e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }
init2_cfg = {
        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.000]),
        }
manip2_cfg = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }
read2_cfg = {
        'step1' : set_step(time = 0.262e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
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
        'step3' : set_step(time = 2.85e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step4' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.000]),
        }

sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence_cfg_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

sequence1_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence1_cfg_type = ['init', 'manip21','read', 'init2', 'manip22', 'read2']

sequence11_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence11_cfg_type = ['init', 'manip31','read', 'init2', 'manip32', 'read2']

sequence21_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence21_cfg_type = ['init', 'manip41','read', 'init2', 'manip42', 'read2']

sequence31_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence31_cfg_type = ['init', 'manip51','read', 'init2', 'manip52', 'read2']



sequence2_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence2_cfg_type = ['init', 'manip','read',]

sequence3_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence3_cfg_type = ['init', 'manip2','read',]

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

#%%

#digitizer, dig = set_digitizer(experiment.digitizer)


qubit_1.Pi_pulse_length = 250e-9
qubit_2.Pi_pulse_length = 250e-9
qubit_1.halfPi_pulse_length = 125e-9
qubit_2.halfPi_pulse_length = 125e-9
qubit_2.CRot_pulse_length = 286e-9
def reset_experiment():
    experiment.reset()
    
    experiment.qubit_number = 1
    experiment.threshold = 0.017
    experiment.seq_repetition = 100
    
    print('experiment reset')

    return True
#%%        Set Experiment

experiment0 = Experiment(name = 'RB_experiment', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                        awg = awg, awg2 = awg2, pulsar = pulsar, 
                        vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

print('experiment initialized')

#self.digitizer_trigger_channel = 'ch5_marker1'
#self.digitier_readout_marker = 'ch6_marker1'

experiment0.qubit_number = 2
experiment0.readnames = ['Qubit2', 'Qubit1']
experiment0.threshold = 0.034
experiment0.seq_repetition = 100
experiment0.saveraw = False



rb = RB(name = 'RB', pulsar = pulsar)

rb2 = RB(name = 'RB2', pulsar = pulsar, qubit = 'qubit_1')

rb12 = RB_all(name = 'RB12', pulsar = pulsar)

ramsey12 = Ramsey_all(name = 'Ramsey12', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.2, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 300e-9,)

rabi12 = Rabi_all(name = 'Rabi12', pulsar = pulsar)

crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.027, frequency_shift = 0.0573006e9, duration_time = 275.516e-9)

rabi_off = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1', frequency_shift = -30e6)

crot_freq_bare = 0.0483323e9
crot_freq = 0.0577177e9


#%%     Randomized_Benchmarking Marcus
'''
rb_marcus = RB_Marcus(name = 'RB_M', pulsar = pulsar, detuning_time = 80e-9, phase_1 = 85, phase_2 = 130, Pi_amplitude = 1,)

rb_marcus0 = RB_Marcus(name = 'RB_M0', pulsar = pulsar, detuning_time = 80e-9, phase_1 = 85, phase_2 = 130, Pi_amplitude = 0,)

experiment0.qubit_number = 2
experiment0.seq_repetition = 100
experiment0.calibration_qubit = 'all'
experiment0.saveraw = True

experiment0.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment0.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment0.add_measurement('RB_1', ['RB_M', 'CRot'], [rb_marcus, crot], sequence1_cfg, sequence1_cfg_type)
experiment0.add_X_parameter('RB_1', parameter = 'clifford_number', sweep_array = sweep_array(0, 21, 22), element = 'RB_M')

experiment0.add_measurement('RB_0', ['RB_M0', 'CRot'], [rb_marcus0, crot], sequence21_cfg, sequence21_cfg_type)
experiment0.add_X_parameter('RB_0', parameter = 'clifford_number', sweep_array = sweep_array(0, 21, 22), element = 'RB_M0')


experiment0.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment0.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment0.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 39, 40), element = 'RB_M', with_calibration = True)

experiment0.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment0.generate_1D_sequence()
#experiment0.load_sequence()
print('sequence loaded')
'''
#%%     



#experiment0.run_experiment()
