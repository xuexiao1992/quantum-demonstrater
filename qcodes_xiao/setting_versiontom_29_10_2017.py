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
from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY, Ramsey_all, AllXY_all, CPhase_Calibrate, Charge_Noise, DCZ, Sychpulses1, Sychpulses2, Rabi_all, Wait, MeasureTminus, Ramsey_00_11_basis, Rabi_detuning
#from manipulation_library import RB, Rabi_detuning, RB_all,RB_Marcus, Ramsey_withnoise, MultiCPhase_Calibrate, RBinterleavedCZ
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

from plot_functions import plot1D, plot2D, fitcos, plot1Ddata, plot2Ddata

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
station = stationF006.initialize()
pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)
vsg = station.vsg
vsg2 = station.vsg2

Count = StandardParameter(name = 'Count', set_cmd = Counts)


time.sleep(1)
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
keithley = station.keithley

digitizer = station.digitizer

T = G.T
LP = G.LP


AMP = keithley.amplitude

T_factor = 1
LP_factor = 1

init_cfg = {
        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.004, LP_factor*30*0.5*-0.001]),
        'step2' : set_step(time = 4e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.005]),
        'step3' : set_step(time = 0.01e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.007, LP_factor*30*0.5*0]),
        'step4' : set_step(time = 5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step5' : set_step(time = 1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*0.000]),
#        'step6' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.00, LP_factor*30*0.5*0.01]),
        }

manip_cfg = {
        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.004,LP_factor*30*0.5*0.018],)
#        'step1' : set_manip(time = 101e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.004,LP_factor*30*0.5*0.016],)
        }

read0_cfg = {
#        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_shift+30*0.5*-0.004, 30*0.5*0.016],)
        'step1' : set_step(time = 0.23e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.002]),
        }


read_cfg = {
#        'step1' : set_step(time = 0.2e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.002]),
        'step1' : set_step(time = 0.30e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        }
init2_cfg = {
        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*0.000]),
        }
manip2_cfg = {
        'step1' : set_manip(time = 2.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.004,LP_factor*30*0.5*0.018],)
#        'step1' : set_manip(time = 101e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.004,LP_factor*30*0.5*0.016],)
        }
read2_cfg = {
        'step1' : set_step(time = 0.302e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 0.688e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
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


vsg.frequency(18.3612e9)
vsg2.frequency(19.6972e9)

vsg.power(18.8)
vsg2.power(8)



qubit_1.Pi_pulse_length = 250e-9
qubit_2.Pi_pulse_length = 250e-9

qubit_1.halfPi_pulse_length = 125e-9
qubit_2.halfPi_pulse_length = 125e-9

qubit_2.CRot_pulse_length = 300e-9
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


print('experiment initialized')

#self.digitizer_trigger_channel = 'ch5_marker1'
#self.digitier_readout_marker = 'ch6_marker1'

experiment.qubit_number = 2
experiment.readnames = ['Qubit2', 'Qubit1']
#experiment.threshold = 0.020
experiment.threshold = 0.015
#experiment.threshold = 0.0240
#experiment.threshold = 0.0045
experiment.seq_repetition = 100
experiment.saveraw = False


experiment.readout_time = 0.0008



#calibration.qubit_number = 2
#calibration.threshold = 0.013
#calibration.seq_repetition = 100

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 100e-9, waiting_time = 300e-9)
ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, qubit = 'qubit_1', duration_time = 100e-9, waiting_time = 300e-9)

ramsey3 = Ramsey(name = 'Ramsey3', pulsar = pulsar,qubit = 'qubit_1', waiting_time = 300e-9)
ramsey4 = Ramsey(name = 'Ramsey4', pulsar = pulsar,qubit = 'qubit_2', waiting_time = 300e-9)

allxy = AllXY(name = 'AllXY', pulsar = pulsar, qubit = 'qubit_2')

'''
rb = RB(name = 'RB', pulsar = pulsar)

rb2 = RB(name = 'RB2', pulsar = pulsar, qubit = 'qubit_1')

rb12 = RB_all(name = 'RB12', pulsar = pulsar)
'''

allxy2 = AllXY(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1')

rabi12 = Rabi_all(name = 'Rabi12', pulsar = pulsar)

rabi123 = Rabi_all(name = 'Rabi123', pulsar = pulsar)

ramsey12 = Ramsey_all(name = 'Ramsey12', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.2, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 260e-9,detune_q1= False)


allxy12 = AllXY_all(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1',)

finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)

rabi = Rabi(name = 'Rabi', pulsar = pulsar)

wait = Wait(name = 'Wait', pulsar = pulsar)
#
crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.028*T_factor, amplitudepi = 0.8, 
            frequency_shift = 0.0512e9, duration_time = 300e-9)

rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)

rabi3 = Rabi(name = 'Rabi3', pulsar = pulsar, amplitude = 1, qubit = 'qubit_2',)

rabi_off = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1', frequency_shift = -30e6)

crot_freq_bare = 0.05188e9
crot_freq = 0.0564e9


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

#19572236000
experiment.saveraw = True

experiment.qubit_number = 1
experiment.seq_repetition = 100

#experiment.threshold = 0.018
experiment.calibration_qubit = 'qubit_2'


rabi = Rabi(name = 'Rabi', pulsar = pulsar, length = Pi_pulse_length)
finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)
#ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 110e-9, waiting_time = 300e-9)


experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey')


#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg, sequence3_cfg_type)
experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg2, sequence3_cfg2_type)
#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi12,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Finding_resonance'], [finding_resonance,], sequence3_cfg2, sequence3_cfg2_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.72e9, 19.84e9, 61))
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.2e9, 20e9, 51))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.006e9, 0.006e9, 31), element = 'Rabi')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey')

experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi')
#
#experiment.add_X_parameter('Rabi_Scan', parameter = 'phase_2', sweep_array = sweep_array(0, 360, 41), element = 'Ramsey')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 30e-6, 51), element = 'Ramsey')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)

#experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 3),)
#experiment.add_Y_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.475e9, 19.490e9, 31))
#experiment.set_sweep(repetition = True, plot_average = False, count = 3)
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
for i in range(1,10):
    for j in range(10,20):
        pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,i,j,:])


ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,0,0,0,0,:])
for i in range(20,30):
    for j in range(10,30):
        pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,i,0,0,j,:])



import matplotlib.pyplot as plt
plt.hist(yy.reshape(3100*48,))


'''
#%%
'''
experiment.calibration_qubit = 'qubit_2'

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = halfPi_pulse_length, waiting_time = 300e-9)

ramsey_echo = Ramsey(name = 'Ramsey_Echo', pulsar = pulsar, duration_time = halfPi_pulse_length, waiting_time = 0e-9, echo = True)


experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey')

experiment.add_measurement('Echo_Scan', ['Ramsey_Echo'], [ramsey_echo,], sequence3_cfg2, sequence3_cfg2_type)
experiment.add_X_parameter('Echo_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 25e-6, 51), element = 'Ramsey_Echo')

experiment.add_Y_parameter('Echo_Scan', parameter = Count, sweep_array = sweep_array(1, 8, 8), with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')

'''




#%%     CRot
'''
experiment.qubit_number = 1
qubit_2.Pi_pulse_length = 240e-9
experiment.threshold = 0.011
experiment.saveraw = True
experiment.readout_time = 0.0008


experiment.add_measurement('CRot_Scan', ['CRot'], [crot], sequence3_cfg, sequence3_cfg_type)

experiment.add_X_parameter('CRot_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.03e9, 0.07e9, 31), element = 'CRot')
#experiment.add_X_parameter('CRot_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1e-6, 41), element = 'CRot')

#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.385e9, 18.40e9, 16))
#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.power, sweep_array = sweep_array(17.25, 17.35, 51))
experiment.add_Y_parameter('CRot_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3),)
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 4)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
'''

#%%     Q1 adiabatic
'''
finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar, qubit = 'qubit_1')

experiment.saveraw = True
experiment.qubit_number = 2
experiment.seq_repetition = 100
#experiment.threshold = 0.018
#qubit_2.Pi_pulse_length = 280e-9
#qubit_1.Pi_pulse_length = 280e-9
#experiment.readout_time = 0.0008
rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)

#18328513000
experiment.add_measurement('Rabi_Scan', ['Finding_resovsnance','CRot'], [finding_resonance, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)


#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.32e9, 18.37e9, 51))
experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.1e9, 18.7e9, 31))

#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.03e9, 0.07e9, 31), element = 'CRot')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.9e-6, 31), element = 'Rabi2')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.006, 30*0.5*-0.012, 41), element = 'init_step3')

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
#clifford_sets = generate_randomized_clifford_sequence()
experiment.qubit_number = 1
experiment.seq_repetition = 100
experiment.calibration_qubit = 'qubit_2'


experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey')

experiment.add_measurement('RB', ['RB'], [rb,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 27, 28), element = 'RB')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')
experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 31, 32), element = 'RB', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%     Randomized_Behcnmarking Qubit1
'''
experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'qubit_1'

experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey2')


experiment.add_measurement('RB_Q1', ['RB2', 'CRot'], [rb2, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB_Q1', parameter = 'clifford_number', sweep_array = sweep_array(0, 27, 28), element = 'RB2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')
experiment.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 31, 32), element = 'RB2', with_calibration = True)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''

#%%        Randomized_Benchmarking Q1 & Q2

'''
experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB', ['RB12', 'CRot'], [rb12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 23, 24), element = 'RB12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')


experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 31, 32), element = 'RB12', with_calibration = True)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(15)
experiment.run_experiment()
'''

#%%        Randomized_Benchmarking Q1 & Q2 interleaved with CZ

'''
phase_1 = 84 + 180
phase_2 =  80 + 180
AMP_C = 30*0.5*-0.02549
AMP_T = 30*0.5*0.04

phase_2 = 80


RBI = RBinterleavedCZ(name = 'RBI', pulsar = pulsar, amplitude_control = AMP_C, amplitude_target = AMP_T, phase_1 = phase_1, phase_2 = phase_2 , 
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1', control = 0)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'

experiment.calibration_qubit == 'qubit_2'

#
#experiment.add_measurement('Ramsey_Scan', ['Ramsey3','CRot'], [ramsey3, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.05e9, 0.05e9, 11), element = 'Ramsey3')


experiment.add_measurement('Ramsey_Scan', ['Ramsey4','CRot'], [ramsey4, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')

#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB', ['RBI', 'CRot'], [RBI, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB', parameter = 'clifford_number', sweep_array = sweep_array(0, 23, 24), element = 'RBI')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')


experiment.add_measurement('Rabi_Scan2', ['Rabi3','CRot'], [rabi3, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi3')

#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)

experiment.add_Y_parameter('RB', parameter = 'sequence_number', sweep_array = sweep_array(0, 29, 30), element = 'RBI', with_calibration = True)
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(15)
experiment.run_experiment()

'''


#%%     Randomized_Benchmarking Marcus
'''
rb_marcus = RB_Marcus(name = 'RB_M', pulsar = pulsar, detuning_time = 80e-9, phase_1 = 182, phase_2 = 206, Pi_amplitude = 1,)

experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.calibration_qubit = 'all'
experiment.saveraw = True

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('RB_Q1', ['RB_M', 'CRot'], [rb_marcus, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.655e9, 19.695e9, 21))
experiment.add_X_parameter('RB_Q1', parameter = 'clifford_number', sweep_array = sweep_array(0, 19, 20), element = 'RB_M')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('RB_Q1', parameter = 'sequence_number', sweep_array = sweep_array(0, 49, 50), element = 'RB_M', with_calibration = True)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
#time.sleep(15)
#experiment.run_experiment()
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


phase_1 = 150
phase_2 = 343
AMP_C = 30*0.5*-0.0263
AMP_T = 30*0.5*0.02

charge_noise_bob = Charge_Noise_Bob3(name = 'Charge_Noise', pulsar = pulsar, detuning_time = 80e-9, 
                                    phase_1 = Phase1, phase_2 = Phase2, off_resonance_amplitude = 1.2,
                                    add_dephase = False, decoupled_qubit = 'qubit_1',
                                    amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 0,
                                    decoupled_cphase = False)

charge_noise_bob2 = Charge_Noise_Bob3(name = 'Charge_Noise_2', pulsar = pulsar, detuning_time = 80e-9, 
                                     phase_1 = Phase1, phase_2 = Phase2, off_resonance_amplitude = 1.2,
                                     add_dephase = False, decoupled_qubit = 'qubit_1',
                                     amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 1,
                                     decoupled_cphase = False)


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')

experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [charge_noise_bob, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 41), element = 'Charge_Noise')
#
#experiment.add_measurement('CN2', ['Charge_Noise_2', 'CRot'], [charge_noise_bob2, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('CN2', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 41), element = 'Charge_Noise_2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')



#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)
experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 10), with_calibration = True)

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


phase_1 = 84
phase_2 = 13
AMP_C = 30*0.5*-0.0255
AMP_T = 30*0.5*0.04


charge_noise_bob = Charge_Noise_Bob_withaddednoise(name = 'Charge_Noise', pulsar = pulsar, detuning_time = 80e-9, 
                                    phase_1 = phase_1, phase_2 = phase_2 , off_resonance_amplitude = 1.2,
                                    add_dephase = False, decoupled_qubit = 'qubit_1',
                                    amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 0,
                                    decoupled_cphase = False, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.0e6 , sigma4=0.0e6)

charge_noise_bob2 = Charge_Noise_Bob_withaddednoise(name = 'Charge_Noise_2', pulsar = pulsar, detuning_time = 80e-9, 
                                     phase_1 = phase_1, phase_2 = phase_2 , off_resonance_amplitude = 1.2,
                                     add_dephase = False, decoupled_qubit = 'qubit_1',
                                     amplitude_control = AMP_C, amplitude_target = AMP_T, DFS = 1,
                                     decoupled_cphase = False, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.0e6 , sigma4=0.0e6)


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')
##
#experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [charge_noise_bob, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 41), element = 'Charge_Noise')

experiment.add_measurement('CN2', ['Charge_Noise_2', 'CRot'], [charge_noise_bob2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('CN2', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, 1e-6, 41), element = 'Charge_Noise_2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#
#experiment.add_measurement('Rabi_Scan3', ['Rabi123','CRot'], [rabi123, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('Rabi_Scan3', parameter = 'duration_time', sweep_array = sweep_array(0, 30e-6, 2), element = 'Rabi123')

#experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)
experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 100, 10), with_calibration = True)


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
phase_1 = 150
phase_2 = 343

#cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')
#
#cphase2 = CPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.0283, detuning_amplitude2 = 30*0.5*0.01,
#                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.23, control_qubit = 'qubit_2')


cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 0, 
                          detuning_amplitude = 30*0.5*-0.0263, detuning_amplitude2 = 30*0.5*0.02,
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1')

cphase2 = CPhase_Calibrate(name = 'CPhase2', pulsar = pulsar, Pi_amplitude = 1, 
                           detuning_amplitude = 30*0.5*-0.0263, detuning_amplitude2 = 30*0.5*0.02,
                          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1')


experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('CPhase_Calibration', ['CPhase','CRot'], [cphase, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase')
experiment.add_X_parameter('CPhase_Calibration', parameter = 'phase', sweep_array = sweep_array(0, 360, 31), element = 'CPhase')

experiment.add_measurement('CPhase_Calibration2', ['CPhase2','CRot'], [cphase2, crot], sequence21_cfg, sequence21_cfg_type)
experiment.add_X_parameter('CPhase_Calibration2', parameter = 'phase', sweep_array = sweep_array(0, 360, 31), element = 'CPhase2')
#experiment.add_X_parameter('CPhase_Calibration2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 30), element = 'CPhase2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('CPhase_Calibration', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)

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
AMP_C = 30*0.5*-0.02549
AMP_T = 30*0.5*0.04
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

experiment.add_Y_parameter('CPhase_Calibration', parameter = Count, sweep_array = sweep_array(1, 5, 10), with_calibration = True)

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



phase1 = 90
phase2 = 90


AMP_C = 30*0.5*-0.027
AMP_T = 30*0.5*0.00
#AMP_C = 30*0.5*-0.00
#AMP_T = 30*0.5*0.00

dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = AMP_C, detuning_amplitude2 = AMP_T, 
          detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1')

dcz2 = DCZ(name = 'DCZ2', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude =  AMP_C, detuning_amplitude2 = AMP_T, 
           detuning_time = 80e-9, phase = 0, off_resonance_amplitude = 1.2, control_qubit = 'qubit_1')

experiment.calibration_qubit = 'all'


experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('DCZ', ['DCZ','CRot'], [dcz, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('DCZ', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 3e-6, 40), element = 'DCZ')
#experiment.add_X_parameter('DCZ', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ')

#experiment.add_measurement('DCZ2', ['DCZ2','CRot'], [dcz2, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('DCZ2', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ2')
#experiment.add_X_parameter('DCZ2', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 200e-9, 40), element = 'DCZ2')

experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

experiment.add_Y_parameter('DCZ', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
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
#experiment.threshold = 0.021
experiment.seq_repetition = 100

experiment.calibration_qubit = 'qubit_2'

experiment.add_measurement('Ramsey_Scan', ['Ramsey4'], [ramsey4,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey4')

experiment.add_measurement('AllXY_calibration', ['AllXY'], [allxy,], sequence2_cfg, sequence2_cfg_type)
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 21), element = 'AllXY')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 10), with_calibration = True)
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
experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.040e9, 0.065e9,21), element = 'CRot')
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
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.065e9, 21), element = 'CRot')
experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'CRot')

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

experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.25e9, 18.4e9, 76))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 21), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.07e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.06e9, 21), element = 'CRot')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.5e-6, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.005, 30*0.5*-0.011, 41), element = 'init_step3')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.30e9, 18.50e9, 11))
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 11, 3))
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
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9,21), element = 'Ramsey2')



experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3))
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
rabi2_det = Rabi_detuning(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_2',)

experiment.seq_repetition = 200
experiment.qubit_number = 2

experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2_det, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(30*0.5*-0.020, 30*0.5*-0.033, 31), element = 'Rabi2')

experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.610e9, 19.64e9, 31))
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

experiment.seq_repetition = 100
experiment.saveraw = True
experiment.add_measurement('Rabi_Scan', ['Rabi3','CRot'], [rabi3, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1e-6, 21), element = 'Rabi3')
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.5e9, 19.65e9, 76))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 3), element = 'Rabi3')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 21), element = 'Rabi3')

#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.565e9, 19.575e9, 21))

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(0, 11, 3))

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)

#%% Measure Q2 ramsey
'''
experiment.qubit_number = 2
experiment.seq_repetition = 100
experiment.add_measurement('Ramsey_Scan', ['Ramsey4','CRot'], [ramsey4, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.004e9, 0.004e9,21), element = 'Ramsey4')

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3))
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.380e9, 18.40e9, 21))


print('sweep parameter set')

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

'''
#%% Simultaneous pulse measure Q1 and Q2 frequency Rabi
'''
experiment.calibration_qubit = 'all'

experiment.saveraw = True
experiment.readout_time = 0.0008
#experiment.calibration_qubit = 'all'
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9,21), element = 'Rabi12')
experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 3e-6, 51), element = 'Rabi12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.008, 30*0.5*-0.012, 41), element = 'init_step3')
#
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.06e9, 31), element = 'CRot')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.67e9, 19.68e9, 20))
#experiment.add_X_parameter('Rabi_Scan', parameter = 'T_amplitude', sweep_array = sweep_array(0, 30*0.5*0.008, 31), element = 'Rabi12')

#experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = True)
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 11, 10), with_calibration = True)
#experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.642e9, 19.672e9, 31))


print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%% Simultaneous pulse measure Q2 frequency ramsey
'''
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.002e9, 0.002e9, 21), element = 'Ramsey12')
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey12')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 10), with_calibration = False)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
'''
#%%Simultaneous pulse measure Q2 frequency ramsey with noise
'''
ramsey_withnoise = Ramsey_withnoise(name = 'Ramsey_withnoise', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.2, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 300e-9,detune_q1= True, sigma1 =30*0.5*0.0000 , sigma2=30*0.5*0.000, sigma3 =0.0e6 , sigma4=0.0e6)



experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')



experiment.add_measurement('Ramsey_Scan2', ['ramsey_withnoise','CRot'], [ramsey_withnoise, crot], sequence21_cfg, sequence21_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 31), element = 'Ramsey12')
experiment.add_X_parameter('Ramsey_Scan2', parameter = 'waiting_time', sweep_array = sweep_array(0, 1.6e-6, 41), element = 'ramsey_withnoise')

#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.66e9, 19.68e9, 21))
#experiment.add_Y_parameter('Ramsey_Scan', parameter = 'dummy', sweep_array = sweep_array(0, 10, 10), element = 'ramsey_withnoise', with_calibration = False)
experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 10), with_calibration = True)


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
experiment.calibration_qubit = 'all'
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 2e-6, 21), element = 'Ramsey12')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 21))
#

experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21, 21), element = 'AllXY12')

experiment.add_Y_parameter('Ramsey_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = True)

print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count =5)

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