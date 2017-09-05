# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from experiment_version2 import Experiment
from calibration import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey, Finding_Resonance, Rabi
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


#%% make manipulation
#def make_manipulation(manipulation = Manipulation(name = 'Manip'), qubits = [], **kw):
#
#    waiting_time = kw.pop('waiting_time', None)
#    amplitude = kw.pop('amplitude', None)
#
#    manip = make_Ramsey(manipulation = manipulation, qubits = qubits, waiting_time = waiting_time)
#
#    return manip
#


#%%
def make_manipulation_cfg():

    manipulation_cfg = {
            'gate1': ['X','Y'],
            'gate2': ['Y','X'],
            'gate3': ['CPhase'],
            'gate4': ['Z', 'X']
            }

    return manipulation_cfg

#%%

def make_experiment_cfg():
    global station

#    station = stationF006.initialize()
    time.sleep(1)
    awg = station.awg
    awg2 = station.awg2
#    awg.clock_freq(1e9)
#    awg2.clock_freq(1e9)
    
    vsg = station.vsg
    vsg2 = station.vsg2
    global digitizer
    digitizer = station.digitizer
#    awg.ch3_amp
#    pulsar = set_5014pulsar(awg = awg, awg2= awg2)

    qubit_1 = station.qubit_1
    qubit_2 = station.qubit_2

    qubits = [qubit_1, qubit_2]

    experiment = Experiment(name = 'experiment_test', qubits = [qubit_1, qubit_2], awg = awg, awg2 = awg2, pulsar = pulsar, 
                             vsg = vsg, vsg2 = vsg2, digitizer = digitizer)
    global sweep_loop1
    
    sweep_loop1 = {
#            'para1': [0.8,0.2,0.53,0.14,0.3],
            'para1': sweep_array(start = 0, stop = 1.2e-6, points = 31),
#            'para2': sweep_array(start = 0.1, stop = 0.5, points = 5),
            }

    sweep_loop2 = {
#            'para1': [-0.4,-0.5,-0.3,-0.8,-0.5],
            }
    sweep2_loop1 = {
#            'para1': sweep_array(start = 0, stop = 10e-6, points = 6),
            }
    sweep2_loop2 = {}
    
    experiment.sweep_loop1 = [sweep_loop1, sweep2_loop1]
    experiment.sweep_loop2 = [sweep_loop2, sweep2_loop2]

#    loop1_para1 = [1,2,344,553,3]
#    loop1_para2 = [33,2,11,22,3]
#    loop2_para1 = [1,2,3,4]

    loop1_para1 = 'loop1_para1'
    loop1_para2 = 'loop1_para2'
    loop2_para1 = 'loop2_para1'
    
    init_cfg = {
            'step1' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*-0.001]),
            'step2' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*-0.004, 30*0.5*0]),
            'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*-0.008, 30*0.5*0]),
            'step4' : set_step(time = 4e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#            'step5' : set_step(time = 500e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step6' : set_step(time = 2000e-6, qubits = qubits, voltages = [0.4, 0.5]),
            }

    manip_cfg = {
            'step1' : set_manip(time = 5e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016], length = loop1_para1, manip_elem = 'Rabi')
            }

    read_cfg = {
            'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#            'step2' : set_step(time = 3e-3, qubits = qubits, voltages = [0.4, 0.2]),
#            'step3' : set_step(time = 1e-6, qubits = qubits, voltages = [0.5, 0.2]),
            }
    
    init2_cfg = {
            'step1' : set_step(time = 1.5e-6, qubits = qubits, voltages = [30*0.5*1, 30*0.5*1]),
            'step2' : set_step(time = 1.5e-6, qubits = qubits, voltages = [30*0.5*1, 30*0.5*1]),
            }
    
    manip2_cfg = {
            'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016], parameter1 = loop1_para1, manip_elem = 'Ramsey')
            }
    
    read2_cfg = {
            'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
            'step2' : set_step(time = 3e-3, qubits = qubits, voltages = [0, 0.02]),
#            'step3' : set_step(time = 1e-6, qubits = qubits, voltages = [0.5, 0.2]),
            }

#    experiment.sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]
#    experiment.sequence_cfg_type = ['init', 'manip','read','init2', 'manip2', 'read2']

    experiment.sequence_cfg = [init_cfg, manip_cfg, read_cfg, ]         ## the NAME here in this list is not important , only the order matters
    experiment.sequence_cfg_type = ['init', 'manip','read',]
    
#    experiment.sequence2_cfg = [init_cfg, manip2_cfg, read2_cfg,]
#    experiment.sequence2_cfg_type = ['init', 'manip2','read2',]
    
    experiment.seq_cfg = [experiment.sequence_cfg, ]
    experiment.seq_cfg_type = [experiment.sequence_cfg_type,]

#    experiment.manip_elem = Ramsey(name = 'Ramsey', pulsar = pulsar)
    manip3_elem = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)
    manip_elem = Rabi(name = 'Rabi', pulsar = pulsar)
#    experiment.manip_elem.pulsar = None
    
    manip2_elem = Ramsey(name = 'Ramsey', pulsar = pulsar)
#    experiment.manip2_elem.pulsar = None
#    experiment.manip_elem = [manip_elem, manip2_elem]
    
    experiment.add_manip_elem('Finding_resonance', manip3_elem, seq_num = 1)
    experiment.add_manip_elem('Rabi', manip_elem, seq_num = 1)
    experiment.add_manip_elem('Ramsey', manip2_elem, seq_num = 1)
    experiment.make_sequencers(seq1_name = 'Seq1', )
    
#    experiment.manipulation_elements = {
#            'Rabi': None,
#            }

#    experiment.set_sweep(0)
#    experiment.set_sweep(1)
    return experiment


#%% sweep  outside a  sequence

def function(x):
    return True

Count = StandardParameter(name = 'Count', set_cmd = function)
Sweep_Count = Count[1:5:1]
#Sweep_VSGFreq = vsg.frequency[5e9:15e9:3e9]
#LP = Loop(sweep_values = Sweep_Count).each(dig)
#LP2 = Loop(sweep_values = Sweep_VSGFreq).loop(sweep_values = Sweep_Count).each(dig)

formatter = HDF5FormatMetadata()
data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
#data_location = '2017-08-18/20-40-19_T1_Vread_sweep'
experiment_name = 'Finding_Resonance'
sweep_type = 'Rabi_Sweep'
data_location = time.strftime("%Y-%m-%d/%H-%M-%S") + experiment_name + sweep_type
#data_set = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = data_IO,)

#print('loop.data_set: %s' % LP.data_set)

#data_set = LP.run()

#dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size)
def scan_outside_awg(name, label, set_parameter, measured_parameter, start, end, step):
    
    Sweep_Value = set_parameter[start:end:step]
    
    LOOP = Loop(sweep_values = Sweep_Value).each(measured_parameter)
    
    data_set = LOOP.get_data_set(location = data_location, loc_record = {'name': name, 'label': sweep_type}, io = data_IO,)
    
    data_set = LOOP.run()
    
    awg.stop()
    awg2.stop()
    
    vsg.status('Off')
    vsg2.status('Off')
    return data_set

def scan_inside_awg(name, label,):
    
    data = dig.get()
    
    pulse_length = sweep_loop1['para1'] if 'para1' in sweep_loop1 else 1

    data = np.array([data])
    
    pulse_length = np.array([pulse_length])

    data_array = DataArray(preset_data = data, name = 'digitizer',)

    pulse_array = DataArray(preset_data = pulse_length, name = name+'_set', is_setpoint = True)

    set_array = DataArray(preset_data = np.array([1]), name = 'none_set',  array_id = 'pulse_length_set', is_setpoint = True)

    data_set = new_data(arrays= [set_array, pulse_array, data_array] , location=data_location, loc_record = {'name':experiment_name, 'label':sweep_type}, io = data_IO,)

    return data_set

#%%

class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, 
                posttrigger_size, label=None, unit=None, instrument=None,
                **kwargs):
        
        global digitizer
        channel_amount = bin(digitizer.enable_channels()).count('1')
       
        super().__init__(name=name, shape=(channel_amount*memsize,), instrument=instrument, **kwargs)
        
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        
    def get(self):
#        res = digitizer.single_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        time.sleep(0.2)
        res = digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
#        res = multiple_trigger_acquisition(digitizer, self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
#        res = digitizer.single_software_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        print(res.shape)
        return res
        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)
#%%
def set_digitizer(digitizer):
    pretrigger=16
    mV_range=1000
    
    sample_rate = int(np.floor(61035/1))
    
    digitizer.sample_rate(sample_rate)
    
    sample_rate = digitizer.sample_rate()
    
    readout_time = 1e-3
    
    qubit_num = 1
    
    seg_size = ((readout_time*sample_rate+pretrigger) // 16 + 1) * 16
    
    sweep_num = len(sweep_loop1['para1']) if 'para1' in sweep_loop1 else 1
    import data_set_plot
    data_set_plot.loop_num = sweep_num
    
    repetition = 200
    
    memsize = int((repetition+1)*sweep_num*qubit_num*seg_size)
    posttrigger_size = seg_size-pretrigger
    
    #digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
    digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)
    #digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)
    
    digitizer.enable_channels(pyspcm.CHANNEL1 | pyspcm.CHANNEL2)
    
#    digitizer.enable_channels(pyspcm.CHANNEL1)
    digitizer.data_memory_size(memsize)
    
    digitizer.segment_size(seg_size)
    
    digitizer.posttrigger_memory_size(posttrigger_size)
    
    digitizer.timeout(60000)
    
    
    
    digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)
    
    #trig_mode = pyspcm.SPC_TMASK_SOFTWARE
    #trig_mode = pyspcm.SPC_TM_POS
    trig_mode = pyspcm.SPC_TM_POS | pyspcm.SPC_TM_REARM
    
    digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 800, level1 = 900)
    
    dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size)

    return digitizer, dig

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

def close():
    awg.stop()
    awg2.stop()
    time.sleep(0.5)
    vsg.status('Off')
    vsg2.status('Off')
    awg.delete_all_waveforms_from_list()
    awg2.delete_all_waveforms_from_list()
    return True


#%%

def data_set_plot(data_set, data_location, sweep_type):
    
    Plot = MatPlot()
    
    threshold = 0.025
    name = 'Rabi'
    
    raw_data_set = load_data(location = data_location, io = data_IO,)
    """
    for 1D sweep outside a sequence, sequence is only one unit sequence
    """
    if sweep_type == 1:
        
#        raw_data_set = load_data(location = data_location, io = data_IO,)
        
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        
        x_data = data_set_P.arrays['vsg2_frequency_set'].ndarray
        P_data = data_set_P.arrays['digitizer'].ndarray.T[0]
        
        x = x_data
        y = P_data
        
        plt.plot(x, y)
        datacursor()
    """
    for 2D sweep both inside sequence and outside a sequence
    """
    if sweep_type == 2:
#        raw_data_set = load_data(location = data_location, io = data_IO,)
        
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        
        x_data = data_set_P.arrays['vsg2_frequency_set'].ndarray
        y_data = data_set_P.arrays[name+'_set'].ndarray[0]
        P_data = data_set_P.arrays['digitizer'].ndarray
        
        X, Y = np.meshgrid(x_data, y_data)
        
        plt.pcolor(X,Y,P_data.T)
        datacursor()
    """
    for 1D sweep inside a sequence, no qcodes-loop function
    """
    if sweep_type == 0:
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        x_data = data_set_P.arrays[name+'_set'].ndarray[0]

        P_data = data_set_P.arrays['digitizer'].ndarray[0]
        
        plt.plot(x_data, P_data)
        datacursor()

        return 0
#%% plot with cursor

#import matplotlib.pyplot as plt
#import numpy as np
#from mpldatacursor import datacursor
'''
data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
lines = ax.plot(data)
ax.set_title('Click somewhere on a line')

datacursor(lines)

plt.show()
'''
#%% test
experiment = make_experiment_cfg()
pulsar = experiment.pulsar
awg = experiment.awg
awg2 = experiment.awg2
vsg = experiment.vsg
vsg2 = experiment.vsg2
digitizer, dig = set_digitizer(experiment.digitizer)

experiment.generate_1D_sequence()

vsg.frequency(18.4e9)
vsg2.frequency(19.672e9)
#experiment.load_sequence()
time.sleep(1)
experiment.run_experiment()
#time.sleep(1)
#pulsar.start()


#data_set = scan_outside_awg(name = experiment_name, label = sweep_type, set_parameter = vsg2.frequency, measured_parameter = dig, 
#                            start=19.6650e9, end=19.68e9, step=1e6)

#data_set = scan_inside_awg(name = experiment.name, label = sweep_type)

#data_set_plot(data_set, data_location,1)

#data = dig.get()
#pulse_length = sweep_loop1['para1']
##
#data = np.array([data])
#pulse_length = np.array([pulse_length])
#
#data_array = DataArray(preset_data = data, name = 'digitizer',)
##
#pulse_array = DataArray(preset_data = pulse_length, name = 'pulse_length', is_setpoint = True)
##
#set_array = DataArray(preset_data = np.array([1]), name = 'pulse_length',  array_id = 'pulse_length_set', is_setpoint = True)
##
#data_set = new_data(arrays= [set_array, pulse_array, data_array] , location=data_location, loc_record = {'name':experiment_name, 'label':sweep_type}, io = data_IO,)
#
#DS = convert_to_probability(data_set, 0.025)
#
#x_data = DS.arrays['frequency_set'].ndarray[0]
#
#P_data = DS.arrays['digitizer'].ndarray[0]

#
