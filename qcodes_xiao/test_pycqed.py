# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:59:34 2017

@author: think
"""

import sys
import qcodes as qc

qc_config = {'datadir': 'D:/users/LukaBavdaz/users/petitlp/measurements/', 'PycQEDdir': 'D:/users/LukaBavdaz/Pycqed'}

import pycqed as pq
import numpy as np
from pycqed.measurement import measurement_control
from pycqed.measurement.sweep_functions import None_Sweep
import pycqed.measurement.detector_functions as det

import temp
#station = temp.initialize(server_name = default_server_name)
station = temp.initialize()

MC = measurement_control.MeasurementControl('MC',live_plot_enabled=True, verbose=True)
MC.station = station
#station = MC.station
station.add_component(MC)

def MC_sweep1D(gate, r, outputs, avg=1):
    
    sweep_gates = [getattr(station.gates, x) for x in gate]
    instrument = output_map[outputs]
    sweep_gates_range = []
    
    if np.size(r)==3 :
        sweep_range = np.linspace(r[0], r[1], 1+np.floor(abs(r[0]-r[1])/r[2]))
        if len(gate)>1:
            sweep_gates_range = [[x for i in gate ] for x in sweep_range] 
        else:
            sweep_gates_range =  sweep_range
    else:
        sweep_range = [np.linspace(r[i][0], r[i][1], 1+np.floor(abs(r[i][0]-r[i][1])/r[i][2])) for i, val in enumerate(gate)]
        sweep_gates_range = [[x[i] for x in sweep_range] for i, val in enumerate(sweep_range[0]) ] 
    
    MC.soft_avg(avg)
    MC.set_sweep_functions(sweep_gates)
    MC.set_sweep_points(sweep_gates_range)    
    MC.set_detector_function(instrument)
    dat = MC.run('1D test')
    
    return dat


def MC_sweep2D(gate1, r1, gate2, r2, outputs, avg=1):

    sweep_gate = getattr(station.gates, gate1)
    step_gate = getattr(station.gates, gate2) 
    
    sweep_pts = np.linspace(r1[0], r1[1], 1+np.floor(abs(r1[0]-r1[1])/r1[2]))
    step_pts = np.linspace(r2[0], r2[1], 1+np.floor(abs(r2[0]-r2[1])/r2[2]))
    
    instrument = output_map[outputs]
    
    MC.soft_avg(avg)
    MC.set_sweep_function(sweep_gate)
    MC.set_sweep_function_2D(step_gate)
    MC.set_sweep_points(sweep_pts)
    MC.set_sweep_points_2D(step_pts)
    MC.set_detector_function(instrument)
    dat=MC.run('test', mode='2D')
    
    return dat
    
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.pulse import SquarePulse
from pycqed.measurement.waveform_control.viewer import *
    
def AWG_Test():
    
    station.pulsar =  Pulsar()
    station.pulsar.AWG = station.components['awg']
    marker1highs=[2,2,2.7,2]
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        station.pulsar.define_channel(id='ch{}'.format(i+1),
                                    name='ch{}'.format(i+1), type='analog',
                                    # max safe IQ voltage
                                    high=.7, low=-.7,
                                    offset=0.0, delay=0, active=True)
        station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                    name='ch{}_marker1'.format(i+1),
                                    type='marker',
                                    high=marker1highs[i], low=0, offset=0.,
                                    delay=0, active=True)
        station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                    name='ch{}_marker2'.format(i+1),
                                    type='marker',
                                    high=2.0, low=0, offset=0.,
                                    delay=0, active=True)
   
    #Implementation of a sequence element.
    #Basic idea: add different pulses, and compose the actual numeric
    #arrays that form the amplitudes for the hardware (typically an AWG)
    V_empty = 0
    V_load = 0.05
    V_read = 0.025
    
    t_empty = 20e-6
    t_load = 10e-6
    t_read = 20e-6
    
    Npoints = 1
    
    Empty_elt = Element('Empty_elt', pulsar=station.pulsar)
    Empty_elt.add(SquarePulse(name='square_empty', channel='ch1', amplitude=V_empty, start=0, length=t_empty), name='Empty')

    Load_elt = Element('Load_elt', pulsar=station.pulsar)
    Load_elt.add(SquarePulse(name='square_load', channel='ch1', amplitude=V_load, start=0, length=t_load), name='Load')
    
    Read_elt = Element('Read_elt', pulsar=station.pulsar)
    Read_elt.add(SquarePulse(name='square_read', channel='ch1', amplitude=V_read, start=0, length=t_read), name='Read')
    
    ReadEmpty_elt = Element('ReadEmpty_elt', pulsar=station.pulsar)
    ReadEmpty_elt.add(SquarePulse(name='square_read', channel='ch1', amplitude=V_read, start=0, length=t_read), name='Read')
    ReadEmpty_elt.add(SquarePulse(name='square_empty', channel='ch1', amplitude=V_empty, start=0, length=t_empty), name='Empty', refpulse='Read', refpoint='end', refpoint_new='start')
    
    elts = [Empty_elt, Load_elt, Read_elt, ReadEmpty_elt]

    T1_seq = Sequence('T1_Sequence')
    
    T1_seq.append('Empty_0', 'Empty_elt')

    for i in range(Npoints):
    
        name_Load = 'Load_{}'.format(i)
        name_ReadEmpty = 'ReadEmpty_{}'.format(i)
        T1_seq.append(name_Load, 'Load_elt', repetitions=i+1)
        T1_seq.append(name_ReadEmpty, 'ReadEmpty_elt')
    
        
    ss = station.pulsar.program_awg(T1_seq, *elts)

    win=show_element_pyqt(ReadEmpty_elt, QtPlot_win=None, color_idx=None, channels=['ch1', 'ch2', 'ch3', 'ch4'])
    ch1_wf = test_elt.waveforms()[1]['ch1']