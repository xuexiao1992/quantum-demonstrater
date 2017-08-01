# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:31:17 2017

@author: X.X
"""

import numpy as np
from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.sequence import Sequence
from gate import Single_Qubit_Gate#, Two_Qubit_Gate
from manipulation import Manipulation
#from initialize import Initialize
#from readout import Readout
from qubit import Qubit
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
import stationF006
#from stationF006 import station
from copy import deepcopy
from manipulation_library import Ramsey

#%%
class Experiment:

    def __init__(self, name, qubits, awg, pulsar, **kw):

#        self.station = stationF006.initialize()

#        self.qubits_name = qubits_name

        self.awg = awg
#        self.digitizer =
        self.qubits = qubits

        self.channel_I = [qubit.microwave_gate['channel_I'] for qubit in qubits]
        self.channel_Q = [qubit.microwave_gate['channel_Q'] for qubit in qubits]
        self.channel_VP = [qubit.plunger_gate['channel_VP'] for qubit in qubits]
        self.channel_PM = [qubit.microwave_gate['channel_PM'] for qubit in qubits]

        self.qubits_number = len(qubits)

#        self.awg = station.awg
#        self.digitizer = station.digitizer

        self.sequence = Sequence(name = name)

        self.sweep_point1 = 0
        self.sweep_point2 = 0

        self.sweep_loop1 = {}
        self.sweep_loop2 = {}

        self.sweep_loop = {
            'loop1': self.sweep_loop1,
            'loop2': self.sweep_loop2,
            }

        self.sweep_loop3 = {}

        self.sweep_set = {}         ## {''}
        self.sweep_type = 'NoSweep'

        self.manip_elem = None
        self.sequence_cfg = []      ## [segment1, segment2, segment3]
#        for element in self.sequence_cfg.keys():
#        sweep_dimension = 0
#        segment_number = 0


#        for segment in self.sequence_cfg:
#            for step in segment.keys():
#                for parameter in segment[step].keys():
#                    if type(segment[step][parameter]) == str:
#                        ss = {}
#                        ss['segment_number'] = segment_number
##                        ss['segment'] = segment
#                        ss['step'] = step
#                        ss['parameter'] = parameter
#                        print(segment[step][parameter])
##                        ss['loop_number'] = segment[step][parameter][5]
#                        self.sweep_set[segment[step][parameter]] = ss       ## sweep_set: {'loop1_para1':   'loop1_para2'}
#                        sweep_dimension+=1
#            segment_number+=1


        """
        i = 0
        for step in self.init_cfg.keys():
            for parameter in self.init_cfg[step].keys():
                if type(self.init_cfg[step][parameter]) == list:
                    self.sweep_set[i] = self.init_cfg[step][parameter]
                    i+=1
        for step in self.manip_cfg.keys():
            for parameter in self.manip_cfg[step].keys():
                if type(self.manip_cfg[step][parameter]) == list:
                    self.sweep_set[i] = self.manip_cfg[step][parameter]
                    i+=1
        for step in self.read_cfg.keys():
            for parameter in self.read_cfg[step].keys():
                if type(self.read_cfg[step][parameter]) == list:
                    self.sweep_set[i] = self.read_cfg[step][parameter]
                    i+=1
        """

        self.initialze_segment = []

        self.readout_segment = []

        self.element = {}           ##  will be used in    pulsar.program_awg(myseq, self.elements)
         ##  a dic with {'init': ele, 'manip_1': ele, 'manip_2': ele, 'manip_3': ele, 'readout': ele,......}
        self.elts = []
#        self.initialize_element = None
#        self.readout_element = None
#        self.manipulation_element = {}

        self.pulsar = pulsar

        self.channel = {}          ## unsure if it is necessary yet

        self.experiment = {}        ## {'Ramsey': , 'Rabi': ,}  it is a list of manipulation element for different experiment

        self.sweep_matrix = np.array([])



    def set_sweep(self,):
        sweep_dimension = 0
        segment_number = 0

        for segment in self.sequence_cfg:
            for step in segment.keys():
                for parameter in segment[step].keys():
                    if type(segment[step][parameter]) == str:
                        ss = {}
                        sweep_parameter = segment[step][parameter]
                        ss['segment_number'] = segment_number
#                        ss['segment'] = segment
                        ss['step'] = step
                        ss['parameter'] = parameter
#                        print(segment[step][parameter])
#                        ss['loop_number'] = segment[step][parameter][5]
                        self.sweep_set[sweep_parameter] = ss       ## sweep_set: {'loop1_para1':   'loop1_para2'}
                        sweep_dimension+=1
            segment_number+=1


        if len(self.sweep_loop1) != 0:
            if len(self.sweep_loop2) != 0:
                self.sweep_type = '2D'
            else:
                self.sweep_type = '1D'

        self.sweep_loop = {
            'loop1': self.sweep_loop1,
            'loop2': self.sweep_loop2,
            }

        return True

    def initialize_element(self, name, amplitudes = []):

#        print(amplitudes[0])

        initialize = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'init1'
            initialize.add(SquarePulse(name='init', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='init%d'%(i+1),refpulse = refpulse, refpoint = 'start')

        return initialize



    def readout_element(self, name, amplitudes = []):

        readout = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'read1'
            readout.add(SquarePulse(name='read', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='read%d'%(i+1), refpulse = refpulse, refpoint = 'start')

        return readout

    def manipulation_element(self, name, time = 0, amplitudes = [], **kw):

        manip = deepcopy(self.manip_elem)

#        manip = Ramsey(name=name, pulsar = self.pulsar)

        waiting_time = kw.pop('waiting_time', None)
        duration_time = kw.pop('duration_time', None)
        frequency = kw.pop('frequency', None)
        print(name)
        manipulation = manip(name = name, qubits = self.qubits, pulsar = self.pulsar, waiting_time = waiting_time, duration_time = duration_time, frequency = frequency)

        manipulation.make_circuit()

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'manip1'
            start = -500e-9 if i ==0 else 0
            manipulation.add(SquarePulse(name='manip%d'%(i+1), channel=self.channel_VP[i], amplitude=amplitudes[i], length=time),
                           name='manip%d'%(i+1), refpulse = refpulse, refpoint = 'start', start = start)

        return manipulation





    def make_initialize_segment(self,):
        
        
        
        return True















    def load_sequence(self,):
#        elts = list(self.element.values())
        self.awg.delete_all_waveforms_from_list()
        elts = self.elts
        sequence = self.sequence
        self.pulsar.program_awg(sequence, *elts)       ## elts should be list(self.element.values)

        return True


    def update_element(self,):
        return True



    def run_experiment(self,):

        self.awg.write('SOUR1:ROSC:SOUR INT')

        self.awg.ch3_state.set(1)
        self.awg.force_trigger()

        self.awg.run()

        return True



    def run_all(self, name):                    ## the idea is to make a shortcut to run all the experiment one time

        self.initialize()

#        self.readout()

#        self.manipulation(name = manipulate)

        self._generate_sequence(name)

        self.awg.delete_all_waveforms_from_list()

        self.awg.stop()

        self.load_sequence(name)

        self.run_experiment(name)

        return True



    def Sweep_1D(self, parameter, start, stop, points):

        sweep_array = np.linspace(start, stop, points)


#        sweep_array.tolist()
#        self.sweep_matrix = sweep_array
        self.sweep_matrix = [{parameter: value} for value in sweep_array]
#        for i in range(points):
#            self.sweep_matrix.append({parameter: sweep_array[i]})

        return self.sweep_matrix

    def Sweep_2D(self, parameter1, start1, stop1, points1, parameter2, start2, stop2, points2):

        sweep_array1 = np.linspace(start1, stop1, points1)

        sweep_array2 = np.linspace(start2, stop2, points2)


        self.sweep_matrix = [[{parameter1: value1, parameter2:value2} for value1 in sweep_array1] for value2 in sweep_array2]

#        for i in range(points1):
#            for j in range(points2):
#                self.sweep_matrix[i][j] = {parameter1: sweep_array1[i], parameter2: sweep_array2[j]}


        return self.sweep_matrix
