# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:31:17 2017

@author: X.X
"""


import numpy as np
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.element import Element
from Gates import Single_Qubit_Gate#, Two_Qubit_Gate
from manipulation import Manipulation
from initialize import Initialize
from readout import Readout
from qubit import Qubit
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse


class Experiment:

    def __init__(self, name, qubits, qubits_name, awg, pulsar, **kw):

        self.qubits_name = qubits_name

        self.qubits = qubits

        self.qubits_number = len(qubits)

        self.sequence = Sequence(name = name)

        self.sweep_point1 = 0
        self.sweep_point2 = 0

        self.sweep_set1 = np.array([])
        self.sweep_set2 = np.array([])

        self.sweep_set = []

        self.sweep_type = '2D'

        self.sequence_cfg = []      ## [segment1, segment2, segment3]
"""
        self.init_cfg = {
                'step1' : {},
                'step2' : {},
                'setp3' : {},
                         }

        self.read_cfg = {
                'step1' : {},
                'step2' : {},
                'setp3' : {},
                }

        self.manip_cfg = {
                'step1' : {},
                'step2' : {},
                'setp3' : {},
                }
    """
        #for element in self.sequence_cfg.keys():
        sweep_dimension = 0
        segment_number = 0
        for segment in self.sequence_cfg:
            for step in segment.keys():
                for parameter in segment[step].keys():
                    if type(segment[step][parameter]) == list:
                        ss = {}
                        ss['segment_number'] = segment_number
                        ss['step'] = step
                        ss['parameter'] = parameter
                        ss['values'] = segment[step][parameter]
                        self.sweep_set.append(ss)
                        sweep_dimension+=1
            segment_number+=1

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

        self.awg = awg              ## list of AWG used in experiment

        self.pulsar = pulsar

        self.channel = {}          ## unsure if it is necessary yet

        self.experiment = {}        ## {'Ramsey': , 'Rabi': ,}  it is a list of manipulation element for different experiment

        self.sweep_matrix = np.array([])





    def initialize_element(self, name, step = 'step1'):

        initialize = Initialize(name = name)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'init1'
            initialize.add(SquarePulse(name='init', channel='ch1', amplitude=value['Qubit_%d'%i], length=value['time']),
                           name='init%d'%(i+1),refpulse = None)

        return initialize



    def readout_element(self, name):

        readout = Readout(name = name)

        readout.add(SquarePulse(name='square_load', channel='ch1', amplitude=0.05, length=10e-6), name='Load')

        return readout


    def manipulation_element(self, name):

        manipulation = Manipulation(name = name)

        return manipulation




    def make_initialize(self, name = None, qubits_name = None):

        if self.sweep_matrix == []:

            for i in range(len(self.init_cfg)):
                step = self.init_cfg['step%d'%i]
                self.initialize_segment.append(self.initialize_element(name = name,amplitude=step[''],length = ))

        else:
            for sweep_item in self.sweep_matrix:

                for i in range(len(self.init_cfg)):
                    self.initialize_segment.append(self.initialize_element(name = name))

                sweep_item['initialize'] = self.initialze_segment

                self.initialze_segment = []

        return True

    def make_manipulation(self, name, qubits_name = None):

#        manipulation = self.experiment['name']

        manipulation = Manipulation(name = name, qubits_name = qubits_name)
#        manipulation = self.experiment['name']
#        self.manipulation_element.append(manipulation)
        self.element['Manipulation_%d'%(len(self.element)-1)] = manipulation


        return True

    def make_readout(self, name = None, qubits_name = None):           ## consists of Elzerman, PSB...

        readout = Element('readout', pulsar = self.pulsar)
        readout.add(SquarePulse(name='square_empty', channel='ch1', amplitude=0.05, length=10e-6), name='Load')
        self.element['Readout'] = readout

        return True


    def generate_sequence(self, name):

        for i

"""
        for d in range(len(self.sweep_matrix)):

            self.sequence.append(name = 'Initialize_%d'%d, wfname = 'Initialize_%d'%d, trigger_wait=False)
            self.sequence.append(name = 'Manipulation_%d'%d, wfname = 'Manipulation_%d'%d, trigger_wait=False)
            self.sequence.append(name = 'Readout_%d'%d, wfname = 'Readout_%d'%d, trigger_wait=False)
"""
        return True

    def load_sequence(self, name):
        elts = list(self.element.values())
        self.pulsar.program_awg(self.sequence, *elts)       ## elts should be list(self.element.values)

        return True


    def update_element(self,):
        return True



    def run_experiment(self, name):

        self.awg.run()

        return True



    def run_all(self, name):                    ## the idea is to make a shortcut to run all the experiment one time

        self.initialize()

#        self.readout()

#        self.manipulation(name = manipulate)

        self.generate_sequence(name)

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
