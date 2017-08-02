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


        self.initialze_segment = []

        self.readout_segment = []

        self.element = {}           ##  will be used in    pulsar.program_awg(myseq, self.elements)
         ##  a dic with {'init': ele, 'manip_1': ele, 'manip_2': ele, 'manip_3': ele, 'readout': ele,......}
        self.elts = []

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

    def _is_in_loop(self,segment_num,step,parameter = ''):

        is_in_loop = False
        loop_num = 0


        for para in self.sweep_set:
            if segment_num == self.sweep_set[para]['segment_number'] and step == self.sweep_set[para]['step']:
                if self.sequence_cfg_type[segment_num] == 'manip':
                    is_in_loop = True
                    loop_num = para[4]
                else:
                    for i in range(self.qubits_number):
                        P = parameter
                        if P == self.sweep_set[para]['parameter']:
                            is_in_loop = True
                            loop_num = para[4]

        return is_in_loop, loop_num

    def make_initialize_segment(self, segment_num, name = 'initialize', rep_idx = 0, qubits_name = None,):

        for i in range(len(self.sequence_cfg[segment_num])):

            step = self.sequence_cfg[segment_num]['step%d'%(i+1)]           ## i is step number in one segment
            is_in_loop = []
            loop_num = []
            for k in range(self.qubits_number):
                m, n = self._is_in_loop(segment_num, 'step%d'%(i+1),'voltage_%d'%(k+1))
                is_in_loop.append(m)
                loop_num.append(n)

            idx_i = rep_idx//10
            idx_j = rep_idx%10

            if self.sweep_type == '2D':
                idx_i = rep_idx//10
                idx_j = rep_idx%10
            else:
                idx_i = rep_idx
                idx_j = 0

            if '1' in loop_num and '2' in loop_num:
                idx = idx_i
                outer = idx_j

            elif '1' in loop_num:
                idx = idx_i
                outer = idx_j

            elif '2' in loop_num:
                idx = idx_i
                outer = idx_j

            else:
                idx = 0
                outer = 0

            if is_in_loop or rep_idx == 0:
                if outer == 0:

                    print('generate')

                    amplitudes = [step['voltage_%d'%(i+1)] for i in range(self.qubits_number)]

                    initialize_element = self.initialize_element(name = name+'%d%d'%(idx,(i+1)), amplitudes=amplitudes,)

                    self.elts.append(initialize_element)

            if is_in_loop:

                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(idx,(i+1)),
                                     trigger_wait=False, repetitions= int(step['time']/(1e-6)))
            else:
                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(0,(i+1)),
                                     trigger_wait=False, repetitions= int(step['time']/(1e-6)))

        return True

    def make_manipulation_segment(self, segment_num, name = 'manipulation', rep_idx=0, qubits_name = [None, None]):


        for i in range(len(self.sequence_cfg[segment_num])):

            step = self.sequence_cfg[segment_num]['step%d'%(i+1)]

            is_in_loop, loop_num = self._is_in_loop(segment_num, 'step%d'%(i+1),)

            print('is_in_loop, loop_num',is_in_loop, loop_num)

            if self.sweep_type == '2D':
                idx_i = rep_idx//10
                idx_j = rep_idx%10
            else:
                idx_i = rep_idx
                idx_j = 0

            if loop_num == '1':
                idx = idx_i
                other = idx_j

            elif loop_num == '2':
                idx = idx_j
                other = idx_i

            else:
                idx = 0
                other = 0

            print('idx:', idx)
            print('other:', other)
            print('rep_idx:',rep_idx)

            if is_in_loop or rep_idx == 0:
                if other == 0:
                    amplitudes = [step['voltage_%d'%(i+1)] for i in range(self.qubits_number)]

                    time = step['time']

#                    waiting_time = step.pop('waiting_time', 0)

                    waiting_time = step['waiting_time']


                    manipulation_element = self.manipulation_element(name = name+'%d%d'%(idx,(i+1)),
                                                                     amplitudes = amplitudes, time = time,
                                                                     waiting_time = waiting_time,)

                    self.elts.append(manipulation_element)

            if is_in_loop:
                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(idx,(i+1)),
                                     trigger_wait=False, )
            else:
                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(0,(i+1)),
                                     trigger_wait=False, )

        return True

    def make_readout_segment(self, segment_num, name = 'readout', rep_idx = 1, qubits_name = None):           ## consists of Elzerman, PSB...

        for i in range(len(self.sequence_cfg[segment_num])):
            step = self.sequence_cfg[segment_num]['step%d'%(i+1)]

            is_in_loop, loop_num = self._is_in_loop(segment_num, 'step%d'%(i+1),'voltage')

            idx = rep_idx//10 if loop_num == 1 else rep_idx%10

            if is_in_loop or rep_idx == 0:
                amplitudes = [step['voltage_%d'%(i+1)] for i in range(self.qubits_number)]

                readout_element = self.readout_element(name = name+'%d%d'%(idx,(i+1)), amplitudes=amplitudes,)

                self.elts.append(readout_element)

                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(idx,(i+1)),
                                     trigger_wait=False, repetitions= int(step['time']/(1e-6)))

            else:
                self.sequence.append(name = name+'%d%d'%(rep_idx,(i+1)), wfname = name+'%d%d'%(0,(i+1)),
                                     trigger_wait=False, repetitions= int(step['time']/(1e-6)))

        return True

    def generate_sweep_matrix(self,):

        for i in range(len(self.sweep_loop1['para1'])):

            for j in range(len(self.sweep_loop2['para1'])):

                0

        return True



    def add_new_element_to_awg_list(self, element):

        name = element.name

        tvals, wfs = element.normalized_waveforms()

        for i in range(1,5):
            self.awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                            m2 = wfs['ch%d_marker2'%i], wfmname = name)

        return True

    def add_new_waveform_to_sequence(self, wfname, element_no):

        for i in range(1,5):
            self.awg.set_sqel_waveform(waveform_name = wfname+'_ch%d'%i, channel = i,
                                        element_no = element_no)

        return True

    def replace_manip_in_sequence(self,):

        return True

    def generate_unit_sequence(self, rep_idx = 0, idx_i = 0, idx_j = 0):          # rep_idx = 10*i+j

        i = 0

        for segment_type in self.sequence_cfg_type:

            if segment_type == 'init':
                self.make_initialize_segment(segment_num = i, rep_idx = rep_idx)

            elif segment_type == 'manip':
                self.make_manipulation_segment(segment_num = i, rep_idx = rep_idx)

            elif segment_type == 'read':
                self.make_readout_segment(segment_num = i, rep_idx = rep_idx)

            i+=1

        return True

    """
    this function is to be used......
    """
    def _loop_information(self, loop = 1):

        para_num = len(self.sweep_loop['loop%d'%loop])

        segment_number = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['segment_number'] for k in range(para_num)]
        step = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['step'] for k in range(para_num)]
        parameter = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['parameter'] for k in range(para_num)]

        return segment_number, step, parameter



    def _update_cfg(self, loop = 1, idx = 1):

        para_num = len(self.sweep_loop['loop%d'%loop])      ## number of parameter in one loop e.g. loop1: para1, para2,,,,para_num = 2
        i = idx

        segment_number = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['segment_number'] for k in range(para_num)]
        step = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['step'] for k in range(para_num)]
        parameter = [self.sweep_set['loop%d_para%d'%(loop,(k+1))]['parameter'] for k in range(para_num)]
#        print('parameter:' , parameter)

        for k in range(para_num):
            self.sequence_cfg[segment_number[k]][step[k]][parameter[k]] = self.sweep_loop['loop%d'%loop]['para%d'%(k+1)][i]

        return True

    def _1D_sweep(self,):

#        segment_number, step, parameter = self._loop_information(loop = 1)

        for i in range(len(self.sweep_loop1['para1'])):

            self._update_cfg(loop = 1, idx = i)

            self.generate_unit_sequence(rep_idx = i)

        return True

    def _2D_sweep(self,):

        for i in range(len(self.sweep_loop1['para1'])):
            self._update_cfg(loop = 1, idx = i)

            for j in range(len(self.sweep_loop2['para1'])):
                self._update_cfg(loop = 2, idx = j)

                self.generate_unit_sequence(rep_idx = 10*i+j, idx_i = i, idx_j = j)

        return True

    def _1D_sweep_new(self,):

        for i in range(len(self.sweep_loop1['para1'])):

            self._update_cfg(loop = 1, idx = i)

            self.generate_unit_sequence(rep_idx = i)

        self.load_sequence()

        return True


    def _2D_sweep_new(self,):

        for j in range(len(self.sweep_loop2['para1'])):

            self._update_cfg(loop = 2, idx = j)

            if j == 0:
                self._1D_sweep()
            else:
                self.add_new_element_to_awg_list()

        return True


    def generate_sequence(self,):

        if self.sweep_type == '1D':
            self._1D_sweep()
        elif self.sweep_type == '2D':
            self._2D_sweep()

        return True


    def _generate_sequence(self,):
        if len(self.sweep_loop1) == 0:
            self.generate_unit_sequence()

        elif len(self.sweep_loop1) != 0:
            for i in range(len(self.sweep_loop1['para1'])):
#                segment = [self.sweep_set['loop1_para%d'%(k+1)]['segment'] for k in range(len(self.sweep_loop1))]
                segment_number = [self.sweep_set['loop1_para%d'%(k+1)]['segment_number'] for k in range(len(self.sweep_loop1))]
                step = [self.sweep_set['loop1_para%d'%(k+1)]['step'] for k in range(len(self.sweep_loop1))]
                parameter = [self.sweep_set['loop1_para%d'%(k+1)]['parameter'] for k in range(len(self.sweep_loop1))]
                for k in range(len(segment_number)):
                    self.sequence_cfg[segment_number[k]][step[k]][parameter[k]] = self.sweep_loop1['para%d'%(k+1)][i]
#                    segment[step[k]][parameter[k]] = self.sweep_loop1['para%d'%(k+1)][i]

                if len(self.sweep_loop2) == 0:
                    self.generate_unit_sequence(rep_idx = i)

                elif len(self.sweep_loop2) != 0:
                    for j in range(len(self.sweep_loop2['para1'])):
#                        segment = [self.sweep_set['loop2_para%d'%(k+1)]['segment'] for k in range(len(self.sweep_loop2))]
                        segment_number = [self.sweep_set['loop2_para%d'%(k+1)]['segment_number'] for k in range(len(self.sweep_loop2))]
                        step = [self.sweep_set['loop2_para%d'%(k+1)]['step'] for k in range(len(self.sweep_loop2))]
                        parameter = [self.sweep_set['loop2_para%d'%(k+1)]['parameter'] for k in range(len(self.sweep_loop2))]

                        for k in range(len(segment_number)):
                            self.sequence_cfg[segment_number[k]][step[k]][parameter[k]] = self.sweep_loop2['para%d'%(k+1)][j]

                        self.generate_unit_sequence(rep_idx = 10*i+j)

        """
        for d in range(len(self.sweep_matrix)):

            self.sequence.append(name = 'Initialize_%d'%d, wfname = 'Initialize_%d'%d, trigger_wait=False)
            self.sequence.append(name = 'Manipulation_%d'%d, wfname = 'Manipulation_%d'%d, trigger_wait=False)
            self.sequence.append(name = 'Readout_%d'%d, wfname = 'Readout_%d'%d, trigger_wait=False)
        """
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
