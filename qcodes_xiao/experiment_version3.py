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

        self.awg = awg
        self.qubits = qubits

        self.channel_I = [qubit.microwave_gate['channel_I'] for qubit in qubits]
        self.channel_Q = [qubit.microwave_gate['channel_Q'] for qubit in qubits]
        self.channel_VP = [qubit.plunger_gate['channel_VP'] for qubit in qubits]
        self.channel_PM = [qubit.microwave_gate['channel_PM'] for qubit in qubits]

        self.qubits_number = len(qubits)

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
        self.sequence_cfg_type = {}
        
        self.dimension_1 = 1
        self.dimension_2 = 1


        self.initialize_segment = {
                'step1': None,
                'step2': None,
                'step3': None,
                }

        self.readout_segment = {
                'step1': None,
                'step2': None,
                'step3': None,
                }

        self.manipulation_segment = {
                'step1': None,
                }
        
        self.segment = {
                'init': self.initialize_segment,
                'manip': self.manipulation_segment,
                'read': self.readout_segment,
                }
        
        self.initialzie_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }
        
        self.readout_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }

        self.repetition = {
                'init': self.initialize_repetitions,
                'manip': None,
                'read': self.readout_repetitions,
                }
        
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
#                        ss['loop_number'] = segment[step][parameter][5]
                        self.sweep_set[sweep_parameter] = ss       ## sweep_set: {'loop1_para1':   'loop1_para2'}
                        sweep_dimension+=1
            segment_number+=1


        if len(self.sweep_loop1) != 0:
            self.dimension_1 = len(self.sweep_loop1['para1'])
            if len(self.sweep_loop2) != 0:
                self.dimension_2 = len(self.sweep_loop2['para1'])
                self.sweep_type = '2D'
            else:
                self.sweep_type = '1D'
                
        self.sweep_loop = {
            'loop1': self.sweep_loop1,
            'loop2': self.sweep_loop2,
            }

        return True

    def _initialize_element(self, name, amplitudes = [],**kw):

#        print(amplitudes[0])

        initialize = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'init1'
            initialize.add(SquarePulse(name='init', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='init%d'%(i+1),refpulse = refpulse, refpoint = 'start')

        return initialize



    def _readout_element(self, name, amplitudes = [],**kw):

        readout = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'read1'
            readout.add(SquarePulse(name='read', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='read%d'%(i+1), refpulse = refpulse, refpoint = 'start')

        return readout

    def _manipulation_element(self, name, time = 0, amplitudes = [], **kw):

        manip = deepcopy(self.manip_elem)

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

    def make_element(self, name, segment, time = 0, amplitudes = [], **kw):
        
        if segment == 'initialize':
            element = self._initialize_element(name, amplitudes = [],)
        elif segment == 'manipulation':
            element = self._manipulation_element(name, time = 0, amplitudes = [],)
        elif segment == 'readout':
            element = self._readout_element(name, amplitudes = [],)
        else:
            element = None
        
        return element


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
    
    def make_segment_step(self, segment_num, step_num, name):
        
        s = step_num
        seg = self.sequence_cfg_type[segment_num]                       ## seg is string like 'init', 'manip', 'read'
        
        step = self.sequence_cfg[segment_num]['step%d'%(s+1)]           ## s is step number in one segment
        is_in_loop = []
        loop_num = []
        for k in range(self.qubits_number):
            m, n = self._is_in_loop(segment_num, 'step%d'%(s+1),'voltage_%d'%(k+1))
            is_in_loop.append(m)
            loop_num.append(n)
            
        m, n = self._is_in_loop(segment_num, 'step%d'%(s+1),'time')
        is_in_loop.append(m)
        loop_num.append(n)
            
        dimension_1 = len(self.sweep_loop1['para1']) if '1' in loop_num else 1
        dimension_2 = len(self.sweep_loop2['para1']) if '2' in loop_num else 1
        
        if not is_in_loop:
                
                amplitudes = [step['voltage_%d'%(q+1)] for q in range(self.qubits_number)]

                element = self.make_element(name = name+'step%d'%(s+1), segment = seg, amplitudes=amplitudes,)
                
#                self.segment[seg]['step%d'%(s+1)] = element
                
                segment = element
                
#                self.repetitions[seg]['step%d'%(s+1)] = int(step['time']/(1e-6))
                
                repetition = int(step['time']/(1e-6))
                
        elif is_in_loop:
                
#            self.segment[seg]['step%d'%(s+1)] = []
            
            segment = []
            
            repetition = []
            
            for j in range(dimension_2):
                    
#                self.segment[seg]['step%d'%(s+1)].append([])
                    
                segment.append([])
                
                repetition.append([])
                
                self._update_cfg(loop = 2, idx = j)
                    
                for i in range(dimension_1):
                    
                    self._update_cfg(loop = 1, idx = i)
                        
                    amplitudes = [step['voltage_%d'%(q+1)] for q in range(self.qubits_number)]
                        
                    element = self.make_element(name = name+'step%d_%d%d'%((s+1),j,i), segment = seg, amplitudes=amplitudes,)
                
#                    self.initialize_segment['step%d'%(s+1)][j][i] = initialize_element
#                    self.segment[seg]['step%d'%(s+1)][j].append(element)
                        
                    segment[j].append(element)
                    
#                    self.repetitions[seg]['step%d'%(s+1)][j].append(int(step['time']/(1e-6)))
                    
                    repetition[j].append(int(step['time']/(1e-6)))
        
        
        return segment, repetition


    def make_initialize_segment(self, segment_num, name = 'initialize', rep_idx = 0, qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.initialize_segment['step%d'%s], self.initialize_repetition['step%d'%s] = self.make_segment_step(segment_num = segment_num, step = s, name = name)
            
        return True

    def make_readout_segment(self, segment_num, name = 'readout', rep_idx = 0, qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.readout_segment['step%d'%s], self.readout_repetition['step%d'%s] = self.make_segment_step(segment_num = segment_num, step = s, name = name)
            
        return True
    
    def make_manipulation_segment(self, segment_num, name = 'manipulation', rep_idx = 0, qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.manipulation_segment['step%d'%s] = self.make_segment_step(segment_num = segment_num, step = s, name = name)[0]
            
        return True
    
    
    def make_1D_sweep_sequence(self, idx_j = 0):
        
        j = idx_j
        
        for i in range(self.dimension_1):
            
            for segment_type in self.sequence_cfg_type:
            
                if segment_type is 'init':
                
                    for step in self.initialize_segment:
                        
                        element = self.initialize_segment[step] if '1' else self.initialize_segment[step][j][i]
                        
                        wfname = element.name
                        
                        name = wfname if '1' else wfname + '_%d_%d'%(j,i)
                        
                        repetitions = self.initialzie_repetition[step] if '1' else self.initialzie_repetition[step][j][i]
                        
                        self.sequence.append(name = name, wfname = wfname, trigger_wait = False, repetitions = repetitions)
                        self.elts.append(element)
                
                
#                elif segment_type is 'read':
#                    self.sequence.append(name = , wfname = , trigger_wait = False, repetitions = )
#                    self.elts.append()
#                elif segment_type is 'manip':
#                    self.sequence.append(name = , wfname = , trigger_wait = False, repetitions = )
#                    self.elts.append()
#        
        
        return self.sequence, self.elts
    
    def update_1D_sweep_sequence(self,):
        
        for i in range(self.dimension_1):
            for segment_type in self.sequence_cfg_type:
                if segment_type is 'init':
                    
        
        return
    
    
    
    
    def run_1D_sweep(self,):
        
        self.make_1D_sweep_sequence()
        self.load_sequence()
        self.run_experiment
        
        return True
    
    def run_2D_sweep(self,)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    def make_elements(self,):
        
        i = 0

        for segment_type in self.sequence_cfg_type:

            if segment_type == 'init':
                self.make_initialize_segment(segment_num = i,)

            elif segment_type == 'manip':
                self.make_manipulation_segment(segment_num = i,)

            elif segment_type == 'read':
                self.make_readout_segment(segment_num = i,)

            i+=1
        
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
