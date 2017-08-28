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

    def __init__(self, name, qubits, awg, awg2, pulsar, **kw):

        self.awg = awg
        self.awg2 = awg2
        self.qubits = qubits
        
        self.vsg = kw.pop('vsg',None)
        self.vsg2 = kw.pop('vsg2',None)
        self.digitizer = kw.pop('digitizer', None)
        
        self.awg_file = None

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
#                'step2': None,
#                'step3': None,
                }

        self.manipulation_segment = {
                'step1': None,
                }
        
        self.segment = {
                'init': self.initialize_segment,
                'manip': self.manipulation_segment,
                'read': self.readout_segment,
                }
        
        self.initialize_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }
        self.manipulation_repetition = {
                'step1': 1,
                }
        self.readout_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }

        self.repetition = {
                'init': self.initialize_repetition,
                'manip': None,
                'read': self.readout_repetition,
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
                        ss['step'] = step
                        ss['parameter'] = parameter
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
        
        self.make_all_segment_list()

        return True

    def _initialize_element(self, name, amplitudes = [],**kw):

        initialize = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'init1'
            initialize.add(SquarePulse(name='init', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='init%d'%(i+1),refpulse = refpulse, refpoint = 'start')
            
        initialize.add(SquarePulse(name='init_c1m2', channel='ch2_marker2', amplitude=2, length=1e-6),
                                   name='init%d_c1m2'%(i+1),refpulse = 'init1', refpoint = 'start')
        initialize.add(SquarePulse(name='init_c5m2', channel='ch5_marker2', amplitude=2, length=1e-6),
                                   name='init%d_c5m2'%(i+1),refpulse = 'init1', refpoint = 'start')

        return initialize

    def _readout_element(self, name, amplitudes = [],**kw):

        readout = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'read1'
            readout.add(SquarePulse(name='read', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='read%d'%(i+1), refpulse = refpulse, refpoint = 'start')
        
        """
        for trigger digitizer
        """
        readout.add(SquarePulse(name='read_trigger1', channel='ch2_marker1', amplitude=2, length=1e-6),
                                name='read%d_trigger1'%(i+1),refpulse = 'read1', refpoint = 'start', start = 0)
        readout.add(SquarePulse(name='read_trigger2', channel='ch6_marker1', amplitude=2, length=1e-6),
                                name='read%d_trigger2'%(i+1),refpulse = 'read1', refpoint = 'start', start = 0)
        
        """
        to make all elements equal length in different AWGs
        """
        readout.add(SquarePulse(name='read_c1m2', channel='ch2_marker2', amplitude=2, length=1e-6),
                                name='read%d_c1m2'%(i+1),refpulse = 'read1', refpoint = 'start')
        readout.add(SquarePulse(name='read_c5m2', channel='ch5_marker2', amplitude=2, length=1e-6),
                                name='read%d_c5m2'%(i+1),refpulse = 'read1', refpoint = 'start')

        return readout

    def _manipulation_element(self, name, time = 0, amplitudes = [], **kw):

        manip = deepcopy(self.manip_elem)

#        waiting_time = kw.pop('waiting_time', None)
#        duration_time = kw.pop('duration_time', None)
#        frequency = kw.pop('frequency', None)
#        power = kw.pop('power', None)
        parameter1 = kw.pop('parameter1', None)
        parameter2 = kw.pop('parameter2', None)
        print(name)
#        manip = Ramsey()
        manipulation = manip(name = name, qubits = self.qubits, pulsar = self.pulsar, 
                             parameter1 = parameter1, parameter2 = parameter2,)
#                             waiting_time = waiting_time, duration_time = duration_time,
#                             frequency = frequency, power = power)

        manipulation.make_circuit()
        
        VP_start_point = -manip.VP_before
        VP_end_point = manip.VP_after

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'manip1'
            start = VP_start_point if i ==0 else 0
            manipulation.add(SquarePulse(name='manip%d'%(i+1), channel=self.channel_VP[i], amplitude=amplitudes[i], length=time),
                           name='manip%d'%(i+1), refpulse = refpulse, refpoint = 'start', start = start)
            
        manipulation.add(SquarePulse(name='manip_c1m2', channel='ch2_marker2', amplitude=0.1, length=time),
                           name='manip%d_c1m2'%(i+1),refpulse = 'manip1', refpoint = 'start')
        manipulation.add(SquarePulse(name='manip_c5m2', channel='ch5_marker2', amplitude=2, length=time),
                           name='manip%d_c5m2'%(i+1),refpulse = 'manip1', refpoint = 'start')

        return manipulation

    def make_element(self, name, segment, time = 0, amplitudes = [], **kw):
        
        if segment == 'init':
            element = self._initialize_element(name, amplitudes = amplitudes,)
        elif segment == 'manip':
#            waiting_time = kw.pop('waiting_time',0)
            parameter1 = kw.pop('parameter1', 0)
            parameter2 = kw.pop('parameter2', 0)
            element = self._manipulation_element(name, time = time, amplitudes = amplitudes,
                                                 parameter1 = parameter1, parameter2 = parameter2)
        elif segment == 'read':
            element = self._readout_element(name, amplitudes = amplitudes,)
        else:
            element = None
        
        return element


    def _is_in_loop(self,segment_num,step,parameter = ''):

        is_in_loop = False
        loop_num = 0

        for para in self.sweep_set:
            if segment_num == self.sweep_set[para]['segment_number'] and step == self.sweep_set[para]['step']:
                if self.sequence_cfg_type[segment_num] == 'manip':
                    P = parameter
                    if P == self.sweep_set[para]['parameter']:
                        is_in_loop = True
                        loop_num = para[4]
                else:
                    for i in range(self.qubits_number):
                        P = parameter
                        if P == self.sweep_set[para]['parameter']:
                            is_in_loop = True
                            loop_num = para[4]

        return is_in_loop, loop_num
    
    def make_segment_step(self, segment_num, step_num, name):           ## each step consist 1 element is no sweep, 1D/2D list if is 1D/2D sweep
        
        s = step_num
        seg = self.sequence_cfg_type[segment_num]                       ## seg is string like 'init', 'manip', 'read'
        
        step = self.sequence_cfg[segment_num]['step%d'%s]           ## s is step number in one segment
        is_in_loop = []
        loop_num = []
        for k in range(self.qubits_number):
            m, n = self._is_in_loop(segment_num, 'step%d'%s,'voltage_%d'%(k+1))
            is_in_loop.append(m)
            loop_num.append(n)
        
        for parameter in ['time', 'waiting_time', 'duration_time', 'IQ_amplitude', 'parameter1', 'parameter2']:
#            if parameter not in:
#                continue
            m, n = self._is_in_loop(segment_num, 'step%d'%s,parameter)
            is_in_loop.append(m)
            loop_num.append(n)
        
        dimension_1 = self.dimension_1 if '1' in loop_num else 1
        dimension_2 = self.dimension_2 if '2' in loop_num else 1
        
        
        
        if True not in is_in_loop:
                
                amplitudes = [step['voltage_%d'%(q+1)] for q in range(self.qubits_number)]

                element = self.make_element(name = name+'step%d'%s, segment = seg, amplitudes=amplitudes,)
                """
                for trigger, not used
                if segment_num == 0 and step_num == 1:
                    element.add(SquarePulse(name = 'trigger', channel = 'ch4_marker2', amplitude = 2, length = 100e-9),
                                name = 'trigger',)
                """
                segment = element
                
                repetition = int(step['time']/(1e-6))
                
        elif True in is_in_loop:
                
            segment = []
            
            repetition = []
            
            for j in range(dimension_2):
                    
                segment.append([])
                
                repetition.append([])
                
                self._update_cfg(loop = 2, idx = j)
                    
                for i in range(dimension_1):
                    
                    self._update_cfg(loop = 1, idx = i)
                        
                    amplitudes = [step['voltage_%d'%(q+1)] for q in range(self.qubits_number)]
                    
                    waiting_time = step.pop('waiting_time', None)
                    parameter1 = step.pop('parameter1', None)
                    parameter2 = step.pop('parameter2', None)
                    element = self.make_element(name = name+'step%d_%d_%d'%(s,j,i), segment = seg, time = step['time'], amplitudes=amplitudes,
                                                waiting_time = waiting_time, parameter1 = parameter1, parameter2 = parameter2)
                    """
                    for trigger, not used
                    if segment_num == 0 and step_num == 1:
                        element.add(SquarePulse(name = 'trigger', channel = 'ch4_marker2', amplitude = 2, length = 100e-9),
                                    name = 'trigger',)
                    """
                    segment[j].append(element)
                    rep = 1 if seg is 'manip' else int(step['time']/(1e-6))
                    repetition[j].append(rep)
        
        if seg is 'manip':
            print('dimension2', dimension_2)
            print('dimension1', dimension_1)
            print('loop_num',loop_num)
            print('repetition', repetition)
        return segment, repetition


    def make_initialize_segment_list(self, segment_num, name = 'initialize', qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.initialize_segment['step%d'%(s+1)], self.initialize_repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
            
        return True

    def make_readout_segment_list(self, segment_num, name = 'readout', qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.readout_segment['step%d'%(s+1)], self.readout_repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
            
        return True
    
    def make_manipulation_segment_list(self, segment_num, name = 'manipulation', qubits_name = None,):

        for s in range(len(self.sequence_cfg[segment_num])):

           self.manipulation_segment['step%d'%(s+1)], self.manipulation_repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
#           self.readout_repetition['step%d'%(s+1)] = 1
        return True
    
    def make_all_segment_list(self,):
        
        i = 0

        for segment_type in self.sequence_cfg_type:

            if segment_type == 'init':
                self.make_initialize_segment_list(segment_num = i,)

            elif segment_type == 'manip':
                self.make_manipulation_segment_list(segment_num = i,)

            elif segment_type == 'read':
                self.make_readout_segment_list(segment_num = i, name = 'readout')

            i+=1

        return True
    
    
    """
    this function below organizes segments. e.g. self.initialize_segment = {'step1': element1, 'step2': 1D/2D list, 'step3': element3...}
    """
    
    def first_segment_into_sequence_and_elts(self, segment, repetition, rep_idx = 0,idx_j = 0, idx_i = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
#        print('rep_idx',rep_idx)
        for step in segment:
            print('step',step)
            
            if type(segment[step]) is not list:             ## smarter way for this
                element = segment[step]
                wfname = element.name
                name = wfname + '_%d_%d'%(j,i)
                repe = repetition[step]
                if i == 0:
                    self.elts.append(element)
            else:
                jj = 0 if len(segment[step]) == 1 else j
                ii = 0 if len(segment[step][0]) == 1 else i
                element = segment[step][jj][ii]
                wfname=element.name
                name = wfname + '_%d_%d'%(j,i)
                
                repe = repetition[step][jj][ii]
                if ii != 0 or i==0:                                                         ## !!!!!!!!! here is a potential BUG!!!!!!
                    self.elts.append(element)
            
            self.sequence.append(name = name, wfname = wfname, trigger_wait = False, repetitions = repe)
        
        return True
    
    def update_segment_into_sequence_and_elts(self, segment, repetition, rep_idx = 0,idx_j = 0, idx_i = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
        unit_seq_length=0
        for seg_type in self.sequence_cfg_type:
            unit_seq_length += len(self.segment[seg_type])
        if segment == self.initialize_segment:
            former_element = 0
        elif segment == self.manipulation_segment:
            former_element =  len(self.initialize_segment)
        elif segment == self.readout_segment:
            former_element =  len(self.initialize_segment) + len(self.manipulation_segment)
        
        
        for step in segment:                                ## step = 'step1'... is a string
            
            if type(segment[step]) is list:
                jj = 0 if len(segment[step]) == 1 else j
                ii = 0 if len(segment[step][0]) == 1 else i
                element = segment[step][jj][ii]
                wfname=element.name
                name = wfname + '_%d_%d'%(j,i)
                repe = repetition[step][jj][ii]
                if jj != 0:
                    wfname_to_delete = segment[step][jj-1][ii].name
                    element_no = i*unit_seq_length + former_element + int(step[-1])+1
                    print('element_no',element_no)
                    if i == 0:
                        self.add_new_element_to_awg_list(element = element)
                    self.add_new_waveform_to_sequence(wfname = wfname, element_no = element_no, repetitions = repe)
                    self.delete_element_from_awg_list(wfname = wfname_to_delete)
            
        return True
    
    
    def add_segment(self, segment, repetition, rep_idx = 0, idx_j = 0, idx_i = 0):
        
        j = idx_j
        
        if j == 0:
            self.first_segment_into_sequence_and_elts(segment, repetition, idx_j = idx_j, idx_i = idx_i)
        else:
            self.update_segment_into_sequence_and_elts(segment, repetition, idx_j = idx_j, idx_i = idx_i)
        
        return True
    
    
    def generate_unit_sequence(self, rep_idx = 0, idx_i = 0, idx_j = 0):          # rep_idx = 10*i+j
        
        i = 0           ## not used in this version
        
        for segment_type in self.sequence_cfg_type:

            if segment_type == 'init':
                
                segment = self.initialize_segment
                
                repetition = self.initialize_repetition
                
            elif segment_type == 'manip':
                
                segment = self.manipulation_segment
                
                repetition = self.manipulation_repetition
                
            elif segment_type == 'read':
                
                segment = self.readout_segment
                
                repetition = self.readout_repetition
                
            self.add_segment(segment = segment, repetition = repetition, idx_j = idx_j, idx_i = idx_i)
            i+=1

        return True
 
    
    def generate_1D_sequence(self, idx_j = 0):
        
        for idx_i in range(self.dimension_1):
            rep_idx = 10*idx_j+idx_i
            self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i)

        if idx_j == 0:
            self.load_sequence()
        
        return self.sequence
    

    def run_1D_sweep(self, idx_j = 0):
        
        self.generate_1D_sweep_sequence( idx_j = idx_j)
        
        self.run_experiment()
        
        return True
    
    def run_2D_sweep(self, ):
        
        for j in range(self.dimension_2):
            self.run_1D_sweep(idx_j = j)
        
        return True
    
    
    

    def add_new_element_to_awg_list(self, element):

        name = element.name

        tvals, wfs = element.normalized_waveforms()

        for i in range(1,5):
            self.awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                            m2 = wfs['ch%d_marker2'%i], wfmname = name+'_ch%d'%i)

        return True

    def add_new_waveform_to_sequence(self, wfname, element_no, repetitions):

        for i in range(1,5):
            self.awg.set_sqel_waveform(waveform_name = wfname+'_ch%d'%i, channel = i,
                                        element_no = element_no)
        
        self.awg.set_sqel_loopcnt(loopcount = repetitions, element_no = element_no)
        
        return True
    
    def delete_element_from_awg_list(self, wfname):
        
        for i in range(1,5):
            self.awg.write('WLISt:WAVeform:DELete "{}"'.format(wfname+'_ch%d'%i))
        
        return True
    
    def store_sequence(self, filename,):
        
        self.awg.send_awg_file(filename = filename, awg_file = self.awg_file)
        
        return True
    
    def restore_previous_sequence(self, filename = 'setup_0_.AWG'):
        
        self.awg.load_awg_file(filename = filename)
        
#        self.awg.write('AWGCONTROL:SRESTORE "{}"'.format(filename))
        
        return True
    
    def save_sequence(self,filename,disk):
        
        self.awg.write('AWGCONTROL:SSAVE "{}","{}"'.format(filename,'C:'))
        
        return True
#

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



    def set_trigger(self,):
        
        trigger_element = Element('trigger', self.pulsar)

        trigger_element.add(SquarePulse(name = 'TRG2', channel = 'ch8_marker2', amplitude=2, length=300e-9),
                            name='trigger2',)
        trigger_element.add(SquarePulse(name = 'TRG1', channel = 'ch4_marker2', amplitude=2, length=1.43e-6),
                            name='trigger1',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)
        
        extra_element = Element('extra', self.pulsar)
        extra_element.add(SquarePulse(name = 'EXT2', channel = 'ch8_marker2', amplitude=2, length=5e-9),
                            name='extra2',)
        extra_element.add(SquarePulse(name = 'EXT1', channel = 'ch4_marker2', amplitude=2, length=2e-6),
                            name='extra1',)

        self.elts.insert(0,trigger_element)
#        self.elts.append(extra_element)
        self.sequence.insert_element(name = 'trigger', wfname = 'trigger', pos = 0)
#        self.sequence.append(name ='extra', wfname = 'extra', trigger_wait = False)
        return True

    def load_sequence(self,):
        
        print('load sequence')
#        elts = list(self.element.values())
        self.awg.delete_all_waveforms_from_list()
        self.awg2.delete_all_waveforms_from_list()
        self.set_trigger()
        elts = self.elts
        sequence = self.sequence
        self.pulsar.program_awgs(sequence, *elts, AWGs = ['awg','awg2'],)       ## elts should be list(self.element.values)
        
        
        self.awg2.trigger_level(0.5)
        self.awg2.set_sqel_trigger_wait(element_no = 1, state = 1)
        last_element_num = self.awg2.sequence_length()
        self.awg.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)
        self.awg2.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)

        return True




    def run_experiment(self,):
        
        print('run experiment')

        self.awg2.write('SOUR1:ROSC:SOUR EXT')
        self.awg.write('SOUR1:ROSC:SOUR INT')
#        self.awg.clock_source('EXT')
#        self.awg.ch3_state.set(1)
        self.awg.all_channels_on()
        self.awg.force_trigger()
        self.awg2.all_channels_on()
#        self.awg2.force_trigger()
#        self.awg.run()
#        self.awg2.run()
#        self.pulsar.start()

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




        return self.sweep_matrix
