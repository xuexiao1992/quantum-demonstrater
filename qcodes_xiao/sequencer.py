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
class Sequencer:

    def __init__(self, name, qubits, awg, awg2, pulsar, **kw):
        
        self.name = name

        self.awg = awg
        self.awg2 = awg2
        self.qubits = qubits
        
        self.vsg = kw.pop('vsg',None)
        self.vsg2 = kw.pop('vsg2',None)
        self.digitizer = kw.pop('digitizer', None)
        
        self.dig = None
        
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
        
        self.X_parameter = None
        self.Y_parameter = None
        self.X_parameter_type = None
        self.Y_parameter_type = None
        self.X_sweep_array = np.array([])
        self.Y_sweep_array = np.array([])

        self.manip_elem = None

        self.manipulation_elements = {
#                'Rabi': None,
#                'Ramsey': None
                }
        
        self.sequence_cfg = []      ## [segment1, segment2, segment3]
        self.sequence_cfg_type = {}
        
        self.dimension_1 = 1
        self.dimension_2 = 1


        self.segment = {
                'init': {},
                'manip': {},
                'read': {},
                'init2': {},
                'manip2': {},
                'read2': {},
                }

        self.repetition = {
                'init': {},
                'manip': None,
                'read': {},
                'init2': {},
                'manip2': None,
                'read2': {},
                }
        
        self.element = {}           ##  will be used in    pulsar.program_awg(myseq, self.elements)
         ##  a dic with {'init': ele, 'manip_1': ele, 'manip_2': ele, 'manip_3': ele, 'readout': ele,......}
        self.elts = []

        self.pulsar = pulsar

        self.channel = {}          ## unsure if it is necessary yet

        self.experiment = {}        ## {'Ramsey': , 'Rabi': ,}  it is a list of manipulation element for different experiment

        self.sweep_matrix = np.array([])
        
        self.digitizer_trigger_channel = 'ch5_marker1'
        self.digitier_readout_marker = 'ch6_marker1'
        self.occupied_channel1 = 'ch2_marker2'
        self.occupied_channel2 = 'ch5_marker2'
 
    def add_manip_elem(self, name, manip_elem):
        
        self.manipulation_elements[name] = manip_elem
        
        return True
    
    def set_sweep(self,):
        
        sweep_dimension = 0
        segment_number = 0
        for segment in self.sequence_cfg:
            for step in segment.keys():
                for parameter in segment[step].keys():
                    if type(segment[step][parameter]) == str and segment[step][parameter].startswith('loop'):
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
        
#        self.make_all_segment_list()

        return True

    def _initialize_element(self, name, amplitudes = [],**kw):

        initialize = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'init1'
            initialize.add(SquarePulse(name='init', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='init%d'%(i+1),refpulse = refpulse, refpoint = 'start')
            
        initialize.add(SquarePulse(name='init_c1m2', channel=self.occupied_channel1, amplitude=2, length=1e-6),
                                   name='init%d_c1m2'%(i+1),refpulse = 'init1', refpoint = 'start')
        initialize.add(SquarePulse(name='init_c5m2', channel=self.occupied_channel2, amplitude=2, length=1e-6),
                                   name='init%d_c5m2'%(i+1),refpulse = 'init1', refpoint = 'start')

        return initialize

    def _readout_element(self, name, amplitudes = [], trigger_digitizer = False, **kw):

        readout = Element(name = name, pulsar = self.pulsar)

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'read1'
            readout.add(SquarePulse(name='read', channel=self.channel_VP[i], amplitude=amplitudes[i], length=1e-6),
                           name='read%d'%(i+1), refpulse = refpulse, refpoint = 'start')
        
        """
        for trigger digitizer
        """
        if trigger_digitizer:
            readout.add(SquarePulse(name='read_trigger1', channel='ch2_marker1', amplitude=2, length=1e-6),
                                    name='read%d_trigger1'%(i+1),refpulse = 'read1', refpoint = 'start', start = 0)
            readout.add(SquarePulse(name='read_trigger2', channel=self.digitizer_trigger_channel, amplitude=2, length=1e-6),
                                    name='read%d_trigger2'%(i+1),refpulse = 'read1', refpoint = 'start', start = 0)
        
        """
        to make all elements equal length in different AWGs
        """
        readout.add(SquarePulse(name='read_c1m2', channel=self.occupied_channel1, amplitude=2, length=1e-6),
                                name='read%d_c1m2'%(i+1),refpulse = 'read1', refpoint = 'start')
        readout.add(SquarePulse(name='read_c5m2', channel=self.occupied_channel2, amplitude=2, length=1e-6),
                                name='read%d_c5m2'%(i+1),refpulse = 'read1', refpoint = 'start')
        readout.add(SquarePulse(name='read_trigtest', channel='ch8', amplitude=0.3, length=1e-6),
                                name='read_trigtest',refpulse = 'read1', refpoint = 'start')
        return readout

    def _manipulation_element(self, name, time, amplitudes = [], **kw):

#        waiting_time = kw.pop('waiting_time', None)
#        duration_time = kw.pop('duration_time', None)
#        frequency = kw.pop('frequency', None)
#        power = kw.pop('power', None)
        print('manip time:', time)
        parameter1 = kw.get('parameter1', None)
        parameter2 = kw.get('parameter2', None)
        manip_elem = kw.get('manip_elem', Element(name = name, pulsar = self.pulsar))
        print(name)
        
        
#        manip = deepcopy(self.manip_elem)
        
        if manip_elem not in self.manipulation_elements:
            raise NameError('Manipulation Element [%s] not in Experiment.'%manip_elem)
#        
        manip = deepcopy(self.manipulation_elements[manip_elem])
        
        
#        manip = Ramsey()
        manipulation = manip(name = name, qubits = self.qubits, pulsar = self.pulsar, **kw)
#                             parameter1 = parameter1, parameter2 = parameter2,)
#                             waiting_time = waiting_time, duration_time = duration_time,
#                             frequency = frequency, power = power)

        manipulation.make_circuit(**kw)
        
        VP_start_point = -manip.VP_before
#        VP_end_point = manip.VP_after
        wfs, tvals = manipulation.normalized_waveforms()
        max_length = max([len(tvals[ch]) for ch in tvals])/1e9

        for i in range(len(self.qubits)):
            refpulse = None if i ==0 else 'manip1'
            start = VP_start_point if i ==0 else 0
            
            time = max(time, max_length-VP_start_point)
            
            manipulation.add(SquarePulse(name='manip%d'%(i+1), channel=self.channel_VP[i], amplitude=amplitudes[i], length=time),
                           name='manip%d'%(i+1), refpulse = refpulse, refpoint = 'start', start = start)
            
        manipulation.add(SquarePulse(name='manip_c1m2', channel=self.occupied_channel1, amplitude=0.1, length=time),
                           name='manip%d_c1m2'%(i+1),refpulse = 'manip1', refpoint = 'start')
        manipulation.add(SquarePulse(name='manip_c5m2', channel=self.occupied_channel2, amplitude=2, length=time),
                           name='manip%d_c5m2'%(i+1),refpulse = 'manip1', refpoint = 'start')
        

        manipulation.add(SquarePulse(name='manip_c1m2', channel='ch8_marker1', amplitude=2, length=time),
                           name='mmanip%d_c1m2'%(i+1),refpulse = 'manip1', refpoint = 'start')
        manipulation.add(SquarePulse(name='manip_c5m2', channel='ch8_marker2', amplitude=2, length=time),
                           name='mmanip%d_c5m2'%(i+1),refpulse = 'manip1', refpoint = 'start')

        return manipulation

    def make_element(self, name, segment, time=0, amplitudes = [], **kw):
        
        if segment[:4] == 'init':
            element = self._initialize_element(name, amplitudes = amplitudes,)
        elif segment[:5] == 'manip':
            element = self._manipulation_element(name, time = time, amplitudes = amplitudes, **kw)
        elif segment[:4] == 'read':
            trigger = True if name.find('step2')>=0 else False
            element = self._readout_element(name, amplitudes = amplitudes, trigger_digitizer = trigger)
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
        seg = self.sequence_cfg_type[segment_num]                       ## seg is string like 'init1', 'manip1', 'read2'
        
        step = self.sequence_cfg[segment_num]['step%d'%s]           ## s is step number in one segment
        is_in_loop = []
        loop_num = []
        """
        for k in range(self.qubits_number):
            m, n = self._is_in_loop(segment_num, 'step%d'%s,'voltage_%d'%(k+1))
            is_in_loop.append(m)
            loop_num.append(n)
        """
        for parameter in step:
#            if parameter not in:
#                continue
            m, n = self._is_in_loop(segment_num, 'step%d'%s ,parameter)
            is_in_loop.append(m)
            loop_num.append(n)
        
        dimension_1 = self.dimension_1 if '1' in loop_num else 1
        dimension_2 = self.dimension_2 if '2' in loop_num else 1
        
        
        
        if True not in is_in_loop:
                
                amplitudes = [step['voltage_%d'%(q+1)] for q in range(self.qubits_number)]
                
                specific_parameters = deepcopy(step)
                specific_parameters.pop('time')
                specific_parameters.pop('voltage_1')
                specific_parameters.pop('voltage_2')
#                parameter1 = step.get('parameter1', None)
#                parameter2 = step.get('parameter2', None)
#                manip_elem = step.get('manip_elem', None)

                element = self.make_element(name = name+'step%d'%s, segment = seg, time = step['time'], amplitudes=amplitudes, **specific_parameters)
#                                            parameter1 = parameter1, parameter2 = parameter2, manip_elem = manip_elem)
                """
                for trigger, not used
                if segment_num == 0 and step_num == 1:
                    element.add(SquarePulse(name = 'trigger', channel = 'ch4_marker2', amplitude = 2, length = 100e-9),
                                name = 'trigger',)
                """
                segment = element
                
#                repetition = int(step['time']/(1e-6))
                repetition = 1 if seg.startswith('manip') else int(step['time']/(1e-6))
                
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
                    
                    specific_parameters = deepcopy(step)
                    specific_parameters.pop('time')
                    specific_parameters.pop('voltage_1')
                    specific_parameters.pop('voltage_2')
                    
                    element = self.make_element(name = name+'step%d_%d_%d'%(s,j,i), segment = seg, time = step['time'], amplitudes=amplitudes, **specific_parameters)
#                                                waiting_time = waiting_time, parameter1 = parameter1, parameter2 = parameter2, manip_elem = manip_elem)
                    """
                    for trigger, not used
                    if segment_num == 0 and step_num == 1:
                        element.add(SquarePulse(name = 'trigger', channel = 'ch4_marker2', amplitude = 2, length = 100e-9),
                                    name = 'trigger',)
                    """
                    segment[j].append(element)
                    rep = 1 if seg.startswith('manip') else int(step['time']/(1e-6))
                    repetition[j].append(rep)
        
        return segment, repetition


    def make_initialize_segment_list(self, segment_num, name = 'initialize', qubits_name = None,):
        
        segment = {}
        repetition = {}

        for s in range(len(self.sequence_cfg[segment_num])):

           segment['step%d'%(s+1)], repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
            
        return segment, repetition

    def make_readout_segment_list(self, segment_num, name = 'readout', qubits_name = None,):
        
        segment = {}
        repetition = {}

        for s in range(len(self.sequence_cfg[segment_num])):

           segment['step%d'%(s+1)], repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
            
        return segment, repetition
    
    def make_manipulation_segment_list(self, segment_num, name = 'manipulation', qubits_name = None,):
        
        segment = {}
        repetition = {}

        for s in range(len(self.sequence_cfg[segment_num])):

           segment['step%d'%(s+1)], repetition['step%d'%(s+1)] = self.make_segment_step(segment_num = segment_num, step_num = (s+1), name = name)
           
        return segment, repetition
    
    """
    this function below organizes segments. e.g. self.initialize_segment = {'step1': element1, 'step2': 1D/2D list, 'step3': element3...}
    """
    
    def first_segment_into_sequence_and_elts(self, segment, repetition, rep_idx = 0,idx_j = 0, idx_i = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
#        print('rep_idx',rep_idx)
        for step in segment:
            print('step',step, 'idx_i', idx_i)
            
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
    
    """
    problems below
    """
    
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
    
    def add_compensation(self, idx_j = 0, idx_i = 0):
        unit_length = 0
        for segment in self.sequence_cfg:
            unit_length += len(segment) 
        amplitude = [0,0]
        
        for i in range(unit_length):
            wfname = self.sequence.elements[-(i+1)]['wfname']
            for elt in self.elts:
                if elt.name == wfname:
                    element = elt
                    break
            tvals, wfs = element.ideal_waveforms()
            repe = self.sequence.elements[-(i+1)]['repetitions']
            amplitude[0] += np.sum(wfs[self.channel_VP[0]])*repe
            amplitude[1] += np.sum(wfs[self.channel_VP[1]])*repe
            
        comp_amp = [-amplitude[k]/3000000 for k in range(2)]
        
        if np.max(np.abs(comp_amp)) >= 0.5:
            raise ValueError('amp too large')
        
        compensation_element = Element('compensation_%d'%idx_i, self.pulsar)

        compensation_element.add(SquarePulse(name = 'COMPEN1', channel = self.channel_VP[0], amplitude=comp_amp[0], length=1e-6),
                            name='compensation1',)
        compensation_element.add(SquarePulse(name = 'COMPEN2', channel = self.channel_VP[1], amplitude=comp_amp[1], length=1e-6),
                            name='compensation2',refpulse = 'compensation1', refpoint = 'start', start = 0)
        
        compensation_element.add(SquarePulse(name='comp_c1m2', channel=self.occupied_channel1, amplitude=2, length=1e-6),
                                           name='comp%d_c1m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        compensation_element.add(SquarePulse(name='comp_c5m2', channel=self.occupied_channel2, amplitude=2, length=1e-6),
                                           name='comp%d_c5m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        
        self.elts.append(compensation_element)
        self.sequence.append(name = 'compensation_%d'%idx_i, wfname = 'compensation_%d'%idx_i, trigger_wait = False, repetitions = 3000)
        return True
    
    def generate_unit_sequence(self, seq_num = 0, rep_idx = 0, idx_i = 0, idx_j = 0):          # rep_idx = 10*i+j
        
        i = 0           ## not used in this version
        
        for segment_type in self.sequence_cfg_type:

            if segment_type.startswith('init'):   #[:4] == 'init':
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('manip'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('read'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            self.add_segment(segment = segment, repetition = repetition, idx_j = idx_j, idx_i = idx_i)
            i+=1
            
        self.add_compensation(idx_j = idx_j, idx_i = idx_i)

        return True
 
    
    def generate_1D_sequence(self, idx_j = 0):

        for idx_i in range(self.dimension_1):
#            rep_idx = 10*idx_j+idx_i
            self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i)
        
        return self.sequence
    

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

    



