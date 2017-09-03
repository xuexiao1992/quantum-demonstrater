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
from sequencer import Sequencer
#from initialize import Initialize
#from readout import Readout
from qubit import Qubit
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
import stationF006
#from stationF006 import station
from copy import deepcopy
from manipulation_library import Ramsey
import time
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
        self.manip2_elem = None
        self.manipulation_elements = {
#                'Rabi': None,
#                'Ramsey': None
                }
        
        self.sequence_cfg = []      ## [segment1, segment2, segment3]
        self.sequence_cfg_type = {}
        
        self.sequence2_cfg = []
        self.sequence2_cfg_type = {}
        
        self.seq_cfg = []
        self.seq_cfg_type = []
        
        self.sequencer = []
        
        self.dimension_1 = 1
        self.dimension_2 = 1


        self.initialize_segment = {
#                'step1': None,
                }

        self.readout_segment = {
#                'step1': None,
#                'step2': None,
#                'step3': None,
                }

        self.manipulation_segment = {
#                'step1': None,
                }
        
        self.initialize2_segment = {
#                'step1': None
                }
        
        self.readout2_segment = {
#                'step1': None,
#                'step2': None,
#                'step3': None,
                }

        self.manipulation2_segment = {
#                'step1': None,
                }
        
        self.segment = {
                'init': self.initialize_segment,
                'manip': self.manipulation_segment,
                'read': self.readout_segment,
                'init2': self.initialize2_segment,
                'manip2': self.manipulation2_segment,
                'read2': self.readout2_segment,
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
        self.initialize2_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }
        self.manipulation2_repetition = {
                'step1': 1,
                }
        self.readout2_repetition = {
                'step1': 1,
                'step2': 1,
                'step3': 1,
                }

        self.repetition = {
                'init': self.initialize_repetition,
                'manip': None,
                'read': self.readout_repetition,
                'init2': self.initialize2_repetition,
                'manip2': None,
                'read2': self.readout2_repetition,
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
    
    def make_sequencers(self,):
        
        sequencer_amount = len(self.seq_cfg)
        
        for i in range(sequencer_amount):
            self.sequencer[i] = Sequencer(qubits=self.qubits, awg=self.awg, awg2=self.awg2, pulsar=self.pulsar,
                          vsg=self.vsg, vsg2=self.vsg2,digitizer=self.digitizer)
            
            self.sequencer[i].sequence_cfg = self.sequence_cfg
            self.sequencer[i].sequence_cfg_type = self.sequence_cfg_type
        
        
        return self.seq_cfg, self.seq_cfg_type
    
    
    
    def make_all_segment_list(self,):
        
        i = 0

        for segment_type in self.sequence_cfg_type:

            if segment_type.startswith('init') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[j].make_initialize_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('manip') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[j].make_manipulation_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('read') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[j].make_readout_segment_list(segment_num = i, name = segment_type)

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
        seq_num = 0
        for idx_i in range(self.dimension_1):
            rep_idx = 10*idx_j+idx_i
            self.generate_unit_sequence(seq_num, idx_j = idx_j, idx_i = idx_i)
            

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
    
    
    def add_marker_into_first_readout(self, awg, element = None):
        
        first_read_step = self.segment['read']['step1']

        if type(first_read_step) is list:
            if type(first_read_step[0]) is list:
                first_read = first_read_step[0][0]
            else:
                first_read = first_read_step[0]
        else:
            first_read = first_read_step
        
        tvals, wfs = first_read.normalized_waveforms()
        name = '1st' + first_read.name
        print('readoutname:', name)
#        first_read.add(SquarePulse(name = 'read_marker', channel = self.digitier_readout_marker, amplitude=2, length=1e-6),
#                       name='read_marker', refpulse = 'read1', refpoint = 'start')
#        self.elts.append(first_read)
        
        channel = self.digitier_readout_marker
        wfs[channel] += 1
        i = (int(channel[2])-1)%4+1
        for ch in ['ch%d'%i, 'ch%d_marker1'%i, 'ch%d_marker2'%i]:
            if len(wfs[ch]) != 1000:
                wfs[ch] = np.full((1000,),1)
        wfs[channel] = np.full((1000,),1)
        
        awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
                                  m2 = wfs['ch%d_marker2'%i], wfmname = name+'_ch%d'%i)
        element_no = len(self.segment['init']) + len(self.segment['manip']) + 1 + 1
        element_no = 7
        awg.set_sqel_waveform(waveform_name = name+'_ch%d'%i, channel = i,
                              element_no = element_no)
        return first_read

    def add_marker_into_first_readout_v2(self, awg, element = None):
        
        return True
    def add_new_element_to_awg_list(self, awg, element):

        name = element.name

        tvals, wfs = element.normalized_waveforms()

        for i in range(1,5):
            awg.send_waveform_to_list(w = wfs['ch%d'%i], m1 = wfs['ch%d_marker1'%i],
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
    
    def restore_previous_sequence(self, awg, filename = 'setup_0_.AWG'):
        
        awg = awg
        
        awg.load_awg_file(filename = filename)
        
#        self.awg.write('AWGCONTROL:SRESTORE "{}"'.format(filename))
        
        return True
    
    def save_sequence(self, awg, filename,disk):
        
        awg.write('AWGCONTROL:SSAVE "{}","{}"'.format(filename,'C:'))
        
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
        trigger_element.add(SquarePulse(name = 'TRG1', channel = 'ch4_marker2', amplitude=2, length=1376e-9),
                            name='trigger1',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)
        
        trigger_element.add(SquarePulse(name = 'TRG_digi', channel = 'ch2_marker1', amplitude=2, length=800e-9),
                            name='TRG_digi',refpulse = 'trigger2', refpoint = 'start', start = 200e-9)
        trigger_element.add(SquarePulse(name = 'SIG_digi', channel = 'ch8', amplitude=0.5, length=800e-9),
                            name='SIG_digi',refpulse = 'TRG_digi', refpoint = 'start', start = 0)
        
        
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
        time.sleep(0.2)
        self.awg2.delete_all_waveforms_from_list()
        time.sleep(5)
        self.set_trigger()
        time.sleep(1)
        elts = self.elts
        sequence = self.sequence
        self.pulsar.program_awgs(sequence, *elts, AWGs = ['awg','awg2'],)       ## elts should be list(self.element.values)
        
        time.sleep(5)
        
        self.add_marker_into_first_readout(self.awg2)
#        self.awg2.trigger_level(0.5)
        self.awg2.set_sqel_trigger_wait(element_no = 1, state = 1)
        time.sleep(1)
        last_element_num = self.awg2.sequence_length()
        self.awg.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)
        time.sleep(0.2)
        self.awg2.set_sqel_goto_target_index(element_no = last_element_num, goto_to_index_no = 2)

        return True


    

    def run_experiment(self,):
        
        print('run experiment')

#        self.awg2.write('SOUR1:ROSC:SOUR EXT')
#        self.awg.write('SOUR1:ROSC:SOUR INT')
#        self.awg.clock_source('EXT')
#        self.awg.ch3_state.set(1)
        self.awg.all_channels_on()
#        self.awg.force_trigger()
        self.awg2.all_channels_on()
        
        self.vsg.status('On')
        self.vsg2.status('On')
        time.sleep(0.5)
#        self.awg2.force_trigger()
#        self.awg.run()
#        self.awg2.run()
        self.pulsar.start()

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
