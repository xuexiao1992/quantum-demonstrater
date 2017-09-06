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

from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray

import stationF006
#from stationF006 import station
from copy import deepcopy
from manipulation_library import Ramsey
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
import time
#%%
class Experiment:

    def __init__(self, name, label, qubits, awg, awg2, pulsar, **kw):
        self.name = name
        self.label = label

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

        self.sweep2_loop1 = {}
        self.sweep2_loop2 = {}

        self.sweep_set = {}         ## {''}
        self.sweep_type = 'NoSweep'
        
        self.dig = None
        
        self.Loop = None
        self.X_parameter = None
        self.Y_parameter = None
        
        self.formatter = HDF5FormatMetadata()
        self.data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
        self.data_location = time.strftime("%Y-%m-%d/%H-%M-%S") + name + label
        self.data_set = None


        self.manip_elem = []
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

    def add_manip_elem(self, name, manip_elem, seq_num):
        
        self.manipulation_elements[name] = manip_elem
#        self.manip_elem.append([])
        if seq_num > len(self.manip_elem):
            for i in range(seq_num - len(self.manip_elem)):
                self.manip_elem.append([])
        self.manip_elem[seq_num-1].append(manip_elem)
        print('33333333',self.manip_elem)
        return True
    
    def find_sequencer(self, name):
        
        for sequencer in self.sequencer:
            if name == sequencer.name:
                return sequencer
                break
        return 'sequencer not found'
        
    
    def add_measurement(self, measurement_name, manip_name, manip_elem, sequence_cfg, sequence_cfg_type):
        
        sequencer = Sequencer(name = measurement_name, qubits=self.qubits, awg=self.awg, awg2=self.awg2, pulsar=self.pulsar,
                              vsg=self.vsg, vsg2=self.vsg2,digitizer=self.digitizer)
        
        
        i = len(self.sequencer)
        self.sequencer.append(sequencer)
        
        for seg in range(len(sequence_cfg_type)):
            if sequence_cfg_type[seg].startswith('manip'):
                Name = manip_name.pop(0)
                Elem = manip_elem.pop(0)
                sequence_cfg[seg]['step1'].update({'manip_elem': Name})
                self.add_manip_elem(name = Name, manip_elem = Elem, seq_num = i+1)
                self.sequencer[i].manipulation_elements[Name] = Elem
        self.sequencer[i].sequence_cfg = sequence_cfg
        self.sequencer[i].sequence_cfg_type = sequence_cfg_type
        self.seq_cfg.append(sequence_cfg)
        self.seq_cfg_type.append(sequence_cfg_type)

#        self.sequencer[i].sweep_loop1 = self.sweep_loop1[i]
#        self.sequencer[i].sweep_loop2 = self.sweep_loop2[i]
#        self.sequencer[i].set_sweep()
        
        return self

    def __call__(self,):
        
        return self
    
    def add_X_parameter(self, measurement, parameter, sweep_array):
        
        sequencer = self.find_sequencer(measurement)
        
        if type(parameter) is StandardParameter:
            
            Sweep_Value = parameter[sweep_array[0]:sweep_array[-1]:(sweep_array[1]-sweep_array[0])]
    
            self.Loop = Loop(sweep_values = Sweep_Value).each(self.dig)
            
        elif type(parameter) is str:
            i = len(sequencer.sweep_loop1)+1
            para = 'para'+str(i)
            loop = 'loop1_'+para
            sequencer.sweep_loop1[para] = sweep_array
            for seg in range(len(sequencer.sequence_cfg_type)):
                if sequencer.sequence_cfg_type[seg].startswith('manip'):
                    sequencer.sequence_cfg[seg]['step1'].update({parameter: loop})
            
#        sequencer.set_sweep()
        
        return True
    
    def add_Y_parameter(self, measurement, parameter, sweep_array):
        
        sequencer = self.find_sequencer(measurement)
        
        if type(parameter) is StandardParameter:
            
            Sweep_Value = parameter[sweep_array[0]:sweep_array[-1]:(sweep_array[1]-sweep_array[0])]
            
            if self.Loop is None:
                self.Loop = Loop(sweep_values = Sweep_Value).each(self.dig)
            else:
                self.Loop = Loop(sweep_values = Sweep_Value).each(self.Loop)
            
        else:
            for para in sequencer.sweep_loop2:
                if len(sequencer.sweep_loop2[para]) == 0:
                    break
            parameter = 'loop2_'+para
            sequencer.sweep_loop2[para] = sweep_array
            
#        sequencer.set_sweep()
        
        return True
    
    def function(self, x):
        return True

    def set_sweep(self,):
        
        for seq in range(len(self.sequencer)):
            self.sequencer[seq].set_sweep()
            self.make_all_segment_list(seq)
        
        """
        loop function
        """
        if self.Loop is None:
            Count = StandardParameter(name = 'Count', set_cmd = self.function)
            Sweep_Count = Count[1:2:1]
            self.Loop = Loop(sweep_values = Sweep_Count).each(self.dig)
            
        return True
    
    def make_sequencers(self, **kw):
        
        sequencer_amount = len(self.seq_cfg)
        self.sequencer = []
        for i in range(sequencer_amount):
            name = 'Seq_%d'%(i+1)#self.seq_cfg[i]
            print('sequencer name',name)
            sequencer_i = Sequencer(name = name, qubits=self.qubits, awg=self.awg, awg2=self.awg2, pulsar=self.pulsar,
                          vsg=self.vsg, vsg2=self.vsg2,digitizer=self.digitizer)
            print('sequencer name 2nd check',name)
            self.sequencer.append(sequencer_i)
            print('sequencer list', self.sequencer)
            self.sequencer[i].sequence_cfg = self.seq_cfg[i]
            self.sequencer[i].sequence_cfg_type = self.seq_cfg_type[i]
            
            self.sequencer[i].sweep_loop1 = self.sweep_loop1[i]
            self.sequencer[i].sweep_loop2 = self.sweep_loop2[i]
            self.sequencer[i].set_sweep()
            print('sequencer type3', type(sequencer_i))
            
            self.sequencer[i].manip_elem = self.manip_elem[i]
            for manip in self.manip_elem[i]:
                self.sequencer[i].add_manip_elem(name = manip.name, manip_elem = manip)
                
            self.make_all_segment_list(seq_num = i)
            print('sequencer type4', type(sequencer_i))
        
        return self.sequencer
    
    
    
    def make_all_segment_list(self, seq_num):
        
        i = 0

        for segment_type in self.seq_cfg_type[seq_num]:
            print('segment_type', segment_type)

            if segment_type.startswith('init') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_initialize_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('manip') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_manipulation_segment_list(segment_num = i, name = segment_type)

            elif segment_type.startswith('read') and not self.segment[segment_type]:
                self.segment[segment_type], self.repetition[segment_type] = self.sequencer[seq_num].make_readout_segment_list(segment_num = i, name = segment_type)

            i+=1

        return True
    
    
    """
    this function below organizes segments. e.g. self.initialize_segment = {'step1': element1, 'step2': 1D/2D list, 'step3': element3...}
    """
    
    def first_segment_into_sequence_and_elts(self, segment, repetition, rep_idx = 0,idx_j = 0, idx_i = 0, seq_num = 0):         ## segment = self.initialize_segment
        
        j = idx_j
        
        i = idx_i
        
        sq = seq_num
#        print('rep_idx',rep_idx)
        for step in segment:
            print('step',step, 'idx_i', idx_i)
            
            if type(segment[step]) is not list:             ## smarter way for this
                element = segment[step]
                wfname = element.name
                name = wfname + '_%d_%d_%d'%(sq,j,i)
                repe = repetition[step]
                if i == 0:
                    self.elts.append(element)
            else:
                jj = 0 if len(segment[step]) == 1 else j
                ii = 0 if len(segment[step][0]) == 1 else i
                element = segment[step][jj][ii]
                wfname=element.name
                name = wfname + '_%d_%d_%d'%(sq,j,i)
                
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
    
    
    def add_segment(self, segment, repetition, rep_idx = 0, idx_j = 0, idx_i = 0, seq_num = 0):
        
        j = idx_j
        
        if j == 0:
            self.first_segment_into_sequence_and_elts(segment, repetition, idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
        else:
            self.update_segment_into_sequence_and_elts(segment, repetition, idx_j = idx_j, idx_i = idx_i)
        
        return True
    
    def add_compensation(self, idx_j = 0, idx_i = 0, seq_num = 0):
        unit_length = 0
        for segment in self.sequencer[seq_num].sequence_cfg:
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
        
        compensation_element = Element('compensation_%d_%d'%(seq_num, idx_i), self.pulsar)

        compensation_element.add(SquarePulse(name = 'COMPEN1', channel = self.channel_VP[0], amplitude=comp_amp[0], length=1e-6),
                            name='compensation1',)
        compensation_element.add(SquarePulse(name = 'COMPEN2', channel = self.channel_VP[1], amplitude=comp_amp[1], length=1e-6),
                            name='compensation2',refpulse = 'compensation1', refpoint = 'start', start = 0)
        
        compensation_element.add(SquarePulse(name='comp_c1m2', channel=self.occupied_channel1, amplitude=2, length=1e-6),
                                           name='comp%d_c1m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        compensation_element.add(SquarePulse(name='comp_c5m2', channel=self.occupied_channel2, amplitude=2, length=1e-6),
                                           name='comp%d_c5m2'%(i+1),refpulse = 'compensation1', refpoint = 'start')
        
        self.elts.append(compensation_element)
        self.sequence.append(name = 'compensation_%d_%d'%(seq_num, idx_i), 
                             wfname = 'compensation_%d_%d'%(seq_num, idx_i), 
                             trigger_wait = False, repetitions = 3000)
        return True

   
    def generate_unit_sequence(self, seq_num = 0, rep_idx = 0, idx_i = 0, idx_j = 0):          # rep_idx = 10*i+j
        
        i = 0           ## not used in this version
        
        for segment_type in self.sequencer[seq_num].sequence_cfg_type:

            if segment_type.startswith('init'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('manip'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            elif segment_type.startswith('read'):
                
                segment = self.segment[segment_type]
                
                repetition = self.repetition[segment_type]
                
            self.add_segment(segment = segment, repetition = repetition, 
                             idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
            i+=1
            
        self.add_compensation(idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)

        return True
 
    def generate_1D_sequence(self, idx_j = 0):
        seq_num = 0
        for sequencer in self.sequencer:
            D1 = sequencer.dimension_1
            for idx_i in range(D1):
                self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i, seq_num = seq_num)
            seq_num += 1
        if idx_j == 0:
            self.load_sequence()

        return self.sequence
    
    def generate_1D_sequence_v2(self, idx_j = 0):
       
        for sequencer in self.sequencer:
            sequencer.generate_1D_sequence(idx_j = idx_j)

        for idx_i in range(len(self.sequencer)):
            self.generate_unit_sequence(idx_j = idx_j, idx_i = idx_i)
        if idx_j == 0:
            self.load_sequence()
        
        return self.sequence
    
    
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
        print('channel', 'ch%d'%i)
        wfs_vp = wfs['ch%d'%int(channel[2])][0]
        for ch in ['ch%d'%i, 'ch%d_marker1'%i, 'ch%d_marker2'%i]:
            if len(wfs[ch]) != 1000:
                wfs[ch] = np.full((1000,),1)
        wfs['ch%d_marker1'%i] = np.full((1000,),1)
        wfs['ch%d_marker2'%i] = np.full((1000,),1)
        wfs['ch%d'%i] = np.full((1000,),wfs_vp)
        print('ana', wfs['ch%d'%i], 'm1', wfs['ch%d_marker1'%i], 'm2', wfs['ch%d_marker2'%i])
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
        
        time.sleep(5)
        
        if self.Loop is not None:
            self.data_set = self.Loop.run(location = self.data_location, loc_record = {'name': self.name, 'label': self.label}, io = self.data_IO,)
            
            self.awg.stop()
            self.awg2.stop()
            
            self.vsg.status('Off')
            self.vsg2.status('Off')

        return self.data_set

    def close(self,):
        self.awg.stop()
        self.awg2.stop()
        time.sleep(0.5)
        self.vsg.status('Off')
        self.vsg2.status('Off')
        self.awg.delete_all_waveforms_from_list()
        self.awg2.delete_all_waveforms_from_list()
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


