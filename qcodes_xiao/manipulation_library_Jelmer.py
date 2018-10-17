# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:21:45 2018

@author: jmboter
"""

import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from qcodes.instrument.base import Instrument
#from experiment import Experiment
from manipulation import Manipulation
import stationF006
from copy import deepcopy

class ChargeNoiseBob_Jelmer(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', 1.2)
        self.waiting_time = kw.pop('waiting_time', 100e-9)
        self.detuning_time = kw.pop('detuning_time', 60e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 62)
        self.phase_2 = kw.pop('phase_2', 10)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0275)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.02)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_cphase = kw.pop('decoupled_cphase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')
        self.init_state = kw.pop('init_state')
        self.sigma1 = kw.pop('sigma1', 0)
        self.sigma2 = kw.pop('sigma2', 0)

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        self.init_state = kw.pop('init_state', self.init_state)
        self.sigma1 = kw.pop('sigma1', self.sigma1)
        self.sigma2 = kw.pop('sigma2', self.sigma2)
        return self

    def make_circuit(self, **kw):
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        init_state = kw.pop('init_state', self.init_state)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        if add_dephase:
            sigma1 = kw.pop('sigma1', self.sigma1)
            sigma2 = kw.pop('sigma2', self.sigma2)
            noise1 = np.random.normal(0, sigma1)
            noise2 = np.random.normal(0, sigma2)
        
        te = 10e-9
        
        init_states = {'01+10': 0, '00+11': 1, '00+01+10+11': 2, '00+01': 3, '10+11': 4, '00+10': 5, '01+11': 6};
        init_state = init_states[init_state];
        
        # Prepare one of the Bell states or two Bell states combined.
        if init_state < 3:
            self.add_X(name='Q1_X1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        
            self.add_X(name='Q2_X1', qubit = qubit_2, refgate = 'Q1_X1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
            self.add_Z(name='Q1_Z1', qubit = qubit_1, degree = 90)
            self.add_Z(name='Q2_Z1', qubit = qubit_2, degree = 90)
            
            # CPhase and single qubit gates for the entangled Bell states.
            if init_state < 2:
                if decoupled_cphase:
                    self.add_CPhase(name = 'CP1_1', refgate = 'Q1_X1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
                    
                    self.add_X(name='Q1_X_CP1', qubit = qubit_1, refgate = 'CP1_1',
                           amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
                    self.add_X(name='Q2_X_CP1', qubit = qubit_2, refgate = 'Q1_X_CP1', refpoint = 'start',
                               amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

                    self.add_CPhase(name = 'CP1', refgate = 'Q1_X_CP1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
                    
                else:
                    self.add_CPhase(name = 'CP1', refgate = 'Q1_X1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
                
                self.add_Z(name='Q1_Z_CP1', qubit = qubit_1, degree = phase_1)
                self.add_Z(name='Q2_Z_CP1', qubit = qubit_2, degree = phase_2)
                
                # 01+10
                if init_state == 0:
                    self.add_Z(name='Q1_Z_subspace1', qubit = qubit_1, degree = 180)
                    
                self.add_X(name='Q1_X2', qubit = qubit_1, refgate = 'CP1', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
            
                self.add_X(name='Q2_X2', qubit = qubit_2, refgate = 'Q1_X2', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
                self.add_Z(name='Q1_Z2', qubit = qubit_1, degree = 90)
                self.add_Z(name='Q2_Z2', qubit = qubit_2, degree = 90)
            
        # Prepare 00+01 or 10+11
        if init_state == 3 or init_state == 4:
            self.add_X(name='Q1_X1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
            
            self.add_Z(name='Q1_Z1', qubit = qubit_1, degree = 90)
            
            # Prepare 10+11
            if init_state == 4:
                self.add_X(name='Q2_X1', qubit = qubit_2, refgate = 'Q1_X1',
                           amplitude = amplitude, length = qubit_2.Pi_pulse_length,)
        
        # Prepare 00+10 or 01+11
        if init_state == 5 or init_state == 6:
            self.add_X(name='Q2_X1', qubit = qubit_2,
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
            
            self.add_Z(name='Q2_Z1', qubit = qubit_2, degree = 90)
            
            # Prepare 10+11
            if init_state == 6:
                self.add_X(name='Q1_X1', qubit = qubit_1, refgate = 'Q2_X1',
                           amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
        # WAIT
        self.add_CPhase(name = 'WAIT', refgate = 'Q1_X2', waiting_time = 0,
                        control_qubit = qubit_1, target_qubit = qubit_2,
                        amplitude_control = 0, amplitude_target = 0, 
                        length = waiting_time)
        
        # Add detuning during wait time for easy fitting
        self.add_Z(name='Q1_Zdet', qubit = qubit_1, degree = 360 * 0E6 * waiting_time)
        self.add_Z(name='Q2_Zdet', qubit = qubit_2, degree = 360 * 4E6 * waiting_time)
        
        # Add noise in software
        if add_dephase:
            self.add_Z(name='Z_noise_Q1', qubit = qubit_1, degree = 360 * noise1*waiting_time)
            self.add_Z(name='Z_noise_Q2', qubit = qubit_2, degree = 360 * noise2*waiting_time)
        
        # Reverse sequence to go back to initial state
        
        # Reverse one of the Bell states or two Bell states combined.
        if init_state < 3:
            # CPhase and single qubit gates for the entangled Bell states.
            if init_state < 2:
                self.add_Z(name='Q1_Z3', qubit = qubit_1, degree = 90)
                self.add_Z(name='Q2_Z3', qubit = qubit_2, degree = 90)
                
                self.add_X(name='Q1_X3', qubit = qubit_1, refgate = 'WAIT', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
            
                self.add_X(name='Q2_X3', qubit = qubit_2, refgate = 'Q1_X3', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                
                # 01+10
                if init_state == 0:
                    self.add_Z(name='Q1_Z_subspace2', qubit = qubit_1, degree = 180)
                
                if decoupled_cphase:
                    self.add_CPhase(name = 'CP2_1', refgate = 'Q1_X3', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
                    
                    self.add_X(name='Q1_X_CP2', qubit = qubit_1, refgate = 'CP2_1',
                           amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
                    self.add_X(name='Q2_X_CP2', qubit = qubit_2, refgate = 'Q1_X_CP2', refpoint = 'start',
                               amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

                    self.add_CPhase(name = 'CP2_2', refgate = 'Q1_X_CP2', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
                    
                else:
                    self.add_CPhase(name = 'CP2', refgate = 'Q1_X3', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
                
                self.add_Z(name='Q1_Z_CP2', qubit = qubit_1, degree = phase_1)
                self.add_Z(name='Q2_Z_CP2', qubit = qubit_2, degree = phase_2)
            
            self.add_Z(name='Q1_Z4', qubit = qubit_1, degree = 90)
            self.add_Z(name='Q2_Z4', qubit = qubit_2, degree = 90)
            
            self.add_X(name='Q1_X4', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
            
            self.add_X(name='Q2_X4', qubit = qubit_2, refgate = 'Q1_X4', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        
        # Reverse 00+01 or 10+11
        if init_state == 3 or init_state == 4:
            self.add_Z(name='Q1_Z2', qubit = qubit_1, degree = 90)
            
            self.add_X(name='Q1_X2', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
                        
            # Reverse 10+11
            if init_state == 4:
                self.add_X(name='Q2_X2', qubit = qubit_2, refgate = 'Q1_X2', refpoint = 'start',
                           amplitude = amplitude, length = qubit_2.Pi_pulse_length,)
        
        # Reverse 00+10 or 01+11
        if init_state == 5 or init_state == 6:
            self.add_Z(name='Q2_Z2', qubit = qubit_2, degree = 90)
            
            self.add_X(name='Q2_X2', qubit = qubit_2,
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
                        
            # Reverse 10+11
            if init_state == 6:
                self.add_X(name='Q1_X2', qubit = qubit_1, refgate = 'Q2_X2', refpoint = 'start',
                           amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
        return self

class ChargeNoiseBob_Jelmer2(Manipulation):
    
    def __init__(self, name, pulsar, **kw):

        super().__init__(name, pulsar, **kw)
        self.refphase = {}
        self.qubit = kw.pop('qubit', 'qubit_2')
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = None
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude', 1)
        self.waiting_time = kw.pop('waiting_time', 0)
        self.detuning_time = kw.pop('detuning_time', 80e-9)
        self.amplitude = kw.pop('amplitude', 1)
        self.phase_1 = kw.pop('phase_1', 0)
        self.phase_2 = kw.pop('phase_2', 0)
        self.amplitude_control = kw.pop('amplitude_control', 30*0.5*-0.0277)
        self.amplitude_target = kw.pop('amplitude_target', 30*0.5*0.00)
        self.DFS = kw.pop('DFS', False)
        self.add_dephase = kw.pop('add_dephase', False)
        self.decoupled_qubit = kw.pop('decoupled_qubit', 'qubit_1')
        self.decoupled_cphase = kw.pop('decoupled_cphase', False)
        self.sigma = kw.pop('sigma', 0)
        self.DD = kw.pop('DD', 'None')

    def __call__(self, **kw):
        self.name = kw.pop('name', self.name)
        self.qubits = kw.pop('qubits', None)
        if self.qubits is not None:
            self.qubits_name = [qubit.name for qubit in self.qubits]
            self.refphase = {qubit.name: 0 for qubit in self.qubits}
        self.pulsar = kw.pop('pulsar', None)
        self.off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        self.waiting_time = kw.pop('waiting_time', self.waiting_time)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.phase_1 = kw.pop('phase_1', self.phase_1)
        self.phase_2 = kw.pop('phase_2', self.phase_2)
        self.amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        self.amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
        self.detuning_time = kw.pop('detuning_time', self.detuning_time)
        self.DFS = kw.pop('DFS', self.DFS)
        self.add_dephase = kw.pop('add_dephase', self.add_dephase)
        self.decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        self.decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        self.sigma = kw.pop('sigma', self.sigma)
        self.DD = kw.pop('DD', self.DD)
        
        return self

    def make_circuit(self, **kw):
        decoupled_cphase = kw.pop('decoupled_cphase', self.decoupled_cphase)
        decoupled_qubit = kw.pop('decoupled_qubit', self.decoupled_qubit)
        add_dephase = kw.pop('add_dephase', self.add_dephase)
        DFS = kw.pop('DFS', self.DFS)
        detuning_time = kw.pop('detuning_time', self.detuning_time)
        waiting_time = kw.pop('waiting_time', self.waiting_time)
        amplitude = kw.get('amplitude', self.amplitude)
        off_resonance_amplitude = kw.pop('off_resonance_amplitude',self.off_resonance_amplitude)
        phase_1 = kw.pop('phase_1', self.phase_1)
        phase_2 = kw.pop('phase_2', self.phase_2)
        amplitude_control = kw.pop('amplitude_control', self.amplitude_control)
        amplitude_target = kw.pop('amplitude_target', self.amplitude_target)
#        qubit_name = kw.pop('qubit', self.qubit)
        DD = kw.pop('DD', self.DD)
        
        qubit_1 = Instrument.find_instrument('qubit_1')
        qubit_2 = Instrument.find_instrument('qubit_2')
        
        if add_dephase:
            sigma = kw.pop('sigma', self.sigma)
            noise = np.random.normal(0, sigma)
        
        te = 10e-9

        # Initial X rotations (Pi/2)
        self.add_X(name='Q1_X1', qubit = qubit_1,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='Q2_X1', qubit = qubit_2, refgate = 'Q1_X1', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        # CPhase (decoupled or normal)
        if decoupled_cphase:
            self.add_CPhase(name = 'CP1_1', refgate = 'Q1_X1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Q1_X_CP1', qubit = qubit_1, refgate = 'CP1_1', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Q2_X_CP1', qubit = qubit_2, refgate = 'Q1_X_CP1', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP1', refgate = 'Q1_X_CP1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
        
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP1', refgate = 'Q1_X1', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
        
        # Z rotations to correct for phases in CPhase plus Pi/2
        self.add_Z(name='Q1_Z_CP1', qubit = qubit_1, degree = phase_1+90)
        self.add_Z(name='Q2_Z_CP1', qubit = qubit_2, degree = phase_2+90)
            
        #self.add_Z(name='Z1i_Q1', qubit = qubit_1, degree = 90)
        #self.add_Z(name='Z1i_Q2', qubit = qubit_2, degree = 90)
        
        # Z rotation to prepare 01+10
        if DFS:
            self.add_Z(name='Q1_Z_DFS', qubit = qubit_1, degree = 180)
        
        # X and Z rotations to finish state preparation
        self.add_X(name='Q1_X2', qubit = qubit_1, refgate = 'CP1', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='Q2_X2', qubit = qubit_2, refgate = 'Q1_X2', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        self.add_Z(name='Q1_Z1', qubit = qubit_1, degree = 90)
                
        # WAIT
        if DD == 'None':
            self.add_CPhase(name = 'WAIT', refgate = 'Q1_X2', waiting_time = 0,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = 0, amplitude_target = 0, 
                            length = waiting_time)
        
        if DD == 'Hahn':
            self.add_CPhase(name = 'WAIT0', refgate = 'Q1_X2', waiting_time = 0,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = 0, amplitude_target = 0, 
                            length = waiting_time/2)
            
            self.add_X(name='Q1_X_Hahn', qubit = qubit_1, refgate = 'WAIT0', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
            self.add_X(name='Q2_X_Hahn', qubit = qubit_2, refgate = 'Q1_X_Hahn', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.Pi_pulse_length,)
            
            self.add_CPhase(name = 'WAIT', refgate = 'Q1_X_Hahn', waiting_time = 0,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = 0, amplitude_target = 0, 
                            length = waiting_time/2)
        
        # Add detuning during wait time for easy fitting
        self.add_Z(name='Q1_Zdet', qubit = qubit_1, degree = 360 * 0E6 * waiting_time)
        self.add_Z(name='Q2_Zdet', qubit = qubit_2, degree = 360 * 4E6 * waiting_time)
        
        # Add noise in software
        if add_dephase:
            self.add_Z(name='Z_noise_Q1', qubit = qubit_1, degree = 360 * noise * waiting_time)
            self.add_Z(name='Z_noise_Q2', qubit = qubit_2, degree = 360 * noise * waiting_time)
        
        # Reverse sequence to go back to initial state
        
        
        # Z and X rotations
        self.add_Z(name='Q1_Z2', qubit = qubit_1, degree = 90)
        
        self.add_X(name='Q1_X3', qubit = qubit_1, refgate = 'WAIT', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='Q2_X3', qubit = qubit_2, refgate = 'Q1_X3', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        # Z rotation for 01+10
        if DFS == 1:
            self.add_Z(name='Q1_Z_DFS2', qubit = qubit_1, degree = 180)
        
        # CPhase (decoupled or normal)
        if decoupled_cphase:
            self.add_CPhase(name = 'CP2_1', refgate = 'Q1_X3', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
            
            self.add_single_qubit_gate(name='Q1_X_CP2', qubit = qubit_1, refgate = 'CP2_1', waiting_time = te,
                                       amplitude = amplitude, length = qubit_1.Pi_pulse_length,)
        
            self.add_single_qubit_gate(name='Q2_X_CP2', qubit = qubit_2, refgate = 'Q1_X_CP2', refpoint = 'start',
                                       amplitude = amplitude, length = qubit_2.Pi_pulse_length,)

            self.add_CPhase(name = 'CP2', refgate = 'Q1_X_CP2', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time/2)
        
        if not decoupled_cphase:
            self.add_CPhase(name = 'CP2', refgate = 'Q1_X3', waiting_time = te,
                            control_qubit = qubit_1, target_qubit = qubit_2,
                            amplitude_control = amplitude_control, amplitude_target = amplitude_target, 
                            length = detuning_time)
        
        # Z rotations to correct for phases in CPhase plus Pi/2
        self.add_Z(name='Q1_Z_CP2', qubit = qubit_1, degree = phase_1+90)
        self.add_Z(name='Q2_Z_CP2', qubit = qubit_2, degree = phase_2+90)
        
        # Final X rotations (Pi/2)
        self.add_X(name='Q1_X4', qubit = qubit_1, refgate = 'CP2', waiting_time = te,
                   amplitude = amplitude, length = qubit_1.halfPi_pulse_length,)
        self.add_X(name='Q2_X4', qubit = qubit_2, refgate = 'Q1_X4', refpoint = 'start',
                   amplitude = amplitude, length = qubit_2.halfPi_pulse_length,)
        
        return self