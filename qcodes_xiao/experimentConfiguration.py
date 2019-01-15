# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:58:47 2018

@author: TUD278306
"""

#%% Function definition

def set_step(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

    return step

def set_manip(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    parameter1 = kw.pop('parameter1', 0)
    parameter2 = kw.pop('parameter2', 0)
    manip = kw.pop('manip_elem', 0)

    step = {'time' : time}
    step.update(kw)

    step['manip_elem'] = manip
    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

        step['parameter1'] = parameter1
        step['parameter2'] = parameter2

    return step

def save_object(obj, obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(obj_name = None):
    filename = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\experiment_objs\\{}.pkl'.format(obj_name)
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset

def Func_Gaussian(x, a, x0, ):
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def Counts(x):
    return True

#%% Step definitions

init_cfg = {
#        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.004, LP_factor*30*0.5*-0.001]),
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.000, LP_factor*30*0.5*0.000]),
        'step2' : set_step(time = 4e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.005]),
#        'step2' : set_step(time = 5e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.006]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.0096, LP_factor*30*0.5*-0.000]),
        'step4' : set_step(time = 5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step5' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.0012, LP_factor*30*0.5*-0.0002]),
        'step6' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.0005]),
#        'step6' : set_step(time = 0.5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.002]),  # from Tom
        }

manip_cfg = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.010,LP_factor*30*0.5*0.020],)
#        'step1' : set_manip(time = 2e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.002,LP_factor*30*0.5*0.016],)
#        'step1' : set_manip(time = 3e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.015,LP_factor*30*0.5*0.060],)
        }


read0_cfg = {
        'step1' : set_step(time = 0.40e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.003, LP_factor*30*0.5*-0.0005]),
#        'step1' : set_step(time = 0.02e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.00, LP_factor*30*0.5*0.025]),
#        'step1' : set_step(time = 2e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.010, LP_factor*30*0.5*-0.020]),
        }


read_cfg = {
#        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.000]),
        'step1' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
#        'step3' : set_step(time = 0.888e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 4e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        }

init2_cfg = {
#        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.004, LP_factor*30*0.5*-0.005]),
        'step1' : set_step(time = 2.5e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.0012, LP_factor*30*0.5*-0.0002]),
        'step3' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0.002, LP_factor*30*0.5*-0.0005]),
        }

manip2_cfg = {
        'step1' : set_manip(time = 1.0e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.010, LP_factor*30*0.5*0.020],)
#        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*0.002,LP_factor*30*0.5*0.016],)
#        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [T_factor*30*0.5*-0.015,LP_factor*30*0.5*0.060],)
        }


read2_cfg = {
        'step1' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
#        'step3' : set_step(time = 0.888e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 4e-3, qubits = qubits, voltages = [T_factor*30*0.5*0, LP_factor*30*0.5*0]),
        }

#read3_cfg = {
#        'step1' : set_step(time = 0.262e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step3' : set_step(time = 0.95e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
#        'step4' : set_step(time = 0.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*0.00]),
#        }

readBill_cfg = {
        'step1' : set_step(time = 0.263e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 2.45e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step4' : set_step(time = 0.5e-3, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.000]),
        }

#%%         Seq for T1 Q2


T_shift = 30*0.5*-0.004
LP_shift = 30*0.5*0.00


init_cfg_T1 = {
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.000, LP_shift + 30*0.5*-0.000]),
        'step2' : set_step(time = 4e-3, qubits = qubits, voltages = [T_shift + 30*0.5*-0.004, LP_shift + 30*0.5*-0.005]),
#        'step3' : set_step(time = 0.01e-3, qubits = qubits, voltages = [T_factor*30*0.5*-0.0096, LP_factor*30*0.5*0]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [T_shift + 30*0.5*-0.00787, LP_shift + 30*0.5*0.000]),
        'step4' : set_step(time = 5e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.000, LP_shift + 30*0.5*0.000]),
        'step5' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.0012, LP_shift + 30*0.5*-0.0002]),
        'step6' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0.003, LP_shift + 30*0.5*-0.0005]),
        }

manip_cfg_T1 = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_shift+30*0.5*-0.010, 30*0.5*0.020],)
        }


read0_cfg_T1 = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [T_shift + 30*0.5*0.004, LP_shift + 30*0.5*0.000],)
#        'step1' : set_step(time = 1e-6, qubits = qubits, voltages = [30*0.5*0.002, 30*0.5*0.002]),
        }


read_cfg_T1 = {
        'step1' : set_step(time = 0.3e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        'step2' : set_step(time = 0.05e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        'step3' : set_step(time = 4e-3, qubits = qubits, voltages = [T_shift + 30*0.5*0, LP_shift + 30*0.5*0]),
        }

#%% Sequence definitions

sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence_cfg_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

sequence1_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence1_cfg_type = ['init', 'manip01','read', 'init2', 'manip2', 'read2']

sequence11_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence11_cfg_type = ['init', 'manip11','read', 'init2', 'manip2', 'read2']

sequence23_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence23_cfg_type = ['init', 'manip21','read', 'init2', 'manip2', 'read2']

sequence21_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence21_cfg_type = ['init', 'manip21','read', 'init2', 'manip2', 'read2']

sequence31_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence31_cfg_type = ['init', 'manip31','read', 'init2', 'manip2', 'read2']

sequence41_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence41_cfg_type = ['init', 'manip41','read', 'init2', 'manip2', 'read2']

sequence51_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence51_cfg_type = ['init', 'manip51','read', 'init2', 'manip2', 'read2']

sequence61_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence61_cfg_type = ['init', 'manip61','read', 'init2', 'manip2', 'read2']

sequence71_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence71_cfg_type = ['init', 'manip71','read', 'init2', 'manip2', 'read2']

sequence81_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence81_cfg_type = ['init', 'manip81','read', 'init2', 'manip2', 'read2']

sequence91_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence91_cfg_type = ['init', 'manip91','read', 'init2', 'manip2', 'read2']

sequence101_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence101_cfg_type = ['init', 'manip101','read', 'init2', 'manip2', 'read2']


sequence00_cfg = [init_cfg, manip_cfg, read0_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence00_cfg_type = ['init', 'manip00', 'read0', 'read', 'init2', 'manip2', 'read2']

sequence000_cfg = [init_cfg, manip_cfg, read0_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence000_cfg_type = ['init', 'manip000', 'read0', 'read', 'init2', 'manip2', 'read2']

sequence2_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence2_cfg_type = ['init', 'manip','read',]

sequence3_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence3_cfg_type = ['init', 'manip2','read',]

sequence3_cfg2 = [init_cfg, manip2_cfg, read0_cfg, read_cfg,]
sequence3_cfg2_type = ['init', 'manip', 'read0', 'read',]

#sequence_cfg2 = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
#sequence_cfg2_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

manip3_cfg = deepcopy(manip2_cfg)
manip4_cfg = deepcopy(manip2_cfg)
manip5_cfg = deepcopy(manip2_cfg)
manip6_cfg = deepcopy(manip2_cfg)
manip7_cfg = deepcopy(manip2_cfg)
manip8_cfg = deepcopy(manip2_cfg)
manip9_cfg = deepcopy(manip2_cfg)
manip10_cfg = deepcopy(manip2_cfg)
manip11_cfg = deepcopy(manip2_cfg)
manip12_cfg = deepcopy(manip2_cfg)
manip13_cfg = deepcopy(manip2_cfg)
manip14_cfg = deepcopy(manip2_cfg)
manip15_cfg = deepcopy(manip2_cfg)
manip16_cfg = deepcopy(manip2_cfg)

manip000_cfg = deepcopy(manip2_cfg)


sequenceBill_cfg = [init_cfg, manip000_cfg, 
                    manip2_cfg, readBill_cfg, 
                    manip3_cfg, readBill_cfg, 
                    manip4_cfg, readBill_cfg,
                    manip5_cfg, readBill_cfg,
                    manip6_cfg, readBill_cfg,
                    manip7_cfg, readBill_cfg,
                    manip8_cfg, readBill_cfg,
                    manip9_cfg, readBill_cfg,
                    manip10_cfg, readBill_cfg,
                    manip11_cfg, readBill_cfg,
                    manip12_cfg, readBill_cfg,
                    manip13_cfg, readBill_cfg,
                    manip14_cfg, readBill_cfg,
                    manip15_cfg, readBill_cfg,
                    manip16_cfg, readBill_cfg,]

sequenceBill_cfg_type = ['init', 'manip', 
                         'manip2', 'read', 
                         'manip3', 'read3', 
                         'manip4', 'read4',
                         'manip5', 'read5',
                         'manip6', 'read6',
                         'manip7', 'read7',
                         'manip8', 'read8',
                         'manip9', 'read9',
                         'manip10', 'read10',
                         'manip11', 'read11',
                         'manip12', 'read12',
                         'manip13', 'read13',
                         'manip14', 'read14',
                         'manip15', 'read15',
                         'manip16', 'read16',]

#sequenceBill1_cfg = [init_cfg, manip_cfg, read_cfg, 
#                     init2_cfg, manip2_cfg, read2_cfg, 
#                     read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg, read2_cfg]    
#sequenceBill1_cfg_type = ['init', 'manip21','read', 'init2', 'manip22', 'read2', 
#                          'read3', 'read4', 'read5', 'read6', 'read7', 'read8', 'read9', 'read10']

sequenceBill2_cfg = [init_cfg, manip2_cfg, readBill_cfg,]
sequenceBill2_cfg_type = ['init', 'manip2','read',]

sequenceT1_cfg2 = [init_cfg_T1, manip_cfg_T1, read0_cfg_T1, read_cfg_T1,]
sequenceT1_cfg2_type = ['init', 'manip', 'read0', 'read',]

#%% Experiment definitions
finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)

rabi = Rabi(name = 'Rabi', pulsar = pulsar, amplitude = 1)
rabi12 = Rabi_all(name = 'Rabi12', pulsar = pulsar)
rabi123 = Rabi_all(name = 'Rabi123', pulsar = pulsar)
rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)
rabi3 = Rabi(name = 'Rabi3', pulsar = pulsar, amplitude = 1, qubit = 'qubit_2',)
rabi_off = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1', frequency_shift = -30e6)
rabi2_det = Rabi_detuning(name = 'Rabi2', pulsar = pulsar, amplitude = 30*0.5*-0.025, amplitude2 = 30*0.5*0.012, qubit = 'qubit_2',)

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 300e-9)
ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, qubit = 'qubit_1', duration_time = 100e-9, waiting_time = 300e-9)
ramsey3 = Ramsey(name = 'Ramsey3', pulsar = pulsar,qubit = 'qubit_1', waiting_time = 300e-9)
ramsey4 = Ramsey(name = 'Ramsey4', pulsar = pulsar,qubit = 'qubit_2', waiting_time = 300e-9)
ramsey12 = Ramsey_all(name = 'Ramsey12', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.10, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 260e-9, detune_q1= True)
ramsey_2 = Ramsey_all(name = 'Ramsey_2', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.10, amplitude = 1, 
                      duration_time = 125e-9, waiting_time = 260e-9, detune_q1= False)

crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.0305*T_factor, amplitude2 = 30*0.5*0.012*LP_factor,
            amplitudepi = 0.95, frequency_shift = 0.05799e9, duration_time = 220e-9)

allxy = AllXY(name = 'AllXY', pulsar = pulsar, qubit = 'qubit_2')
allxy2 = AllXY(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1')
allxy12 = AllXY_all(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1',)

wait = Wait(name = 'Wait', pulsar = pulsar)

#%% Hardware & qubit settings

#vsg.frequency(18.3700e9)
#vsg2.frequency(19.6749e9)
#
#vsg.power(17.75)
#vsg2.power(6.0)

qubit_1.Pi_pulse_length = 250e-9
qubit_1.halfPi_pulse_length = qubit_1.Pi_pulse_length/2

qubit_2.Pi_pulse_length = 250e-9
qubit_2.halfPi_pulse_length = qubit_2.Pi_pulse_length/2

qubit_2.CRot_pulse_length = 220e-9

#%% Experiment settings
experiment.qubit_number = 2
experiment.readnames = ['Qubit2', 'Qubit1']
experiment.calibration_qubit = 'all'

experiment.threshold = 0.038

experiment.seq_repetition = 50
experiment.saveraw = True

experiment.readout_time = 0.003