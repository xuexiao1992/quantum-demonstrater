#%% Load packages
import logging
#logging.basicConfig(level=logging.INFO) #used to see logging in console when not using the ZMQ GUI logger

import qcodes
import qtt
import numpy as np

#from qcodes.utils.loggingGUI import installZMQlogger
#import qcodes.instrument_drivers.stanford_research.SR830 as SR830
import qcodes.instrument_drivers.tektronix.AWG5014 as AWG5014
#import qcodes.instrument_drivers.QuTech.IVVI as IVVI
import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qubit import Qubit 
#import users.boterjm.Drivers.QuTech.IVVI as IVVI
#import users.boterjm.Drivers.Spectrum.M4i as M4i
#import users.boterjm.Drivers.american_magnetics.AMI430_IP as AMI430

import E8267D as E8267D

#from qtt.qtt_toymodel import virtual_gates
#from users.petitlp.measurements.virtual_gates import virtual_gates 

#%% Instruments and gate/outputs maps definition
_initialized = False

GenInfo = {
    'Is_Gain' : 1e8,
    'Id_Gain' : 1e8,
    'Ir_Gain' : 1e8
    }


# Format (instrument_index, query, multiplier)
gate_map = {
    # bias dacs
    'G1': (0,'dac1'),
    'G2': (0,'dac2'),
    'G3': (0,'dac3'),
    'ST': (0,'dac4'),
    'LB': (0,'dac5'),
    'RB': (0,'dac6'),
    'C': (0,'dac7'),
    'R': (0,'dac8'),
    'NA9': (0,'dac9'),
    'NA10': (0,'dac10'),
    'NA11': (0,'dac11'),
    'NA12': (0,'dac12'),
    'Vsd': (0,'dac13'),
    'NA14': (0,'dac14'),
    'NA15': (0,'dac15'),
    'NA16': (0,'dac16'),
    
    'Vsd_AC': (2,'amplitude'),
    
    'Vgate_AC': (1, 'amplitude')
}

output_map = None

station = None
ivvi = None
digitizer = None
lockin = None
awg = None
awg2 = None

vsg = None
vsg2 = None

magnet=None
sig_gen=None
keithley=None

qubit_1 = None
qubit_2=None

mwindows=None

location_matfiles = 'D:/Measurements/Apr72017UNSW-19th&20th_Devices_NewBatch/data/BOTTOM/Matlab'

datadir = 'D:/Measurements/Apr72017UNSW-19th&20th_Devices_NewBatch/data/BOTTOM/Qcodes'
qcodes.DataSet.default_io = qcodes.DiskIO(datadir)

#%%
def getStation():
    global station
    return station

#%%
def close(verbose=1):
    global station

    for instr in station.components.keys():
        if verbose:
            print('close %s' % station.components[instr])
        try:
            station.components[instr].close()
        except:
            print('could not close instrument %s' % station.components[instr])

#%%
def initialize(reinit=False, server_name=None):
    global ivvi, digitizer, lockin1, lockin2, awg, awg2, vsg, vsg2, magnet, sig_gen, keithley, gate_map, station, mwindows, output_map, qubit_1, qubit_2
    
    #qcodes.installZMQlogger()
    logging.info('LD400: initialize')
    print('\n')
    
    if _initialized and not reinit:
        return station
    
    
    # initialize qubit object
    
    qubit_1 = Qubit(name = 'qubit_1')

    qubit_2 = Qubit(name = 'qubit_2')

    qubit_1.define_gate(gate_name = 'Microwave1', gate_number = 1, microwave = 1, channel_I = 'ch1', channel_Q = 'ch2', channel_PM = 'ch1_marker1')

#   Qubit_1.define_gate(gate_name = 'RP1', gate_number = 2, microwave = 1, channel_I = 'RP1I', channel_Q = 'RP1Q')
#
    qubit_1.define_gate(gate_name = 'T', gate_number = 3, gate_function = 'plunger', channel_VP = 'ch7')
#
    qubit_2.define_gate(gate_name = 'Microwave2', gate_number = 4, microwave = 1, channel_I = 'ch3', channel_Q = 'ch4', channel_PM = 'ch1_marker2')
#
    qubit_2.define_gate(gate_name = 'LP', gate_number = 5, gate_function = 'plunger', channel_VP = 'ch6')


    
    
    # Loading AWG
#    logging.info('LD400: load AWG driver')
#    awg = AWG5014.Tektronix_AWG5014(name='awg', address='TCPIP0::169.254.141.235::inst0::INSTR', server_name=server_name)
#    print('awg loaded')
    
#    logging.info('LD400: load AWG driver')
#    awg2 = AWG5014.Tektronix_AWG5014(name='awg2', address='TCPIP0::169.254.110.163::inst0::INSTR', server_name=server_name)
#    print('awg2 loaded')


    logging.info('LD400: load AWG driver')
    awg2 = AWG5014.Tektronix_AWG5014(name='awg2', address='TCPIP0::192.168.0.4::inst0::INSTR', server_name=server_name)
    print('awg2 loaded')
    
    logging.info('LD400: load AWG driver')
    awg = AWG5014.Tektronix_AWG5014(name='awg', address='TCPIP0::192.168.0.9::inst0::INSTR', server_name=server_name)
    print('awg loaded')

    awg2.write('SOUR1:ROSC:SOUR EXT')
    awg.write('SOUR1:ROSC:SOUR INT')
    awg.force_trigger()
    awg.clock_freq(1e9)
    awg2.clock_freq(1e9)
    awg2.trigger_level(0.5)
    #Loading Microwave source
    logging.info('Keysight signal generator driver')
    vsg = E8267D.E8267D(name='vsg',address='TCPIP::192.168.0.11::INSTR',server_name=server_name)
    print('VSG loaded')
    
    logging.info('Keysight signal generator driver')
    vsg2 = E8267D.E8267D(name='vsg2',address='TCPIP::192.168.0.12::INSTR',server_name=server_name)
    print('VSG2 loaded')




#    #load keithley driver
#    keithley = Keithley_2700.Keithley_2700(name='keithley', address='GPIB0::15::INSTR', server_name=server_name)
#    

#     Loading digitizer
    logging.info('LD400: load digitizer driver')
    digitizer = M4i.M4i(name='digitizer', server_name=server_name)
    if digitizer==None:
        print('Digitizer driver not laoded')
    else:
        print('Digitizer driver loaded')
    print('')

    
    logging.info('all drivers have been loaded')
    
    
#     Create map for gates and outputs
#    gates = virtual_gates(name='gates', gate_map=gate_map, server_name=server_name, instruments=[ivvi, lockin1, lockin2])

#    output_map = {
#                  'Id': lockin1.X,
#                  'Is': lockin2.X,
#                  'Id_R': lockin1.R,
#                  'Is_R' : lockin2.R,
#                  'Idc': keithley.amplitude
#    }
    
    #Creating the experimental station
    #station = qcodes.Station(ivvi, awg, lockin1, lockin2, digitizer, gates)
#    station = qcodes.Station(ivvi, lockin1, lockin2, digitizer, gates)
    station = qcodes.Station(awg, awg2, vsg, vsg2, digitizer, qubit_1, qubit_2)
    logging.info('Initialized LDHe station')
    print('Initialized LDHe station\n')
    
    return station
    
#%% Initializing station    
#initialize()