#%% Load packages
import logging
#logging.basicConfig(level=logging.INFO) #used to see logging in console when not using the ZMQ GUI logger


#C:\Users\LocalAdmin\AppData\Local\conda\conda\envs\QCoDeS\Lib\site-packages\qtt-0.1.3-py3.6.egg\qtt

import sys
sys.path.append('C:\\Users\\LocalAdmin\\AppData\\Local\\conda\\conda\\envs\\QCoDeS\\Lib\\site-packages\\qtt-0.1.3-py3.6.egg')
#


import qcodes
import qtt
import numpy as np

#from qcodes.utils.loggingGUI import installZMQlogger
#import qcodes.instrument_drivers.stanford_research.SR830 as SR830
import qcodes.instrument_drivers.tektronix.AWG5014 as AWG5014
#import qcodes.instrument_drivers.QuTech.IVVI as IVVI
import qcodes.instrument_drivers.tektronix.AWG5200 as AWG5200
import qcodes.instrument_drivers.QuTech.IVVI as IVVI

from qtt.instrument_drivers.gates import virtual_IVVI

import qcodes.instrument_drivers.tektronix.Keithley_2700 as keith2700

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qubit import Qubit 
#import users.boterjm.Drivers.QuTech.IVVI as IVVI
#import users.boterjm.Drivers.Spectrum.M4i as M4i
#import users.boterjm.Drivers.american_magnetics.AMI430_IP as AMI430

import E8267D as E8267D
from qcodes.station import Station
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
    'VI1': (0, 1), 
    'VI2': (0, 2),
    'acQD': (0, 4), 
    'acres': (0, 16),
    
    'RS': (0, 5),
    'RD': (0, 6),
    'LP': (0, 9), 
    'LPF': (0, 10),
    'RP': (0, 11), 
    'RPF': (0, 12),

    'LS': (0, 13),
    'T': (0, 15),
    
    'LD': (1, 1),
    'B': (1, 11),
    
    'SQD1': (0, 7),
    'SQD2': (1, 4),
    'SQD3': (0, 8),
    'RQPC': (0, 14),

}
#
def twodotboundaries():
    global ivvi1
    gate_boundaries = dict({
            'VI1': (-600, 600), 
            'VI2': (-600, 600),
            'acQD': (-300, 300), 
            'acres': (-300, 300),
            
            'RS': (-1000, 300),
            'RD': (-1500, 200),
            'LP': (-1000, 200), 
            'LPF': (-100, 100),
            'RP': (-1500, 200), 
            'RPF': (-100, 100),
            
            'LS': (-1000, 200),
            'T': (-1000, 200),
        
            'LD': (-1000, 200),
            'B': (-1000, 200),
            
            'SQD1': (-1000, 300),
            'SQD2': (-1000, 300),
            'SQD3': (-1000, 300),
            'RQPC': (-1000, 300),
        })
    if ivvi1 is not None:
        # update boundaries to resolution of the dac        
        for k in gate_boundaries:
            bb=gate_boundaries[k]
            bb=(ivvi1.round_dac(bb[0]), ivvi1.round_dac(bb[1]) )
            gate_boundaries[k] =bb
    return gate_boundaries
'''
def twodotboundaries():
    global ivvi1
    gate_boundaries = dict({
            'VI1': (-600, 2000), 
            'VI2': (-600, 2000),
            'acQD': (-1300, 2000), 
            'acres': (-2000, 1300),
            
            'RS': (-1500, 1200),
            'RD': (-1500, 1500),
            'LP': (-1500, 1500), 
            'LPF': (-2000, 1500),
            'RP': (-1500, 1500), 
            'RPF': (-1700, 1500),
            
            'LS': (-1500, 1500),
            'T': (-1900, 1500),
        
            'LD': (-1500, 1500),
            'B': (-1500, 1500),
            
            'SQD1': (-2000, 1500),
            'SQD2': (-1500, 1900),
            'SQD3': (-1500, 1500),
            'RQPC': (-1500, 1500),
        })

    if ivvi1 is not None:
        # update boundaries to resolution of the dac        
        for k in gate_boundaries:
            bb=gate_boundaries[k]
            bb=(ivvi1.round_dac(bb[0]), ivvi1.round_dac(bb[1]) )
            gate_boundaries[k] =bb
    return gate_boundaries
'''
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

datadir = 'D:\\Data\\RB_experiment'
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
    global ivvi1, ivvi2, gates, digitizer, lockin1, lockin2, awg, awg2, vsg, vsg2, magnet, sig_gen, keithley, gate_map, station, mwindows, output_map, qubit_1, qubit_2
    
    #qcodes.installZMQlogger()
    logging.info('LD400: initialize')
    print('\n')
    
    if _initialized and not reinit:
        return station
    
    if server_name is None:
        server_name_virtual = None
    else:
        server_name_virtual = server_name + '_virtualgates'
    
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
    
    qubit_1.define_neighbor(neighbor_qubit = 'qubit_2', pulse_delay = 0e-9)

    qubit_2.define_neighbor(neighbor_qubit = 'qubit_2', pulse_delay = 10e-9)
    
    # Loading IVVI
    logging.info('LD400: load IVVI driver')
    ivvi1 = IVVI.IVVI(name='ivvi1', dac_step=10, dac_delay=0.025, address='COM5', server_name=server_name, 
                     numdacs=16, use_locks=True)
    print('')
    ivvi2 = IVVI.IVVI(name='ivvi2', dac_step=10, dac_delay=0.025, address='COM6', server_name=server_name, 
                     numdacs=16, use_locks=True)
#    

#    def twodotboundaries():
#        global ivvi1
#        gate_boundaries = dict({
#                'VI1': (-600, 600), 
#                'VI2': (-600, 600),
#                'acQD': (-300, 300), 
#                'acres': (-300, 300),
#                
#                'RS': (-1000, 200),
#                'RD': (-1500, 200),
#                'LP': (-1000, 200), 
#                'LPF': (-100, 100),
#                'RP': (-1500, 200), 
#                'RPF': (-100, 100),
#                
#                'LS': (-1000, 200),
#                'T': (-1000, 200),
#            
#                'LD': (-1000, 200),
#                'B': (-1000, 200),
#            
#                'SQD1': (-1000, 200),
#                'SQD2': (-1000, 200),
#                'SQD3': (-1000, 200),
#                'RQPC': (-1000, 200),
#            })
#
#        if ivvi1 is not None:
#        # update boundaries to resolution of the dac        
#            for k in gate_boundaries:
#                bb=gate_boundaries[k]
#                bb=(ivvi1.round_dac(bb[0]), ivvi1.round_dac(bb[1]) )
#                gate_boundaries[k] =bb
#        return gate_boundaries
#    boundaries = twodotboundaries()


    gates = virtual_IVVI(name='gates', gate_map=gate_map, server_name=server_name_virtual, instruments=[ivvi1,ivvi2])
    
    gate_boundaries = twodotboundaries()
    gates.set_boundaries(gate_boundaries)
    
    logging.info('boundaries set to gates')
    # Loading AWG
    
    logging.info('LD400: load AWG driver')
    awg2 = AWG5014.Tektronix_AWG5014(name='awg2', address='TCPIP0::192.168.0.7::inst0::INSTR', server_name=server_name)
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
    
    


    """
    logging.info('LD400: load AWG driver')
    awg = AWG5200.Tektronix_AWG5014(name='awg', address='TCPIP0::192.168.0.8::inst0::INSTR', timeout = 180, server_name=server_name)
    print('awg loaded')
    """

#    #load keithley driver
    keithley = keith2700.Keithley_2700(name='keithley', address='GPIB0::15::INSTR', server_name=server_name)
#    

#     Loading digitizer
    logging.info('LD400: load digitizer driver')
    digitizer = M4i.M4i(name='digitizer', server_name=server_name)
    if digitizer==None:
        print('Digitizer driver not laoded')
    else:
        print('Digitizer driver loaded')
    print('')
    
#    logging.info('all drivers have been loaded')
    print('digitizer finished')
    
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
#    station = qcodes.Station(awg, awg2, vsg, vsg2, digitizer, qubit_1, qubit_2)
    components = [awg, awg2, vsg, vsg2, digitizer, qubit_1, qubit_2, gates, keithley]
    station = Station(*components, update_snapshot=True)
    print('station initialized')
    logging.info('Initialized LDHe station')
    print('Initialized LDHe station\n')
    
    return station
    
#%% Initializing station    
#initialize()