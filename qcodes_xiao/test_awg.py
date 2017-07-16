# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:37:46 2017

@author: think
"""

import qcodes.instrument_drivers.tektronix.AWG5014 as AWG5014
#from qcodes.instrument_drivers.Spectrum.M4i import M4i
#import pyspcm
#import matplotlib.pyplot as plt
import numpy as np


freq=1e6
sampling_rate=1.2e9
Vpp=1 #from 50mV to 2V pk-pk
channel=1

#Lenght of array of data to upload the waveform
length = int(sampling_rate / freq )

#Create Square Wave
w = np.zeros(length)
w[:int(length/2)] = 1
m1 = np.zeros(length)
m1[0]=1
m2 = np.zeros(length)
m2[0]=1


#Loading awg
awg = AWG5014.Tektronix_AWG5014(name='awg', address='TCPIP0::169.254.141.235::inst0::INSTR')
#Send .wfm file in the specified folder. It is not yet loaded in awg waveform list. 
#w should be an array of float. 1 means maximum amplitude (it is set separately)
awg.pack_waveform(w, m1, m2)
#Loading the waveform in the list
awg.import_waveform_file('TestWave', 'C:/Users/OEM/Documents/Uploaded_Waveforms/TestWave.wfm')
#Set frequency and Amplitude output. How using the driver?
awg.write('AWGC:RRAT %d' % freq)
awg.write('SOURCE1:VOLTAGE:AMPLITUDE %d' % Vpp)

#Parameters Digitizer
rate = int(np.floor(250000000/1)) #Clock rate must be 250M divided by power of 2
mV_range = 2000
input_path = 0
termination = 0
coupling = 0
compensation = 0

#Loading Digitizer driver
#m4 = M4i(name='M4i', server_name=None)

#Set Digitizer Parameters
#m4.clock_mode(pyspcm.SPC_CM_INTPLL)
#m4.sample_rate(rate)
#m4.enable_channels(pyspcm.CHANNEL2)
#m4.set_channel_settings(2,mV_range, input_path, termination, coupling, compensation)
#memsize = 2**8
#posttrigger_size = int(memsize/2)

#Select waveform from the lsit and run channel
awg.ch1_waveform('TestWave')
awg.ch1_state(channel)
awg.run()

#Measure voltage and then plot versus time
#calc = m4.single_software_trigger_acquisition(mV_range,memsize,posttrigger_size)
#time = np.linspace(0,((len(calc)-1)/rate),len(calc))
#plt.plot(10**6*time, calc,'r')

#Stop output AWG
awg.ch1_state(0)
awg.stop()

##Close instruments
awg.close()
#m4.close()