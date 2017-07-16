# Some libs. ...

import sys
# import spectrum driver functions
import pyspcm
from pyspcm import *
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
import time
import math
from decimal import Decimal
import os.path
import h5py

def int32(mynumber=0):
    return ct.c_int32(int(mynumber))

def init_card():
	# define device name
	dev = "/dev/spcm0"
	# put in c readible string
	b_dev = dev.encode('utf-8')
	hCard = spcm_hOpen(b_dev);
	# Check if card is online
	if hCard == None:
	    sys.stdout.write("Kernel drivers for aquisition card not loaded?\n")
	    exit()
	return hCard

def end_session(session):
	spcm_vClose(session)

def reset(session):
	dwError = spcm_dwSetParam_i32 (session, SPC_M2CMD, M2CMD_CARD_RESET)

def get_segment_size(wanted_size):
	# Check that segments size is acceptable. (note min 16, max 8 GB)
	if wanted_size!= wanted_size%16:
		segment_size_actual = round(wanted_size + 16 - wanted_size%16)
	else:
		segment_size_actual = round(wanted_size)

	return segment_size_actual

def chan_to_bin(channels):
	# Formats the number for selection the right active channels.
	channels = 2**channels

	return channels.sum()

def set_trigger_channel_single(session, channel, mode, level0, level1 = None):
	spcm_dwSetParam_i32 (session, SPC_TRIG_ORMASK, 0) 
	spcm_dwSetParam_i32(session, SPC_TRIG_CH_ORMASK0,getattr(pyspcm,'SPC_TMASK0_CH{}'.format(channel)))

	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_TRIG_CH{}_MODE'.format(channel)), mode)
	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_TRIG_CH{}_LEVEL0'.format(channel)), level0)
	if (level1!= None):
		spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_TRIG_CH{}_LEVEL1'.format(channel)), level1)
	spcm_dwSetParam_i32(session, SPC_TRIG_DELAY,int32(0))

def set_ext_trigg(session, mode, level0, level1 = None):
	spcm_dwSetParam_i32(session, SPC_TRIG_ORMASK, SPC_TMASK_EXT0) 
	spcm_dwSetParam_i32(session, SPC_TRIG_CH_ORMASK0,0)

	spcm_dwSetParam_i32(session, SPC_TRIG_EXT0_MODE, mode)
	spcm_dwSetParam_i32(session, SPC_TRIG_EXT0_LEVEL0, int32(level0))
	if (level1!= None):
		spcm_dwSetParam_i32(session, SPC_TRIG_EXT0_LEVEL1, int32(level1))

	spcm_dwSetParam_i32(session, SPC_TRIG_TERM, 1)

def set_trigger_software(session):
	spcm_dwSetParam_i32 (session, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE) 
	spcm_dwSetParam_i32 (session, SPC_TRIG_ANDMASK, 0) 

def set_clock(session,freq):
	# FREQ in HZ
	spcm_dwSetParam_i32(session, SPC_CLOCKMODE, SPC_CM_INTPLL)
	spcm_dwSetParam_i64(session, SPC_SAMPLERATE, int64(int(freq)))
	# NOTE that the clock will not be the same as the actual clock set.

def get_clock(session):
	rate_actual = int64 (0)
	spcm_dwGetParam_i64 (session, SPC_SAMPLERATE, byref(rate_actual))

	return float(rate_actual.value)

def activate_channel(session, channels):
	if(channels.size <=2):
		spcm_dwSetParam_i32 (session, SPC_CHENABLE, int(chan_to_bin(channels)) )
	else: #cannot activate 3 channels, you then need to do 4!
		spcm_dwSetParam_i32 (session, SPC_CHENABLE, 15 )

def set_path(session,channel, path):
	# 0 =  buffered (1MOhm), 1 = HF 50 ohm
	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_PATH{}'.format(channel)), int32(path))	

def get_path(session,channel):
	# 0 =  buffered (1MOhm), 1 = HF 50 ohm
	path = int32()
	spcm_dwGetParam_i32(session, getattr(pyspcm,'SPC_PATH{}'.format(channel)), byref(path))
	return path.value

def set_amp(session, channel, amp):
	# spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_AMP{}'.format(channel)),amp)
	spcm_dwSetParam_i32(session, SPC_READAIPATH, int32(channel))
	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_AMP{}'.format(channel)), int32(amp))

def get_amp(session, channel):
	amp = int32()
	spcm_dwSetParam_i32(session, SPC_READAIPATH, int32(channel))
	spcm_dwGetParam_i32(session, getattr(pyspcm,'SPC_AMP{}'.format(channel)), byref(amp))
	return amp.value

def get_v_ranges(session):
	chan = get_channels(session)
	vrange = np.empty([len(chan)])
	for i in range(len(chan)):
		vrange[i] = get_amp(session,chan[i])

	return vrange

def get_error_info32bit(session):
        """ Read an error from the error register """
        dwErrorReg = pyspcm.uint32(0)
        lErrorValue = pyspcm.int32(0)

        pyspcm.spcm_dwGetErrorInfo_i32(session, pyspcm.byref(dwErrorReg), pyspcm.byref(lErrorValue), None)
        return (dwErrorReg.value, lErrorValue.value)

def set_coupling(session,channel,dc_ac):
	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_ACDC{}'.format(channel)), dc_ac)

def set_ac_dc_offset_calib(session,channel,off_on):
	spcm_dwSetParam_i32(session, getattr(pyspcm,'SPC_ACDC_OFFS_COMPENSATION{}'.format(channel)), off_on)

def set_up_FIFO_multi(session, segment_size, pre_trigger_size, loops = 1):
	spcm_dwSetParam_i32 (session, SPC_CARDMODE, SPC_REC_FIFO_MULTI) 
	spcm_dwSetParam_i32 (session, SPC_SEGMENTSIZE,int(segment_size))
	# spcm_dwSetParam_i32 (session, SPC_MEMSIZE,int(segment_size))
	if(pre_trigger_size < 16):
		# print('pre-triggersize to small (min 16), set to 16')
		pre_trigger_size = 16
	spcm_dwSetParam_i32(session, SPC_POSTTRIGGER,int32(segment_size - pre_trigger_size))
	# trig = int32()
	# spcm_dwGetParam_i32(session, SPC_POSTTRIGGER,byref(trig))
	# print(trig)
	spcm_dwSetParam_i32(session, SPC_LOOPS, int32(loops))
	spcm_dwSetParam_i32(session, SPC_TIMEOUT, int32(0))

def num_channels(session):
	num_channels = int32()
	spcm_dwGetParam_i32(session, SPC_CHCOUNT, byref(num_channels))
	return num_channels.value;

def get_resolution(session):
	resolution = int32()
	spcm_dwGetParam_i32(session,SPC_MIINST_MAXADCVALUE, byref(resolution))
	return resolution.value

def get_channels(session):
	channels_enabled = int32()
	spcm_dwGetParam_i32(session, SPC_CHENABLE, byref(channels_enabled))
	num_chan = num_channels(session)
	# Gets which channels are now active on the device.
	active_chan =np.empty([num_chan])
	for i in range(0, num_chan):
		if (channels_enabled.value%8 == 0):
			channels_enabled.value -= 8
			active_chan[i] = 3
		elif (channels_enabled.value%4 == 0):
			channels_enabled.value -= 4
			active_chan[i] = 2
		elif (channels_enabled.value%2 == 0):
			channels_enabled.value -= 2
			active_chan[i] = 1
		elif (channels_enabled.value%1 == 0):
			channels_enabled.value -= 1
			active_chan[i] = 0

	return active_chan.astype(int)

def to_Volts(session, data_in,chan,v_ranges):
	# Returns an array in Volts
	mydata_volts = np.empty([data_in.shape[0], len(chan),data_in.shape[1]/len(chan)])
	
	for i in range(len(chan)):
		mydata_volts[:,i,:] = data_in[:,i::len(chan)].astype(float)*(v_ranges[i]/32768)

	return mydata_volts

def free_up_memory_space(session, points):
	spcm_dwSetParam_i32(session, SPC_DATA_AVAIL_CARD_LEN, int32(points*2))

def append_to_file(my_data, file_handle, chan, cycle):
	for i in range(my_data.shape[1]):
		file_handle["ch%i"%chan[i]][cycle:cycle + my_data.shape[0]] = my_data[:,i,:]

def save_data(session, data_in, available_dp, start_dp, chunck_len, buffer_size, file_handle, chan, v_ranges, cycle):
	size = int(available_dp/chunck_len)
	if (size == 0):
		return 0
	tmp = np.empty([size*chunck_len])
	# print(available_dp, start_dp, chunck_len, buffer_size, size, len(mychan))
	if(start_dp + chunck_len*size < buffer_size):
		tmp = data_in[start_dp: start_dp + chunck_len*size]
	else:
		tmp[:buffer_size-start_dp] = data_in[start_dp: buffer_size]
		tmp[buffer_size-start_dp:] = data_in[0:size*chunck_len - (buffer_size - start_dp) ]

	free_up_memory_space(session, size*chunck_len)

	tmp = to_Volts(session,tmp.reshape(size,chunck_len),chan,v_ranges)
	append_to_file(tmp,file_handle,chan,cycle)
	return size

def get_buffer_info_fifo(session):
	# Check what the length is of one measurment segment (for all channels)
	num_chan = num_channels(session)

	mysegment_size = int32()
	spcm_dwGetParam_i32(session, SPC_SEGMENTSIZE, byref(mysegment_size))

	one_chunk_size = int(num_chan * mysegment_size.value)

	# make notify size a compatible size around the segmentsize -- tells when to tranfer data.
	# lNotifySize in bytes...
	if(one_chunk_size > KILO(2)):
		Notify_size = int32( one_chunk_size*2 + 4096 - one_chunk_size*2%4096)
	else:
		a = np.array([16,32,64,128,256,512,1024,2048])
		Notify_size = int32(a[np.argmax(a >= one_chunk_size*2)])

	cycles = int32()
	spcm_dwGetParam_i32(session, SPC_LOOPS,byref(cycles))

	if(Notify_size.value*4 > MEGA(1)): 
		buffer_len = Notify_size.value*4
	else:
		buffer_len = MEGA(1)

	return one_chunk_size, Notify_size.value, cycles.value, buffer_len

def start_measurement(session,filename, header_description):
	one_chunk_size, lNotifySize, cycles, buffer_len =get_buffer_info_fifo(session)
	chan = get_channels(session)
	v_ranges = get_v_ranges(session)
	filename = mk_header(session, filename, cycles, one_chunk_size/num_channels(session), header_description)
	print(get_resolution(session))
	print("Notify_size", lNotifySize)
	print("chunck size", one_chunk_size)
	print("Cycles: ", cycles)

	# define buffer, all pointers, even mydata :-)
	pvBuffer = ct.create_string_buffer(buffer_len)
	pnData = ct.cast(pvBuffer, ct.POINTER(ct.c_short))
	
	# Tell card that we have a buffer
	spcm_dwDefTransfer_i64(session, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, lNotifySize, pvBuffer, uint64(0), uint64(buffer_len))
	# Start card.
	dwError = spcm_dwSetParam_i32(session, SPC_M2CMD, M2CMD_CARD_START | M2CMD_DATA_STARTDMA | M2CMD_CARD_ENABLETRIGGER )
	# dwError = spcm_dwSetParam_i32(session, SPC_M2CMD, M2CMD_DATA_WAITDMA )

	# if dwError == ERR_TIMEOUT:
	# 	print("TIME_OUT")
	# 	spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP)
	# time.sleep(1)
	mydata = np.ctypeslib.as_array(pnData,shape=(buffer_len/2,))
	i = 0

	# Open file, expected to be in the right group..
	f_handle = h5py.File( filename + ".h5", "r+")
	while (i<cycles):
		available_data = int32()
		spcm_dwGetParam_i32(session, SPC_DATA_AVAIL_USER_LEN, byref(available_data))
		pos_data = int32()
		spcm_dwGetParam_i32(session, SPC_DATA_AVAIL_USER_POS, byref(pos_data))
		i += save_data(session, mydata, available_data.value/2, pos_data.value/2,
						one_chunk_size, buffer_len/2, f_handle, chan,v_ranges,i)
	f_handle.close()
	# return 1

def check_filename(name,add=-1):
	if (add == -1):
		fullname = name
	else:
		fullname = name + "_" + str(add)
	print(os.path.exists(fullname + ".h5"))
	print(fullname + ".h5")
	if(os.path.exists(fullname + ".h5")):
		return check_filename(name, add + 1)
	else:
		return fullname

def mk_header(session, name, cycles, segmentsize,  description="no descrition provided."):
	# meet the awesomeness of the HDS5 format 

	mychan =  get_channels(session)
	name = check_filename(name)
	f = h5py.File(name + ".h5", "w")
	f.attrs["MEAS_INFO"] = description
	f.attrs["TIME_OF_MEAS"] =  time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()) + "\n"
	f.attrs["# aquisitions"] =  str(cycles)
	clk = get_clock(session)
	f.attrs["FREQ(Hz)"] = str(clk)
	f.attrs["NUM_POINTS"] = str(segmentsize)
	f.attrs["MEAS_TIME_ONE_SEGMENT(s)"] = "{:.2E}".format(Decimal(1/clk*segmentsize))
	f.attrs.create("Channels", mychan, (len(mychan),), "i") 

	for i in range(0,num_channels(session)):
		f.create_dataset("ch%i"%mychan[i],(cycles, segmentsize), dtype="float")
		f["ch%i"%mychan[i]].attrs["v_range"] = get_amp(session, mychan[i])
	f.close()
	return name

def get_data_digitizer(freq_meas, meas_time, cycles, location_and_filename, description="No description provided"):
	# FREQ -> freq of the card in Hz
	# MEAS -> time of measurement in seconds
	# cycles -> number of consecutive measurements
	chan_meas = 0
	chan_trig = 1

	# freq_measurement = 120e3
	# meas_time = 3e-3
	# cycles = 10000

	session = init_card()
	reset(session)

	set_clock(session, freq_meas)
	myclk = get_clock(session)

	segment_size = get_segment_size(myclk*meas_time)

	mychan = np.array([chan_meas,chan_trig])
	activate_channel(session, mychan)

	# SET HF path
	set_path(session,chan_meas,0)
	set_path(session,chan_trig,0)


	# set amp of channel
	set_amp(session,chan_meas,500)
	set_amp(session,chan_trig,500)

	# err,err2 = get_error_info32bit(session)
	# print(err,err2)

	# Since using 50 ohm, DC set correction.
	# set_ac_dc_offset_calib(session,chan_meas,1)

	# err,err2 = get_error_info32bit(session)
	# print(err,err2)

	# set_trigger_channel_single(session,chan_trig, SPC_TM_POS, 300)
	set_ext_trigg(session, SPC_TM_POS, -200)
	# set_trigger_software(session)
	set_up_FIFO_multi(session,segment_size,0,cycles)

	data = start_measurement(session, location_and_filename, description)
	# save_data(session, data, "exp_4", 
	# 	"Init experiment LS + pi pulse (MS + LS DOWN!)")

	# err,err2 = get_error_info32bit(session)
	# print(err,err2)
	end_session(session)