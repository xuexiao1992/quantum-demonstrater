# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:38:33 2017

@author: X.X
"""
#%% import module
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
#import qcodes.instrument_drivers.Spectrum.M4i as M4i
#from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray

from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

from mpldatacursor import datacursor

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent, MouseClickEvent, HoverEvent
from pyqtgraph.GraphicsScene.GraphicsScene import GraphicsScene
import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
import time
try:
    from setting_version2 import sweep_loop1
except ImportError:
    print('sweep_loop1 from setting is not imported')
#%%
"""
def gaussian(A, B, x):
  return A * np.exp(-(x/(2. * B))**2.)

def mouseMoved(evt):
  mousePoint = p.vb.mapSceneToView(evt[0])
  label.setText("<span style='font-size: 14pt; color: white'> x = %0.2f, <span style='color: white'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))


# Initial data frame
x = np.linspace(-5., 5., 10000)
y = gaussian(5., 0.2, x)


# Generate layout
win = pg.GraphicsWindow()
label = pg.LabelItem(justify = "right")
win.addItem(label)

#p = win.addPlot(row = 1, col = 0)
p = win.addPlot()
plot = p.plot(x, y, pen = "y")
#p = win.addPlot(x,y)
proxy = pg.SignalProxy(p.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

# Update layout with new data
#noise = np.random.normal(0, .2, len(y))
#y_new = y + noise

#plot.setData(x, y_new, pen = "y", clear = True)
#p.enableAutoRange("xy", False)

pg.QtGui.QApplication.processEvents()

#%%
def mouseMoved(evt):
  mousePoint = p.vb.mapSceneToView(evt[0])
#  label.setText("<span style='font-size: 14pt; color: black'> x = %0.2f, <span style='color: black'> y = %0.2f</span> , <span style='color: black'> z = %0.2f</span>" % (mousePoint.x(), mousePoint.y(), mousePoint.z()))
  label.setText("<span style='font-size: 14pt; color: black'> x = %0.2f, <span style='color: black'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))

label = pg.LabelItem(justify = "right")

pt = QtPlot(remote = False)
x = [1,2,3]
y = [11,22,33]
pt.add(x,y)
#pt.add(DS.frequency_set, DS.vsg2_frequency_set,DS.digitizerqubit_1)
pt.win.addItem(label)
proxy = pg.SignalProxy(pt.subplots[0].scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
#pg.QtGui.QApplication.processEvents()s

#win.close()
"""
#%%
class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()
#%%
class DataCursor(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.annotation = ax.annotate(self.text_template, 
                xy=(self.x, self.y), xytext=(self.xoffset, self.yoffset), 
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        self.annotation.set_visible(False)

    def __call__(self, event):
        self.event = event
        # xdata, ydata = event.artist.get_data()
        # self.x, self.y = xdata[event.ind], ydata[event.ind]
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        if self.x is not None:
            self.annotation.xy = self.x, self.y
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()
#%%
"""
data = np.array([1,2,3])
data1 = np.array([[1,2,33,5],[5,232,7,3],[1,2,3,4]])

data_array1 = DataArray(preset_data = data, name = 'digitizer', is_setpoint = True)

data_array2 = DataArray(preset_data = data, name = 'digitizer2')

data_array3 = DataArray(preset_data = data, name = 'digitizer3')

#data_array4 = DataArray(parameter=digitizer_param, is_setpoint=True)

data_array5 = DataArray(preset_data = data, name = 'digitizer5')

data_array6 = DataArray(preset_data = data1, name = 'digitizer6')
"""
#%%

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()
try_location = 'trytrytry'
"""
#arrays = LP.containers()
arrays2 = []
arrays3 = [data_array1,]
arrays4 = [data_array1, data_array2, data_array3]

data_set = new_data(arrays=arrays3, location=try_location, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

#data_set.save_metadata()
"""

#%% load data
NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()
#formatter = GNUPlotFormat()
test_location = '2017-08-28/14-38-39_finding_resonance_Freq_sweep'
new1_location = '2017-08-28/newnewnew'
new_location = '2017-09-14/15-44-52experiment_testRabi_scan'
#data_set_2 = DataSet(location = test_location, io = NewIO,)
#data_set_2.read()

#raw_data_set = load_data(location = new_location, io = NewIO,)
#%%
sample_rate = int(np.floor(61035/1))
pretrigger = 16
#readout_time = 1e-3
readout_time = 700e-6
#loop_num = 5
#loop_num = len(sweep_loop1['para1']) if 'para1' in sweep_loop1 else 1
qubit_num = 1
repetition = 500

seg_size = int(((readout_time*sample_rate+pretrigger) // 16 + 1) * 16)
#%%
def convert_to_ordered_data(data_set, qubit_num = 1, repetition = 100, name = 'frequency', unit = 'GHz', sweep_array = None):

    qubit_data_array = []
    set_array = []
    
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter].ndarray
        dimension_1 = data_array.shape[0]
        array_name = parameter
        array_id = data_set.arrays[parameter].array_id

        if parameter.endswith('set'):
            if data_array.ndim == 2 and parameter.startswith('index'):
                dimension_2 = int(data_array.shape[-1]/2/(repetition+1)/seg_size/qubit_num)
                sweep_array = sweep_array if sweep_array is not None else np.linspace(0,dimension_2-1,dimension_2)
                data_array = np.array([sweep_array for k in range(dimension_1)])
                array_name = name+'_set'
                array_id = name+'_set'
            if data_array.ndim != 3 or not parameter.startswith('index'):
                set_array.append(DataArray(preset_data = data_array, name = array_name, array_id = array_id, is_setpoint = True))
            
        elif not parameter.endswith('set') and data_array.ndim == 2:
            
            data_num = int(data_array.shape[-1]/2/(repetition+1) * repetition)
            qubit_data_num = int(data_num/qubit_num)

            dimension_2 = int(data_array.shape[-1]/2/(repetition+1)/seg_size/qubit_num)
            qubit_data = np.ndarray(shape = (qubit_num, dimension_1, dimension_2, int(qubit_data_num/dimension_2)))
            
            for k in range(dimension_1):
                raw_data = data_array[k][::2]
                raw_marker = data_array[k][1::2]
                for seg in range(seg_size*qubit_num*dimension_2):
                    if raw_marker[seg] > 0.2:           ##  a better threshold ???
                        break                
                data = raw_data[seg:data_num+seg]
                print('seg',seg)
                data_reshape = data.reshape(int(data_num/seg_size), seg_size)
                print('data_shape',data_reshape.shape)
                for l in range(dimension_2):
                    for q in range(qubit_num):
                        
                        qubit_data[q][k][l] = data_reshape[qubit_num*l+q::dimension_2*qubit_num].reshape(seg_size*repetition,)

                        qubit_data_array.append(DataArray(preset_data = qubit_data[q], name = parameter+'qubit_%d'%(q+1), 
                                                          array_id = array_id+'qubit_%d'%(q+1), is_setpoint = False))
                
        elif not parameter.endswith('set') and data_array.ndim == 3:
            data_num = int(data_array.shape[-1]/2/(repetition+1) * repetition)
            qubit_data_num = int(data_num/qubit_num)

            dimension_2 = data_array.shape[1]
            print('qubit_num, dimension_1, dimension_2, int(qubit_data_num)', qubit_num, dimension_1, dimension_2, int(qubit_data_num))
            qubit_data = np.ndarray(shape = (qubit_num, dimension_1, dimension_2, int(qubit_data_num)))
            
            for k in range(dimension_1):
                for l in range(dimension_2):
                    raw_data = data_array[k][l][::2]
                    raw_marker = data_array[k][l][1::2]
                    for seg in range(seg_size*qubit_num):
                        if raw_marker[seg] > 0.2:           ##  a better threshold ???
                            break               
                    data = raw_data[seg:data_num+seg]          ## here data consists both data from qubit1 and qubit2
                    for q in range(qubit_num):
                        data_reshape = data.reshape(int(data_num/seg_size), seg_size)
                        qubit_data[q][k][l] = data_reshape[q::qubit_num].reshape(seg_size*repetition,)
                        qubit_data_array.append(DataArray(preset_data = qubit_data[q], name = parameter+'qubit_%d'%(q+1), 
                                                          array_id = array_id+'qubit_%d'%(q+1), is_setpoint = False))
    
    data_set_new = DataSet(location = new_location+'_ordered_raw_data', io = NewIO, formatter = formatter)
    for array in set_array:
        data_set_new.add_array(array)
    for q in range(qubit_num):
        data_set_new.add_array(qubit_data_array[q])

    return data_set_new

#%%

def convert_to_01_state(data_set, threshold, qubit_num = 1, repetition = 100, name = 'frequency', unit = 'GHz', sweep_array = None):
    data_set = convert_to_ordered_data(data_set, qubit_num, repetition, name, unit, sweep_array)
    
    qubit_data_array = []
    set_array = []
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter].ndarray
        dimension_1 = data_array.shape[0]
        array_id = data_set.arrays[parameter].array_id
        if parameter.endswith('set'):     ## or data_set.arrays[parameter].is_setpoint
            set_array.append(DataArray(preset_data = data_array, name = parameter, 
                                       array_id = array_id, is_setpoint = True))

        elif not parameter.endswith('set'):
            dimension_2 = data_array.shape[1]
            data = np.ndarray(shape = (dimension_1, dimension_2, repetition))
            
            for k in range(dimension_1):
                for l in range(dimension_2):
                    for j in range(repetition):
                        data[k][l][j] = 1 if np.min(data_array[k][l][j*seg_size:(j+1)*seg_size]) <= threshold else 0
            
            qubit_data_array.append(DataArray(preset_data = data, name = parameter, 
                                              array_id = array_id, is_setpoint = False))
            
    data_set_new = DataSet(location = new_location+'_01_state', io = NewIO, formatter = formatter)

    for array in set_array:
        data_set_new.add_array(array)
    for q in range(qubit_num):
        data_set_new.add_array(qubit_data_array[q])
    
    return data_set_new
#%%
def convert_to_probability(data_set, threshold, qubit_num = 1, repetition = 100, name = 'frequency', unit = 'GHz', sweep_array = None):
    
    data_set = convert_to_01_state(data_set, threshold, qubit_num, repetition, name, unit, sweep_array)
    qubit_data_array = []
    set_array = []
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter].ndarray
        dimension_1 = data_array.shape[0]
        arrayid = data_set.arrays[parameter].array_id
        if parameter.endswith('set'):     ## or data_set.arrays[parameter].is_setpoint
            set_array.append(DataArray(preset_data = data_array, name = parameter, 
                                       array_id = arrayid, is_setpoint = True))
    
        elif not parameter.endswith('set'):
            dimension_2 = data_array.shape[1]
            probability_data = np.ndarray(shape = (dimension_1, dimension_2))
            
            for k in range(dimension_1):
                for l in range(dimension_2):
                    probability_data[k][l] = np.average(data_array[k][l])
                    
            
            qubit_data_array.append(DataArray(preset_data = probability_data, name = parameter, 
                                              array_id = arrayid, is_setpoint = False))
          
    data_set_new = DataSet(location = new_location+'_average_probability', io = NewIO, formatter = formatter)

    for array in set_array:
        data_set_new.add_array(array)
    for q in range(qubit_num):
        data_set_new.add_array(qubit_data_array[q])
    
    return data_set_new
    

#%% plot
def data_set_plot(data_set, data_location):
    
    Plot = MatPlot()
    
    raw_data_set = load_data(location = data_location, io = NewIO,)

    data_set_P = convert_to_probability(raw_data_set, threshold = 0.025)
    x_data = data_set_P.arrays['vsg2_frequency_set'].ndarray
    P_data = data_set_P.arrays['digitizer'].ndarray.T[0]

    x = x_data

    y = P_data

    plt.plot(x, y) 

#%%
#import stationF006
#digitizer = station.digitizer
class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, posttrigger_size,
                 digitizer, label=None, unit=None, instrument=None, **kwargs):
        
#        global digitizer
        self.digitizer = digitizer
        channel_amount = bin(self.digitizer.enable_channels()).count('1')
       
        super().__init__(name=name, shape=(channel_amount*memsize,), instrument=instrument, **kwargs)
        
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        
    def get(self):
#        res = digitizer.single_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        time.sleep(0.2)
        res = self.digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
#        res = multiple_trigger_acquisition(digitizer, self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
#        res = digitizer.single_software_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        print(res.shape)
        return res
        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

#%%
def set_digitizer(digitizer, sweep_num, qubit_num, repetition):
    pretrigger=16
    mV_range=1000
    
    sample_rate = int(np.floor(61035/1))
    
    digitizer.sample_rate(sample_rate)
    
    sample_rate = digitizer.sample_rate()
    
    readout_time = 0.7e-3
    
    qubit_num = qubit_num
    
    seg_size = ((readout_time*sample_rate+pretrigger) // 16 + 1) * 16
    
    sweep_num = sweep_num#len(sweep_loop1['para1']) if 'para1' in sweep_loop1 else 1
    import data_set_plot
    data_set_plot.loop_num = sweep_num
    
    repetition = repetition
    
    memsize = int((repetition+1)*sweep_num*qubit_num*seg_size)
    posttrigger_size = seg_size-pretrigger
    
    #digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
    digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)
    #digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)
    
    digitizer.enable_channels(pyspcm.CHANNEL1 | pyspcm.CHANNEL2)
    
#    digitizer.enable_channels(pyspcm.CHANNEL1)
    digitizer.data_memory_size(memsize)
    
    digitizer.segment_size(seg_size)
    
    digitizer.posttrigger_memory_size(posttrigger_size)
    
    digitizer.timeout(60000)
    
    digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)
    
    #trig_mode = pyspcm.SPC_TMASK_SOFTWARE
    #trig_mode = pyspcm.SPC_TM_POS
    trig_mode = pyspcm.SPC_TM_POS | pyspcm.SPC_TM_REARM
    
    digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 800, level1 = 900)
    
    dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size, digitizer = digitizer)

    return digitizer, dig
