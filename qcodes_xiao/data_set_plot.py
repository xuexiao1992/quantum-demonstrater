# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:38:33 2017

@author: X.X
"""
#%% import module
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
#import qcodes.instrument_drivers.Spectrum.M4i as M4i
#from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter, MultiParameter
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
#qubit_num = 1
#repetition = 500

seg_size = int(((readout_time*sample_rate+pretrigger) // 16 + 1) * 16)


#%% new functions for single line np_arrays to be passed onto the data_set function

def  organise(data, qubit_num = 2, repetition = 100, seg_size = 64):
    seg_size = int(seg_size)
    qubit_num = qubit_num
    repetition = repetition
    # takes raw digitizer signal and organises it into the right readout positions along the pulse train....
    # it will return an array of (a,b,c,d) where a is the qubit readout number, b is the sweep number , c is the repetition number,
    # and d is the raw readout traces. 

    dimension_2 = int(data.shape[0]/2/(repetition+1)/seg_size/qubit_num) # number of steps in sweep
    
    print('dimension2',dimension_2)

    data_num = int(data.shape[0]/2/(repetition+1) * repetition) #total length of usable raw data
    qubit_data_num = int(data_num/qubit_num)
    qubit_data = np.zeros(shape = (qubit_num, dimension_2, repetition, int(qubit_data_num/dimension_2/repetition)))

    # every even point in data corresponds to the raw data of ch0 (readout signal) and
    # every odd point in data corresponds to the raw data of ch1 (marker signal). 
    raw_data = data[::2] 
    raw_marker = data[1::2]

    #find the position of the first marker (where the useful data stars from)
    for seg in range(seg_size*qubit_num*dimension_2):
        if raw_marker[seg] > 0.2:           ##  a better threshold ???
            break                
    #trim the data before the first marker and after the last marker 
    
    data_trimmed = raw_data[seg:data_num+seg]
    data_reshape = data_trimmed.reshape(int(data_num/seg_size), seg_size)

    for l in range(dimension_2):
        for q in range(qubit_num):
                
            qubit_data[q][l] = data_reshape[qubit_num*l+q::dimension_2*qubit_num]

    return qubit_data

def convert_to_01_state2(data_set, threshold):
    # takes raw readout data and assigns a 1 if the measure signal is below the defined threshold and 0 otherwise.
            
    data_set_new = np.zeros(shape= (data_set.shape[0],data_set.shape[1],data_set.shape[2])) 
    for k1 in range(data_set.shape[0]):
        for k2 in range(data_set.shape[1]):
            for k3 in range(data_set.shape[2]):
                data_set_new[k1][k2][k3] = 1 if np.min(data_set[k1][k2][k3]) <= threshold else 0

    
    return data_set_new

def  convert_to_probability2(data_set):
    
    probability_data = np.zeros(shape = (data_set.shape[0], data_set.shape[1]))
    
    for k in range(data_set.shape[0]):
        for l in range(data_set.shape[1]):
            probability_data[k][l] = np.average(data_set[k][l])
            
    return probability_data
    

#%% 
def seperate_data(data_set, location, NewIO, formatter, qubit_num = 1, repetition = 100, sweep_arrays = None, sweep_names = None):
    #this function will seperate the raw data for each experiment (appended to the same seqeunce)
    #into different data files. This will make plotting and data handling easier. 
    start = 0
    end = 0
    seperated_data = []
    for count, array in enumerate(sweep_arrays):
 
        end = start+ len(sweep_arrays[count]) -1
        seperated_data.append(DataSet(location = location+'_'+sweep_names[count]+'_set', io = NewIO, formatter = formatter))
        for parameter in data_set.arrays:
            if parameter.endswith('set') and data_set.arrays[parameter].ndarray.ndim >1:
                name = sweep_names[count]+'_set'
            else:
                name = parameter
            if data_set.arrays[parameter].ndarray.ndim >1:
                seperated_data[count].add_array(DataArray(preset_data = data_set.arrays[parameter][:,start:end], name = name, array_id = name, is_setpoint = True))
            else:
                seperated_data[count].add_array(DataArray(preset_data = data_set.arrays[parameter], name = name, array_id = name, is_setpoint = True)) 
        start = end+1
        
        
    return seperated_data 

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
                        n = 2 if q == 0 else q
                        if q>=2:
                            n = q+1
                        qubit_data_array.append(DataArray(preset_data = qubit_data[q], name = parameter+'qubit_%d'%(n), 
                                                          array_id = array_id+'qubit_%d'%(n), is_setpoint = False))
                
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
                        n = 2 if q == 0 else q
                        qubit_data_array.append(DataArray(preset_data = qubit_data[q], name = parameter+'qubit_%d'%(n), 
                                                          array_id = array_id+'qubit_%d'%(n), is_setpoint = False))
    
    data_set_new = DataSet(location = new_location+'_ordered_raw_data', io = NewIO, formatter = formatter)
    for array in set_array:
        data_set_new.add_array(array)
    for q in range(qubit_num):
        data_set_new.add_array(qubit_data_array[q])

    return data_set_new

#%%

def convert_to_01_state(data_set, threshold, qubit_num = 1, repetition = 100):
    #data_set = convert_to_ordered_data(data_set, qubit_num, repetition, name, unit, sweep_array)
    
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
def convert_to_probability(data_set, location, NewIO, formatter,  threshold, qubit_num = 1, repetition = 100,):
    for parameter in data_set.arrays:
        if len(data_set.arrays[parameter].ndarray.shape) == 2 and parameter.endswith('set'):
            data_set_new = DataSet(location = location+'_average_probability_'+parameter, io = NewIO, formatter = formatter)    
        
#    data_set = convert_to_01_state(data_set, threshold, qubit_num, repetition, name, unit, sweep_array)
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
          

    for array in set_array:
        data_set_new.add_array(array)
    for q in range(qubit_num):
        data_set_new.add_array(qubit_data_array[q])
    
    return data_set_new

def average_probability(data_set, location, NewIO, formatter,  qubit_num =1):

    
    for parameter in data_set.arrays:
        if len(data_set.arrays[parameter].ndarray.shape) == 2 and parameter.endswith('set'):
            data_set_new = DataSet(location = location+'_average_probability_data_'+parameter, io = NewIO, formatter = formatter)    

    for parameter in data_set.arrays:
        if len(data_set.arrays[parameter].ndarray.shape) == 2:
            data = deepcopy(data_set.arrays[parameter].ndarray)
            data = np.average(data, axis = 0)
            is_setpoint = data_set.arrays[parameter].is_setpoint
            name = data_set.arrays[parameter].name
            array_id = data_set.arrays[parameter].array_id
            data_set_new.add_array(DataArray(preset_data = data, name = name, array_id = array_id, is_setpoint = is_setpoint))
    return data_set_new
#        self.average_plot.add(self.averaged_data.digitizer,figsize=(1200, 500))
    

#%%

def majority_vote(data_set, threshold, qubit_num = 1, repetition = 100, name = 'frequency', unit = 'GHz', sweep_array = None, average = False):
    
    data_set = convert_to_01_state(data_set, threshold, qubit_num, repetition, name, unit, sweep_array)
    
    set_array = []
    
    for parameter in data_set.arrays:
        data_array = data_set.arrays[parameter].ndarray
        dimension_1 = data_array.shape[0]
        arrayid = data_set.arrays[parameter].array_id
        if parameter.endswith('set'):     ## or data_set.arrays[parameter].is_setpoint
            set_array.append(DataArray(preset_data = data_array, name = parameter, 
                                       array_id = arrayid, is_setpoint = True))
            
    dimension_2 = len(sweep_array) if sweep_array is not None else 2
#    dimension_1 = 5
    vote_data = np.ndarray(shape = (dimension_1, dimension_2, repetition))
    average_vote_data = np.ndarray(shape = (dimension_1, dimension_2))
    name = 'vote'
    arrayid = 'vote'
    for k in range(dimension_1):
        for l in range(dimension_2):
            for repe in range(repetition):
                voter = np.array([data_set.digitizerqubit_1[k][l][repe],data_set.digitizerqubit_2[k][l][repe],data_set.digitizerqubit_3[k][l][repe],])
                
                vote_data[k][l][repe] =  1 if np.sum(voter) >= 2 else 0 
    
            if average:
                average_vote_data[k][l] = np.average(vote_data[k][l])
                print('average: ', average_vote_data[k][l])
        
    data = vote_data if not average else average_vote_data
    
    vote_data_array =DataArray(preset_data = data, name = name, 
                               array_id = arrayid, is_setpoint = False)
    
    
    data_set_new = DataSet(location = new_location, io = NewIO, formatter = formatter)

    for array in set_array:
        data_set_new.add_array(array)
    data_set_new.add_array(vote_data_array)
        
    return data_set_new
#%%
#def average_vote(data_set, threshold, qubit_num = 1, repetition = 100, name = 'frequency', unit = 'GHz', sweep_array = None):
#    
#    data_set = majority_vote(data_set, threshold, qubit_num, repetition, name, unit, sweep_array)
#    
#    for parameter in data_set.arrays:
#        data_array = data_set.arrays[parameter].ndarray
#        dimension_1 = data_array.shape[0]
#    dimension_2 = len(sweep_array) if sweep_array is not None else 2
#    
#    sweep_data_array =DataArray(preset_data = np.array([0,1]), name = parameter, 
#                               array_id = arrayid, is_setpoint = False)
#    
#    province = np.ndarray(shape = (dimension_2, repetition))
#    for k in range(dimension_1):
#        for l in range(dimension_2):
#            
#    

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
#        res = self.digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
#        res = multiple_trigger_acquisition(digitizer, self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        res = self.digitizer.single_software_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        print(res.shape)
        return res

        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

class digitizer_multiparam(MultiParameter):
    
    def __init__(self,  mV_range, memsize, seg_size, posttrigger_size,
                 digitizer, threshold, qubit_num, repetition, sweep_num, X_sweep_array, saveraw, label=None, unit=None, instrument=None, **kwargs):
        
#        global digitizer
        self.digitizer = digitizer
        channel_amount = bin(self.digitizer.enable_channels()).count('1')
        
        if saveraw == True:
            names= ('raw_data', 'singleshot_data', 'probability_data', 'sweep_data')
            labels= ('raw_data', 'singleshot_data', 'probability_data','sweep_data')
            shapes= ((qubit_num,sweep_num,repetition,int(seg_size)),(qubit_num,sweep_num,repetition),(qubit_num,sweep_num),(qubit_num,sweep_num))
        else:
            names= ('probability_data', 'sweep_data')
            labels= ('probability_data','sweep_data')
            shapes= ((qubit_num,sweep_num),(qubit_num,sweep_num))
        
        
        super().__init__('readout', names= names, labels= labels, shapes=shapes, instrument=instrument, **kwargs)
#        super().__init__('readout', names= ('raw_data', 'data'), labels= ('raw_data', 'data'), shapes=((qubit_num,sweep_num,repetition,int(seg_size)), (1,)) , instrument=instrument, **kwargs)
        self.saveraw = saveraw 
        self.threshold = threshold
        self.qubit_num = qubit_num
        self.repetition = repetition
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        self.X_sweep_array = X_sweep_array
        
    def get(self):
#        res = digitizer.single_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        time.sleep(0.2)
        res = self.digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
        ordered_data = organise(res, qubit_num = self.qubit_num, repetition = self.repetition, seg_size  = self.seg_size)
        thresholded_data = convert_to_01_state2(ordered_data, threshold = self.threshold)
        probability_data = convert_to_probability2(thresholded_data)
        
        ##not sure why i cant just pass X_sweep_array...
        sweep_data = deepcopy(probability_data)
        for i in range(self.qubit_num):
            sweep_data[i,:] = self.X_sweep_array
        
        print(probability_data)
        if self.saveraw == True:
            return (ordered_data, thresholded_data, probability_data, sweep_data)
        else: 
            return (probability_data,sweep_data )

        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """

#%%
def set_digitizer(digitizer, sweep_num, qubit_num, repetition, threshold, X_sweep_array, saveraw):
    pretrigger=16
    mV_range=1000
    threshold = threshold
    sample_rate = int(np.floor(61035/1))
    
    digitizer.sample_rate(sample_rate)
    
    sample_rate = digitizer.sample_rate()
    
    readout_time = 2e-3
    
    qubit_num = qubit_num
    
#    seg_size = ((readout_time*sample_rate+pretrigger) // 16 + 1) * 16
    seg_size = ((readout_time*sample_rate) // 16 + 1) * 16
    
    sweep_num = sweep_num#len(sweep_loop1['para1']) if 'para1' in sweep_loop1 else 1
#    import data_set_plot
#    data_set_plot.loop_num = sweep_num
    
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
    
#    dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size, digitizer = digitizer)
    dig = digitizer_multiparam(mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size, digitizer = digitizer, threshold = threshold, qubit_num =qubit_num , repetition =repetition, sweep_num =sweep_num, X_sweep_array =X_sweep_array, saveraw = saveraw)

    return digitizer, dig
