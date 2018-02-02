# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:11:37 2017

@author: LocalAdmin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:41:38 2017

@author: twatson
"""

import qtt
import numpy as np
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot
from qtt.tools import addPPTslide, addPPT_dataset 
from matplotlib.widgets import Button

import matplotlib.pyplot as plt


def make_pcolor_array(array):
    """ function to modify array for definining x or y axis  to plot with pcolor
    """
    array1 = np.array(array)
    interval = (array1[1] - array1[0])
    arraynew = array1 - interval/2
    arraynew2 =np.append(arraynew,arraynew[-1]+interval )

    return arraynew2
# plot stability_diagram

def plot2Ddata(data_set):
    """ Plot 2D data from a genereric Qcodes loop, with a singlular measurment
    Input
    ----
    data_set : qcodes data set
    """ 
    for array in data_set.arrays:
        if array.endswith('set'):
            if np.ndim(data_set.arrays[array]) ==2:
                x= data_set.arrays[array]
                xlabel = array
            if np.ndim(data_set.arrays[array]) ==1:
                y= data_set.arrays[array]
                ylabel = array 
        else:
            z = data_set.arrays[array]
            zlabel = array
    plt.figure(21, figsize=(12, 8))
    plt.clf() 
    plt.pcolormesh(x[0,:], y, z)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    
    callback = Index(data_set, plt.figure(21))
    ppt = plt.axes([0.81, 0.05, 0.1, 0.075])
    bppt = Button(ppt, 'ppt')
    bppt.on_clicked(callback.ppt) 
    ppt._button = bppt

def plot1Ddata(data_set, paramname = 'keithley_amplitude'):
    """ Plot 1D data from a genereric Qcodes loop, with a singlular measurment.
    Input
    ----
    data_set : qcodes data set
    """     
    for array in data_set.arrays:
        if array.endswith('set'):
            x= data_set.arrays[array]
            xlabel = array

    y = data_set.arrays[paramname]
    ylabel = paramname
    plt.figure(22, figsize=(8, 8))
    plt.clf() 
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    callback = Index(data_set, plt.figure(22))
    ppt = plt.axes([0.81, 0.05, 0.1, 0.075])
    bppt = Button(ppt, 'ppt')
    bppt.on_clicked(callback.ppt) 
    ppt._button = bppt
        
# plot 1D
def plot1D(data_set, measurements = 'All', xaxis = True, sameaxis = False):
    """ Plot 1D data from a qubit experiment
    Input
    -----
    data_set : qcodes data set with qubit metadata
    measurements : list the measurements form the experiment you want to plot i.e ['DCZ', 'Ramsey']
                   'all' will plot all measurements from experiment
    xaxis:  Decide whether to plot the actual values of the xaxis. 
    
    """ 
    metadata = data_set.snapshot()
    qubit_number = metadata['qubit_number']
    X_parameter = metadata['X_parameter']
    X_parameter_type = metadata['X_parameter_type']
    X_all_measurements = metadata['X_all_measurements']
    X_sweep_points = metadata['X_sweep_points']
    readnames = metadata['readnames']
    
       
    if measurements == 'All':
        measurements = X_all_measurements
        
    y = data_set.arrays['probability_data'].mean(axis = 0)      
    plt.figure(20, figsize=(16, 8))
    plt.clf()

    if X_parameter_type == 'Out_Sequence':
        x1 = data_set.arrays[X_parameter+'_set'].mean(axis = 0)
        for i in range(qubit_number):
            if sameaxis == False:
                plt.subplot(qubit_number,1,i+1)
            plt.plot(x1, y[:,i,0]) #this is probably wrong
            plt.xlabel(X_parameter)
            plt.ylabel(readnames[i] + ' prob')
  
    else:
        
        numberofmeasurements = len(measurements)
        j = 1
        jj = 1
        for i in range(qubit_number):
            for k,measurement in enumerate(measurements):
                pos = X_all_measurements.index(measurement)
                x_start  = X_sweep_points[pos]
                x_end =  X_sweep_points[pos+1]

                
                if xaxis  == True:
                    x1 = data_set.arrays['sweep_data'].mean(axis = 0)
                    x = x1[0,x_start:x_end]
                    if sameaxis == False:
                        plt.subplot(qubit_number,numberofmeasurements,j)
                    else:
                        plt.subplot(qubit_number,1,jj)
                    plt.plot(x, y[i,x_start:x_end])
                    t = plt.axis()
                    plt.axis([t[0], t[1], y[i,:].min(), y[i,:].max()])
                    plt.xlabel(measurement)
                
                
                else:
                    x = np.zeros(shape = (x_end-x_start))
                    x[:] = list(range(x_start, x_end))
                    if sameaxis == False:
                        plt.subplot(qubit_number,1,i+1)
                    plt.plot(x, y[i,x_start:x_end])
        
                if k ==0:
                    plt.ylabel(readnames[i] + ' prob')
                j = j+1
            jj = jj+1
    callback = Index(data_set, plt.figure(20))
    ppt = plt.axes([0.81, 0.05, 0.1, 0.075])
    bppt = Button(ppt, 'ppt')
    bppt.on_clicked(callback.ppt) 
    ppt._button = bppt
#    addPPTslide(txt = data_set.location, fig=plt.figure(20), notes=qtt.tools.reshape_metadata(data_set, printformat='fancy'))                

# plot 2D
def plot2D(data_set, measurements = 'All', xaxis = True):
    """ Plot 2D data from a qubit experiment
    Input
    -----
    data_set : qcodes data set with qubit metadata
    measurements : list the measurements form the experiment you want to plot i.e ['DCZ', 'Ramsey']
                   'all' will plot all measurements from experiment
    xaxis:  Decide whether to plot the actual values of the xaxis. 
    
    To do
    -----
    Include arbitrary y sweep parameter (need data to test this..)
    
    """    
    metadata = data_set.snapshot()
    qubit_number = metadata['qubit_number']
    X_parameter = metadata['X_parameter']
    X_parameter_type = metadata['X_parameter_type']
    Y_parameter_type = metadata['Y_parameter_type']
    X_all_measurements = metadata['X_all_measurements']
    X_sweep_points = metadata['X_sweep_points']
    Y_parameter = metadata['Y_parameter']
    readnames = metadata['readnames']
    
    if measurements == 'All':
        measurements = X_all_measurements
        
    y = data_set.sweep_data2 if Y_parameter_type == 'In_Sequence' else data_set.arrays[Y_parameter+'_set']
    y = data_set.arrays[Y_parameter+'_set']     
    z = data_set.arrays['probability_data']    
    plt.figure(20, figsize=(16, 8))
    plt.clf()
#    plt.subplots_adjust(bottom=0.2)

    

    if X_parameter_type == 'Out_Sequence':
        x = data_set.arrays[X_parameter+'_set'].mean(axis = 0)
        for i in range(qubit_number):
            plt.subplot(qubit_number,1,i+1)
            plt.pcolormesh(make_pcolor_array(x), make_pcolor_array(y), z[:,:, i,0])
            plt.xlabel(X_parameter)
            plt.ylabel(Y_parameter)
  
    else:

        if xaxis ==False and measurements == X_all_measurements :
            for i in range(qubit_number): 
                x = list(range(0, X_sweep_points[-1]))
                plt.subplot(qubit_number,1,i+1)
                make_pcolor_array(x)
                plt.pcolormesh(make_pcolor_array(x), make_pcolor_array(y), z[:, i,:])
                plt.ylabel(Y_parameter) #need to change this for an abitrary y sweep
              
        else:    
            numberofmeasurements = len(measurements)
            j = 1
            for i in range(qubit_number):
                for k,measurement in enumerate(measurements):
                    pos = X_all_measurements.index(measurement)
                    x_start  = X_sweep_points[pos]
                    x_end =  X_sweep_points[pos+1]
    
                    
                    if xaxis  == True:
                        x1 = data_set.arrays['sweep_data'].mean(axis = 0)
                        x = x1[0,x_start:x_end]
                        plt.subplot(qubit_number,numberofmeasurements,j)
                        plt.pcolormesh(make_pcolor_array(x), make_pcolor_array(y), z[:, i,x_start:x_end], vmin=z[:,i,:].min(), vmax=z[:,i,:].max())
                        plt.xlabel(measurement)
                        plt.colorbar()
    

                    else:
                        x = np.zeros(shape = (x_end-x_start))
                        x[:] = list(range(x_start, x_end))
                        plt.subplot(qubit_number,1,i+1)
                        plt.pcolormesh(make_pcolor_array(x), make_pcolor_array(y), z[:, i,x_start:x_end])
            
                    if k ==0:
                        plt.ylabel(Y_parameter) #need to change this for an abitrary y sweep
                    j = j+1
    callback = Index(data_set, plt.figure(20))
    ppt = plt.axes([0.81, 0.05, 0.1, 0.075])
    bppt = Button(ppt, 'ppt')
    bppt.on_clicked(callback.ppt) 
    ppt._button = bppt

class Index(object):
    ind = 0
    def __init__(self, data_set, fig):
        self.data_set = data_set
        self.fig =fig
        
    def ppt(self, event):
        addPPTslide(txt = self.data_set.location, fig=self.fig, notes=qtt.tools.reshape_metadata(self.data_set, printformat='fancy'))






#formatter = HDF5FormatMetadata()
#
##data_set = load_data('H:\My Documents/qcodes data/2017-10-16/19-17-31/allxy_experimentAllXY_sequence', formatter =formatter )
#data_set = load_data('C:\\Users\\LocalAdmin\\Documents\\BillCoish_experiment\\2017-10-21\\10-48-47\\BillCoish_experimentAllXY_sequence', formatter =formatter )
#
##plt.figure(21, figsize=(16, 8))
##callback = Index()
##ppt = plt.axes([0.81, 0.05, 0.1, 0.075])
##bppt = Button(ppt, 'ppt')
##bppt.on_clicked(callback.ppt)  
#
#
#plot1D(data_set, measurements = 'All', xaxis =True)