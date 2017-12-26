# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:00:07 2017

@author: X.X
"""

#%%
import numpy as np
from scipy.optimize import curve_fit

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

#%%
IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\BillCoish_experiment')
formatter = HDF5FormatMetadata()

#%%     old data

location_15crot = '2017-10-25/19-36-23/BillCoish_experimentAllXY_sequence'
location_3init = '2017-10-25/20-04-38/BillCoish_experimentAllXY_sequence'
#location = '2017-12-12/17-47-41/RB_experimentAllXY_sequence'


#%%     new data

location_15crot = '2017-10-25/19-36-23/BillCoish_experimentAllXY_sequence'
location_3init = '2017-10-25/20-04-38/BillCoish_experimentAllXY_sequence'
#location = '2017-12-12/17-47-41/RB_experimentAllXY_sequence'

#%%
ds = load_data(location = location_3init, io = IO, formatter = formatter)



#%%     average trace

x = ds.index3_set[0,0,0,0,:]
i = 0
y = ds.raw_data[:,0,i,:,:].mean(axis = (0,1))
y = ds.raw_data[10,0,i,:,:].mean(axis = 0)
y = ds.raw_data[10,0,i,58,:]
#%%
'''
Qubit = 2
i = 0 if Qubit == 2 else 1

fitting_point = 59
x = np.linspace(1e-6, 25e-3, 60)[:fitting_point]
y = ds.probability_data[:,i,:fitting_point].mean(axis = 0)

pars, pcov = curve_fit(T1_fitting, x, y,)
'''
#%%

pt = MatPlot()
pt.add(x = x, y = y)
#print('T1 is: ', pars[0], 'ms')