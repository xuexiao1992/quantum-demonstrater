# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:10:45 2017

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

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
#    x_new = x/1e6
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def sequence_decay(m, A, B, M):
    
    return A*(np.e**(-m/M))+B

def T1_fitting(x, t0, A, B):
    
    return A*np.e**(-x/t0)+B

#%%
IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')
formatter = HDF5FormatMetadata()

#%%
location = '2017-12-12/16-57-48/RB_experimentAllXY_sequence'
location2 = '2017-12-12/17-28-29/RB_experimentAllXY_sequence'
location3 = '2017-12-12/17-47-41/RB_experimentAllXY_sequence'

ds = load_data(location = location3, io = IO, formatter = formatter)


#%%

Qubit = 2
i = 0 if Qubit == 2 else 1

fitting_point = 59
x = np.linspace(1e-6, 25e-3, 60)[:fitting_point]
y = ds.probability_data[:,i,:fitting_point].mean(axis = 0)

pars, pcov = curve_fit(T1_fitting, x, y,)

#%%

pt = MatPlot()
pt.add(x = x, y = T1_fitting(x,pars[0],pars[1],pars[2]))
pt.add(x = x,y = ds.probability_data[:,i,:fitting_point].mean(axis = 0))
print('T1 is: ', pars[0], 'ms')