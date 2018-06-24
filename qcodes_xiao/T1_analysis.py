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

def T2_star_fitting(t, A, B, F, Phase, T):
    
    return A*(np.e**(-(t/T)**2))*np.sin(2*3.14*F*t+Phase)+B


def T2_star_fitting2(t, A, B, T):
    
    return A*(np.e**(-(t/T)**2))+B
#    return A*(np.exp(-(t/T)**2))*+B

#%%
IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')
formatter = HDF5FormatMetadata()


IO_new = DiskIO(base_location = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\Data\\RB_experiment')


#%%
location = '2017-12-12/16-57-48/RB_experimentAllXY_sequence'
location2 = '2017-12-12/17-28-29/RB_experimentAllXY_sequence'
location3 = '2017-12-12/17-47-41/RB_experimentAllXY_sequence'

location3 = '2018-03-23/14-41-59/RB_experimentAllXY_sequence'
location3 = '2018-03-23/17-16-26/RB_experimentAllXY_sequence'

location3 = '2018-03-23/17-39-47/RB_experimentAllXY_sequence'

location3 = '2018-03-24/14-47-41/RB_experimentAllXY_sequence'
location3 = '2018-03-24/15-08-20/RB_experimentAllXY_sequence'
location3 = '2018-03-24/17-33-21/RB_experimentAllXY_sequence'
#location3 = '2018-03-24/19-54-21/RB_experimentAllXY_sequence'
location3 = '2018-03-26/11-27-04/RB_experimentAllXY_sequence'
location3 = '2018-03-28/16-07-43/RB_experimentAllXY_sequence'
location3 = '2018-03-28/16-40-41/RB_experimentAllXY_sequence'

location3 = '2018-04-05/11-37-02/RB_experimentAllXY_sequence'

location3 = '2018-04-05/14-48-44/RB_experimentAllXY_sequence'
#location3 = '2018-04-05/15-21-16/RB_experimentAllXY_sequence'
location3 = '2018-04-05/16-50-37/RB_experimentAllXY_sequence'

location3 = '2018-04-05/20-26-26/RB_experimentAllXY_sequence'
location3 = '2018-04-05/20-55-38/RB_experimentAllXY_sequence'
location3 = '2018-04-06/10-55-25/RB_experimentAllXY_sequence'
location3 = '2018-04-06/11-26-17/RB_experimentAllXY_sequence'

location3 = '2018-04-06/19-53-52/RB_experimentAllXY_sequence'



location3 = '2018-06-18/14-01-15/RB_experimentAllXY_sequence'

location3 = '2018-06-18/14-19-28/RB_experimentAllXY_sequence'

location3 = '2018-06-18/14-50-55/RB_experimentAllXY_sequence'

#location3 = '2018-04-04/18-38-25/RB_experimentAllXY_sequence'

ds = load_data(location = location3, io = IO_new, formatter = formatter)


#%%

Qubit = 2
i = 0 if Qubit == 2 else 1

fitting_point = 51
start_point = 0


'''
#x = np.linspace(1e-6, 25e-3, 60)[:fitting_point]
x = np.linspace(0, 1.5e-6, 31)[:fitting_point]
y = ds.probability_data[:,i,start_point:start_point+fitting_point].mean(axis = 0)

pars, pcov = curve_fit(T1_fitting, x, y,)

#pars, pcov = curve_fit(T1_fitting, x, y, 
#                        p0 = (0.01, 0.2, 0.15),
#                        bounds = ((0,-np.inf,0),(0.1,np.inf,0.3)))

pars, pcov = curve_fit(T1_fitting, x, y, 
                        p0 = (0.01, -0.2, 0.15),
                        bounds = ((0,-np.inf,-1),(0.1,np.inf,1)))
'''

#%%

fitting_point = 51

x = np.linspace(0, 1.5e-6, fitting_point)[:fitting_point]
y = ds.probability_data[:,i,0:fitting_point].mean(axis = 0)

pars, pcov = curve_fit(T2_star_fitting, x, y,
                       p0 = (0.8, 0.3, 4e6, 0, 0.5e-6),
                       bounds = ((0.25,-np.inf,1e6,-np.inf,0),(2,np.inf,10e6,np.inf,5)))


#pars, pcov = curve_fit(T2_star_fitting2, x, y,
#                       p0 = (0.8, 0.3, 0.5e-6),
#                       bounds = ((0.25,-np.inf,0),(2,np.inf,8e-6)))

#%%

pt = MatPlot()
#pt.add(x = x, y = T1_fitting(x,pars[0],pars[1],pars[2]))
pt.add(x = x, y = T2_star_fitting(x,pars[0],pars[1],pars[2],pars[3],pars[4]))
#pt.add(x = x, y = T2_star_fitting2(x,pars[0],pars[1],pars[2]))
pt.add(x = x,y = ds.probability_data[:,i,start_point:start_point+fitting_point].mean(axis = 0))
print('T1 is: ', pars[0]*1000, 'ms')
#pt1 = MatPlot()
#pt1.add_to_plot(x = x, y = T1_fitting(x,pars[0],pars[1],pars[2]),fmt='*')
#%%


