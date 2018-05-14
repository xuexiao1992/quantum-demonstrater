# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:25:05 2018

@author: X.X
"""

#%%

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.path as mpath
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
from scipy.optimize import curve_fit

#%%

IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')
formatter = HDF5FormatMetadata()


#%%

location = '2017-11-04/11-06-25/RB_experimentAllXY_sequence'  # slow decay

location = '2017-11-06/13-12-43/RB_experimentAllXY_sequence'  # with fast decay

location = '2017-11-06/11-53-47/RB_experimentAllXY_sequence'  # with fast decay

ds = load_data(location = location, io = IO, formatter = formatter)



#%%

def Func_Sin(t,A,F,P,B):
    return A*np.sin(2*np.pi*F*t + P) + B

def Gaussian_Sin(t, A, F, P, B, T):
    
    return A*(np.exp(-(t/T)**2))*np.sin(2*3.14*F*t+P)+B

def Exp_Sin(t, A, F, P, B, T):
    
    return A*(np.exp(-(t/T)))*np.sin(2*3.14*F*t+P)+B
#%%

'''
for Rabi oscillation
'''

start_point =11
x = np.linspace(0,3e-6,61)
y = ds.probability_data[:,0,start_point:].mean(axis = 0)



#%%


Fit = 'G'

if Fit == 'N':
    pars, pcov = curve_fit(Func_Sin, x, y,
                           p0 = (0.3, 1.5e6, 0, 0.3),
                           bounds = ((0.2, 1.1e6, -np.pi, -1),(0.5, 2.5e6, np.pi, 1)))
    
    y_fit = Func_Sin(x, pars[0], pars[1], pars[2], pars[3])

elif Fit == 'G':
    pars, pcov = curve_fit(Gaussian_Sin, x, y,
                           p0 = (0.3, 1.5e6, 0, 0.3, 2e-6),
                           bounds = ((0.2, 1e6, -np.pi, -1, 0),(0.5, 2.5e6, np.pi, 1, 10e-6)))
    
    y_fit = Gaussian_Sin(x, pars[0], pars[1], pars[2], pars[3], pars[4])


elif Fit == 'E':
    pars, pcov = curve_fit(Exp_Sin, x, y,
                           p0 = (0.2, 1.5e6, 0, 0.3, 2e-6),
                           bounds = ((0.3, 1e6, -np.pi, -1, 0),(0.5, 2.5e6, np.pi, 1, 10e-6)))
    
    y_fit = Exp_Sin(x, pars[0], pars[1], pars[2], pars[3], pars[4])


pt1 = MatPlot()
#pt1.add_to_plot(x = x, y = y, fmt='bs')
#pt1.add_to_plot(x = x, y = y_fit, fmt = 'r--')#fmt='*')

pt1.add(x = x*1e6, y = y, fmt='bs', xlabel = 'Burst Time', ylabel = 'Probability |1>', xunit = 'us', yunit = '%')
pt1.add(x = x*1e6, y = y_fit, fmt = 'r--',)# xlabel = 'Clifford Numbers', ylabel = 'probability |1>')#fmt='*')


#%%
'''
star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)


plt.plot(np.arange(10)**2, '--r', marker=cut_star, markersize=15)

plt.show()
'''
#%%
'''

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
y2 = y + 0.1 * np.random.normal(size=x.shape)

fig, ax = plt.subplots()
ax.plot(x, y, 'k--')
ax.plot(x, y2, 'ro')

# set ticks and tick labels
ax.set_xlim((0, 2*np.pi))
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', '$\pi$', '2$\pi$'])
ax.set_ylim((-1.5, 1.5))
ax.set_yticks([-1, 0, 1])

# Only draw spine between the y-ticks
ax.spines['left'].set_bounds(-1, 1)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()
'''