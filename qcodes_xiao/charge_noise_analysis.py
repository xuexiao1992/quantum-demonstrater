# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:04:41 2017

@author: X.X
"""

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

def RB_Fidelity(m, p, A, B,):
    
    return A*(p**(m))+B

def sequence_decay(t, A, B, T):
    
    return A*(np.e**(-t/T))+B

def Exp_Sin_decay(t, A, B, F, Phase, T):
    
    return A*(np.e**(-t/T))*np.sin(F*t+Phase)+B
#%%
IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')
formatter = HDF5FormatMetadata()

'''
data operated with Hardmard gate in 2 qubit space
'''
location1 = '2018-01-09/11-35-49/RB_experimentAllXY_sequence' 
location2 = '2018-01-09/15-00-49/RB_experimentAllXY_sequence'
location3 = '2018-01-09/17-00-07/RB_experimentAllXY_sequence'
location4 = '2018-01-10/16-22-55/RB_experimentAllXY_sequence'

location5 = '2018-01-16/14-39-54/RB_experimentAllXY_sequence'

'''
data operated with Hardmard gate in 2 qubit space
'''
location6 = '2018-01-16/15-26-33/RB_experimentAllXY_sequence'
location7 = '2018-01-16/15-38-47/RB_experimentAllXY_sequence'


location8 = '2018-01-16/18-48-54/RB_experimentAllXY_sequence'

location9 = '2018-01-17/13-51-14/RB_experimentAllXY_sequence'

location11 = '2018-01-17/16-36-17/RB_experimentAllXY_sequence'

location = location11

ds = load_data(location = location, io = IO, formatter = formatter)
#DS = load_data(location = location1, io = IO)

#%%
'''
ds = DS
Qubit = 2
i = 0 if Qubit == 2 else 1

sweep_point = 31

ramsey_point = 11

fitting_point = 18

x = np.linspace(1,fitting_point,fitting_point)

y = ds.probability_data[:,i,ramsey_point:ramsey_point+fitting_point].mean(axis = 0)

pars, pcov = curve_fit(RB_Fidelity, x, y,)
'''
#%%

Exp1 = '00'
Exp2 = '00'

def average_two_qubit(ds):
    
    counts = len(ds.singleshot_data)
 
    fitting_num = 41
    sweep_num = 41
    
    experiment = 2
    
    data = np.ndarray(shape = (experiment, counts, fitting_num, 100,))
    
    for seq in range(counts):
        for exp in range(experiment):
            for i in range(11,11+exp*sweep_num+fitting_num):
                for j in range(100):
                    if exp == 0:
                        if ds.singleshot_data[seq][0][i][j] == int(Exp1[1]) and ds.singleshot_data[seq][1][i][j] == int(Exp1[0]):
                            data[exp][seq][i-11-exp*sweep_num][j] = 1
                        else:
                            data[exp][seq][i-11-exp*sweep_num][j] = 0
                    elif exp == 1:
                        if ds.singleshot_data[seq][0][i][j] == int(Exp2[1]) and ds.singleshot_data[seq][1][i][j] == int(Exp2[0]):
                            data[exp][seq][i-11-exp*sweep_num][j] = 1
                        else:
                            data[exp][seq][i-11-exp*sweep_num][j] = 0
    
    average = data.mean(axis = 3)
    average_11 = average.mean(axis = 1)
    
    return average_11

average = average_two_qubit(ds)

#%%

fitting_num = 41
time_range = 1.2e-6
x = np.linspace(0, fitting_num-1, fitting_num)
t = np.linspace(0, time_range, fitting_num)

#pars1, pcov = curve_fit(sequence_decay, x, average[0],)
pars1, pcov = curve_fit(Exp_Sin_decay, x, average[0], bounds = ((0, -inf, ), (1,x[-1])))

pt = MatPlot()
pt.add(x = t, y = average[0], xlabel = 'dephasing_time', ylabel = 'probability |11>')
pt.add(x = t, y = sequence_decay(x,pars1[0],pars1[1],pars1[-1]))
pt.add(x = t, y = Exp_Sin_decay(x,pars1[0],pars1[1],pars1[2],pars1[3],pars1[4]), )

print('T2* in exp1:', pars1[-1]/(fitting_num-1)*1.2, 'us')

#pars2, pcov = curve_fit(sequence_decay, x, average[1],)
pars2, pcov = curve_fit(Exp_Sin_decay, x, average[1],)

pt2 = MatPlot()
pt2.add(x = t, y = average[1], xlabel = 'dephasing_time', ylabel = 'probability |01>')
pt2.add(x = t, y = sequence_decay(x,pars2[0],pars2[1],pars2[-1]))
pt2.add(x = t, y = Exp_Sin_decay(x,pars2[0],pars2[1],pars2[2],pars2[3],pars2[4]))

print('T2* in exp2:', pars2[-1]/(fitting_num-1)*1.2, 'us')



