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

def sequence_decay(m, A, B, M):
    
    return A*(np.e**(-m/M))+B
#%%
IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents\\RB_experiment')
formatter = HDF5FormatMetadata()
location1 = '2017-10-27/17-42-30/RB_experimentAllXY_sequence'
location2 = '2017-10-28/15-16-42/RB_experimentAllXY_sequence'
location3 = '2017-10-31/15-07-12/RB_experimentAllXY_sequence'
location4 = '2017-10-30/13-01-20/RB_experimentAllXY_sequence'

location5 = '2017-10-31/18-36-10/RB_experimentAllXY_sequence'   # Q2
location6 = '2017-11-01/14-11-19/RB_experimentAllXY_sequence'   # Q1

location7 = '2017-11-01/17-21-09/RB_experimentAllXY_sequence'
location8 = '2017-11-01/19-46-17/RB_experimentAllXY_sequence' # simultaneous RB


location9 = '2017-11-03/16-40-00/RB_experimentAllXY_sequence'

location10 = '2017-11-03/20-35-34/RB_experimentAllXY_sequence'
location11 = '2017-11-04/03-58-33/RB_experimentAllXY_sequence'


location12 = '2017-11-07/00-58-57/RB_experimentAllXY_sequence'
location12 = '2017-11-07/21-21-27/RB_experimentAllXY_sequence'

location13 = '2017-11-08/12-29-49/RB_experimentAllXY_sequence'
location14 = '2017-11-08/13-30-25/RB_experimentAllXY_sequence'
location15 = '2017-11-08/17-51-57/RB_experimentAllXY_sequence'



location16 = '2017-11-09/10-30-58/RB_experimentAllXY_sequence'
location16 = '2017-11-09/14-50-11/RB_experimentAllXY_sequence'
location17 = '2017-11-09/16-02-47/RB_experimentAllXY_sequence'
location18 = '2017-11-09/17-35-19/RB_experimentAllXY_sequence'
location19 = '2017-11-09/19-37-27/RB_experimentAllXY_sequence'

location20 = '2017-11-09/22-27-11/RB_experimentAllXY_sequence'


location21 = '2017-11-10/00-09-24/RB_experimentAllXY_sequence'
location22 = '2017-11-10/01-53-00/RB_experimentAllXY_sequence'
location23 = '2017-11-10/11-45-13/RB_experimentAllXY_sequence'
location24 = '2017-11-10/13-30-08/RB_experimentAllXY_sequence'
location25 = '2017-11-10/15-03-46/RB_experimentAllXY_sequence'

location26 = '2017-11-10/17-00-57/RB_experimentAllXY_sequence'# sequential
location27 = '2017-11-11/14-58-59/RB_experimentAllXY_sequence'# sequential with no interleave
location28 = '2017-11-11/17-13-19/RB_experimentAllXY_sequence'# simultaneous with no interleave
location29 = '2017-11-11/21-30-12/RB_experimentAllXY_sequence'# simultaneous with no interleave

location99 = '2017-11-12/17-07-04/RB_experimentAllXY_sequence'# simultaneous with interleave
location100 = '2017-11-12/21-43-54/RB_experimentAllXY_sequence'# simultaneous with interleave
location101 = '2017-11-12/23-40-02/RB_experimentAllXY_sequence'# simultaneous with interleave

location102 = '2017-11-13/15-34-14/RB_experimentAllXY_sequence'# simultaneous with no interleave

location_a = '2017-11-13/18-47-23/RB_experimentAllXY_sequence'# simultaneous with no interleave
location_b = '2017-11-13/21-29-25/RB_experimentAllXY_sequence'# simultaneous with no interleave

location_c = '2017-11-14/14-25-46/RB_experimentAllXY_sequence'# simultaneous with interleave

#%%

location_1 = '2017-11-15/22-53-31/RB_experimentAllXY_sequence'  # 80.1 Q2  87.8 Q1
location_2 = '2017-11-16/00-45-47/RB_experimentAllXY_sequence'  #85.5 Q2   84 Q1

location_3 = '2017-11-16/16-05-05/RB_experimentAllXY_sequence'  #85.15 Q2   875 Q1

location_4 = '2017-11-16/18-42-49/RB_experimentAllXY_sequence'  #85.13 Q2   885 Q1

location_5 = '2017-11-16/20-22-33/RB_experimentAllXY_sequence'  #82.15 Q2   8705 Q1

location_6 = '2017-11-16/22-52-43/RB_experimentAllXY_sequence'  #82.6 Q2   865 Q1


#%%

location_x = '2017-11-23/00-45-24/RB_experimentAllXY_sequence'
location_y = '2017-11-23/10-00-08/RB_experimentAllXY_sequence'
location_z = '2017-11-25/10-26-51/RB_experimentAllXY_sequence' #80%

location_xx = '2017-11-25/11-30-33/RB_experimentAllXY_sequence' #81%
location_xy = '2017-11-25/16-07-59/RB_experimentAllXY_sequence' #80%
location_xzz = '2017-11-25/17-14-58/RB_experimentAllXY_sequence' #80%

location_xz = '2017-11-25/23-01-05/RB_experimentAllXY_sequence' #80%

location_xyy = '2017-11-26/01-55-19/RB_experimentAllXY_sequence' #80%
location_xyy = '2017-11-26/03-27-08/RB_experimentAllXY_sequence' #80%

location_te = '2017-12-12/12-02-59_DAC_V_sweep'

DS = load_data(location = location_4, io = IO, formatter = formatter)
#DS = load_data(location = location1, io = IO)

#%%

ds = DS
Qubit = 2
i = 0 if Qubit == 2 else 1
ramsey_point = 11
fitting_point = 18
x = np.array([len(clifford_sets[0][i]) for i in range(fitting_point)])
y = ds.probability_data[:,i,ramsey_point:ramsey_point+fitting_point].mean(axis = 0)

pars, pcov = curve_fit(RB_Fidelity, x, y,)

#%%

pt = MatPlot()
pt.add(x = x, y = RB_Fidelity(x,pars[0],pars[1],pars[2]), xlabel = 'Clifford Numbers', ylabel = 'probability |1>')
pt.add(x = x,y = ds.probability_data[:,i,11:11+fitting_point].mean(axis = 0))
#pt.add_to_plot(xlabel = 'Clifford Numbers')
#%%
#pt1 = MatPlot()
#pt1.add(z=ds.probability_data[:,i,11:11+fitting_point])
#%%
fidelity = 1-(1-pars[0])/2
print('fidelity is: ', fidelity)
##
#%%
'''
pt = MatPlot()
y = ds.probability_data[:,i,11:11+fitting_point].
'''
#%% load data
#data_set_2 = DataSet(location = test_location, io = NewIO,)
#data_set_2.read()

#raw_data_set = load_data(location = new_location, io = NewIO,)
#%%

'''
def average_two_qubit(ds):
    seq_num = len(ds.singleshot_data)
    fitting_num = fitting_point
#    clifford_num = len(ds.singleshot_data[0][0])
    data = np.ndarray(shape = (50, fitting_num, 100,))
    for seq in range(seq_num):
        for i in range(11,11+fitting_num):
            for j in range(100):
                if ds.singleshot_data[seq][0][i][j] == 1 and ds.singleshot_data[seq][1][i][j] == 1:
                    data[seq][i-11][j] = 1
                else:
                    data[seq][i-11][j] = 0
    
    average = data.mean(axis = 2)
    average_11 = average.mean(axis = 0)
    return average_11

average = average_two_qubit(ds)
pars, pcov = curve_fit(RB_Fidelity, x, average,)
fidelity = 1-(1-pars[0])*3/4
print('fidelity is: ', fidelity)
pt = MatPlot()
pt.add(x = x, y = average)
pt.add(x = x, y = RB_Fidelity(x,pars[0],pars[1],pars[2]))
'''