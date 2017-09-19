# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:20:20 2017

@author: think
"""

import numpy as np
import qcodes as qc
from qcodes.loops import Loop, ActiveLoop
import numpy as np
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.io import DiskIO
from qcodes.instrument.parameter import ManualParameter, StandardParameter, ArrayParameter
from qcodes.utils.validators import Numbers
from functools import partial
#%%
aa = 5
aaa=4
def Pfunction(a):
    global aaa
    
    aaa = a + 5
    
    return aaa

def Ffunction():
    global aa
    a=0
    a = a +5
    b =3
    aa += a*b
    return aa

def QFunction():
    a = F.get_latest() - 15
    return a



P = StandardParameter(name = 'Para1', set_cmd = Pfunction)

F = StandardParameter(name = 'Fixed1', get_cmd = Ffunction)

Q = StandardParameter(name = 'Para2', set_cmd = Pfunction, get_cmd = QFunction)

E = StandardParameter(name = 'Fixed2', get_cmd = QFunction)

Sweep_Value = P[1:5.5:0.5]

Sweep_2 = Q[2:10:1]

LP = Loop(sweep_values = Sweep_Value).loop(sweep_values = Sweep_2).each(F, E)

#LP = Loop(sweep_values = Sweep_Value).each(F)

print('loop.data_set: %s' % LP.data_set)

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()

OldIO = DiskIO(base_location = 'D:\\文献\\QuTech\\QTlab\\xiaotest\\testIO')

## get_data_set should contain parameter like io, location, formatter and others
data = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, 
                       io = NewIO, formatter = formatter)
#data = LP.get_data_set(data_manager=False, location=None, loc_record = {'name':'T1', 'label':'T_load_sweep'})
print('loop.data_set: %s' % LP.data_set)

DS = new_data(location = 'aaaaaaaaaa',io = NewIO)

def live_plotting():
    for para in data.arrays:
        DS.arrays[para] = data.arrays[para]
    return DS
DS = live_plotting()
#def add_T1exp_metadata(data):
#        
#        data.metadata['Parameters'] = {'Nrep': 10, 't_empty': 2, 't_load': 2.4, 't_read': 2.2}
#        data.write(write_metadata=True)
#
#
#add_T1exp_metadata(data)

#datatata = LP.run(background=False)
#%%



gate = ManualParameter('gate', vals=Numbers(-10, 10))
frequency = ManualParameter('frequency', vals=Numbers(-10, 10))
amplitude = ManualParameter('amplitude', vals=Numbers(-10, 10))
# a manual parameter returns a value that has been set
# so fix it to a value for this example
amplitude.set(-1)

combined = qc.combine(gate, frequency, name="gate_frequency")
combined.__dict__.items()

a = [1,2,3]
b = [c for c in a]
b = {'ele_%d'%i: i for i in a}
b = {'ele_{}'.format(i): i for i in a}
#
#Sweep = Loop(sweep_values = [1,2,3,4,5,6,8,77,32,44,564])
#
#a= np.array([5,6])
#
#lista = {}
#
#aa = np.array([1,2,3,4,5])
#
#bb = np.array([44,55,33,22,77])
#%%

from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot
from mpldatacursor import datacursor
import numpy as np
#%%

NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
formatter = HDF5FormatMetadata()
try_location = '2017-09-04/17-23-05Finding_ResonanceRabi_Sweep'

DS = load_data(location = try_location, io = NewIO,)

DS_P = convert_to_probability(DS, 0.025)

DS_new = new_data(location = try_location, io = NewIO,)


x_data = np.linspace(1,10,10)
y_data = np.linspace(11,20,10)
#z_data = np.linspace(101,201,101)

Mplot = MatPlot(x_data,y_data)
Qplot = QtPlot(x_data,y_data)

Mplot = MatPlot()

config = {
        'x': np.linspace(1,20,20),
        'y': np.linspace(11,30,20)
        }
#Mplot.traces[0]['config'] = config

data = np.array([1,2,3])
data1 = np.array([[1,2,33,5],[5,232,7,3],[1,2,3,4]])

data_array1 = DataArray(preset_data = data, name = 'digitizer', is_setpoint = True)

data_array2 = DataArray(preset_data = data, name = 'digitizer2')
data_set = new_data(arrays=arrays3, location=try_location, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = NewIO,)

#Mplot.add_updaters(updater = , plot_config = )
#%%

#
#def test(x,y,z):
#    
#    global lista
#    
#    y = x+y
#    z = z*y
#    x = z-x
#    
#    lista['num_%d'%x] = x+y+z
#    
#    print(x)
#    
#    return True
#
#c = np.matrix
#
#def func(x,y):
#    
#    global lista
#    
#    c = x
#    y = c+y
#    
#    lista.append(y)
#    return y
#    
#    
#def haha(x,y):
#    a = x
#    b=a*y
#    c = func(a,b)
#    return c


def funca(a,**kw):
    step = {'a':a}
    step.update(kw)
    return step

def funcb(b, **kw):
    step = funca(a = b, **kw)
    return step

#%%


from scipy.optimize import curve_fit

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, sigma):
    x_new = x/1e6
    return a*np.exp(-(x_new-x0)**2/(2*sigma**2))

pars, pcov = curve_fit(Func_Sin, x, y,)
