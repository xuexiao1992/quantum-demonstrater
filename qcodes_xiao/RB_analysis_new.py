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


import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 15
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
IO = DiskIO(base_location = 'D:\\Data\\RB_experiment')
formatter = HDF5FormatMetadata()


IO_K = DiskIO(base_location = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\Data\\RB_experiment')

IO_K2 = DiskIO(base_location = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\Data\\RB_experiment2')


'''
character benchmarking
'''

location_new2 = '2018-07-03/12-43-46/RB_experimentAllXY_sequence'

location_new2 = '2018-07-05/14-41-58/RB_experimentAllXY_sequence'


DS = load_data(location = location_new2, io = IO_K, formatter = formatter)

#%% character benchmarking without normalization

ds = DS
fitting_points = 24
#seq_rep_num = 20
seq_rep_num = int(DS.sequence_number_set.shape[0]/16)
sequence_number = 16*seq_rep_num
repetition = 50
init_state = ['00', '01','10', '11',]

x = np.array([len(clifford_sets[0][i]) for i in range(fitting_points)])

#def average_two_qubit(ds)
data = {}
average = {}
for init in init_state:
    data[init] = np.ndarray(shape = (int(sequence_number/4), fitting_points, repetition,))

for j in range(sequence_number):
    j_new = j%seq_rep_num
    
    Pauli_index = j//seq_rep_num
    Pauli_index = j%16
    
    init_index = Pauli_index//4
    
    for i in range(11, 11+fitting_points):
        for k in range(repetition):
            if ds.singleshot_data[j][0][i][k] == 0 and ds.singleshot_data[j][1][i][k] == 0:
                data[init_state[init_index]][j_new][i-11][k] = 1
            else:
                data[init_state[init_index]][j_new][i-11][k] = 0
for init in init_state:
    average_probability = data[init].mean(axis = 2)
    average[init] = average_probability.mean(axis = 0)
    
P1 = average['00'] - average['01'] + average['10'] - average['11'] 
P2 = average['00'] + average['01'] - average['10'] - average['11']
P3 = average['00'] - average['01'] - average['10'] + average['11']

pars1, pcov1 = curve_fit(RB_Fidelity, x, P1, p0 = (0.9, 0.2, 0), bounds = ((0.7, 0, 0),(1, 0.8, 0.001)))
pars2, pcov2 = curve_fit(RB_Fidelity, x, P2, p0 = (0.9, 0.2, 0), bounds = ((0.7, 0, 0),(1, 0.8, 0.001)))
pars3, pcov3 = curve_fit(RB_Fidelity, x, P3, p0 = (0.9, 0.2, 0), bounds = ((0.7, 0, 0),(1, 0.8, 0.001)))
Fidelity = 3/15 * (pars1[0] + pars2[0]) + 9/15 * pars3[0]

#%%


import matplotlib as mpl

#plot_point = fitting_points
plot_points = fitting_points

pt = MatPlot()
pt.add(x = x[:plot_points],y = P1[:plot_points], fmt = 'rp',xlabel = 'Clifford Numbers', ylabel = '$P_{|1>}$', xunit = 'N', yunit = '%')
pt.add(x = x[:plot_points],y = P2[:plot_points], fmt = 'bp',xlabel = 'Clifford Numbers', ylabel = '$P_{|1>}$', xunit = 'N', yunit = '%')
pt.add(x = x[:plot_points],y = P3[:plot_points], fmt = 'gp',xlabel = 'Clifford Numbers', ylabel = '$P_{|1>}$', xunit = 'N', yunit = '%')
pt.add(x = x[:plot_points], y = RB_Fidelity(x,pars1[0],pars1[1],pars1[2])[:plot_points],fmt = 'r--', )
pt.add(x = x[:plot_points], y = RB_Fidelity(x,pars2[0],pars2[1],pars2[2])[:plot_points],fmt = 'b--', )
pt.add(x = x[:plot_points], y = RB_Fidelity(x,pars3[0],pars3[1],pars3[2])[:plot_points],fmt = 'g--', )


#%%     character benchmarking with normalization

P_pi_1 = 0.98
P_


#%%

ds = DS
Qubit = 1
i = 0 if Qubit == 2 else 1
ramsey_point = 11
fitting_point = 24
x = np.array([len(clifford_sets[0][i]) for i in range(fitting_point)])
#x = x-1
#x[0] = 0
y = ds.probability_data[:,i,ramsey_point:ramsey_point+fitting_point].mean(axis = 0)

#y = 1-y

pars, pcov = curve_fit(RB_Fidelity, x, y,
                       p0 = (0.9, 0.2, 0.4),
                       bounds = ((0.7, 0, 0),(1, 0.8, 0.8)))
#                       bounds = ((), ()))
#pars, pcov = curve_fit(RB_Fidelity, x, y,
#                       p0 = (-0.9, 0.2, -0.4),
#                       bounds = ((-1, 0, -1),(-0.5, 0.8, 0)))


#%%
plot_point = fitting_point
pt = MatPlot()
pt.add(x = x[:plot_point],y = ds.probability_data[:,i,11:11+fitting_point].mean(axis = 0)[:plot_point], fmt = 'bp',xlabel = 'Clifford Numbers', ylabel = '$P_{|1>}$', xunit = 'N', yunit = '%')
pt.add(x = x[:plot_point], y = RB_Fidelity(x,pars[0],pars[1],pars[2])[:plot_point],fmt = 'r--', )
#pt.add_to_plot(xlabel = 'Clifford Numbers')
#%%
#pt1 = MatPlot()
#pt1.add(z=ds.probability_data[:,i,11:11+fitting_point])

ds2 = DS2
x2 = np.array([len(clifford_sets[0][i]) for i in range(fitting_point)])
y2 = ds2.probability_data[:,i,ramsey_point:ramsey_point+fitting_point].mean(axis = 0)

#y2 = 1-y2

pars2, pcov2 = curve_fit(RB_Fidelity, x2, y2,
                         p0 = (0.95, 0.2, 0.4),
                         bounds = ((0.7, 0, 0),(1, 0.8, 0.8)))

#%%
plot_point = fitting_point
offset = 0.005
#pt = MatPlot()
pt.add(x = x2[:plot_point],y = ds2.probability_data[:,i,11:11+fitting_point].mean(axis = 0)[:plot_point]+offset, fmt = 'rp',)#xlabel = 'Clifford Numbers', ylabel = 'probability |1>', xunit = 'N', yunit = '%')
pt.add(x = x2[:plot_point], y = RB_Fidelity(x2,pars2[0],pars2[1],pars2[2])[:plot_point]+offset,fmt = 'r--', )

#%%
fidelity = 1-(1-pars[0])/2
print('fidelity is: ', fidelity)
##
#%%
'''
pt = MatPlot()
y = ds.probability_data[:,i,11:11+fitting_point].
'''

#%%

def interleave_fidelity(F_standard, F_interleave):
    
    pars_standard = 2*F_standard-1
    pars_interleave = 2*F_interleave-1
    fidelity = 1-(1-pars_interleave/pars_standard)/2
    
    return fidelity
#%% load data
#data_set_2 = DataSet(location = test_location, io = NewIO,)
#data_set_2.read()

#raw_data_set = load_data(location = new_location, io = NewIO,)
#%%
#DS = load_data(location = location_xx, io = IO, formatter = formatter)


P_pi_q1 = 0.98
P_pi_q2 = 0.97

P_init_q1 = 1#0.99
P_init_q2 = 1#0.99

fitting_point = 10

def average_normalized_two_qubit(ds):
    
    seq_num = len(ds.singleshot_data)-10   # sequence_num e.g. Count, RB sequences
    
    fitting_num = fitting_point         # sweep points e.g. RB elements numbers
    
    data_11 = np.zeros(shape = (seq_num, fitting_num, 100,))
    data_01 = np.zeros(shape = (seq_num, fitting_num, 100,))
    data_10 = np.zeros(shape = (seq_num, fitting_num, 100,))
    data_00 = np.zeros(shape = (seq_num, fitting_num, 100,))
    
#    data = np.zeros(shape = (4, seq_num, fitting_num, 100,))
    
    
    Fidelity_read = np.ndarray(shape = (seq_num, 2, 2))

    for seq in range(seq_num):
        
        for i in range(11,11+fitting_num):
            for j in range(100):
                
                if ds.singleshot_data[seq][0][i][j] == 1 and ds.singleshot_data[seq][1][i][j] == 1:
                    data_11[seq][i-11][j] = 1
                    
                elif ds.singleshot_data[seq][0][i][j] == 1 and ds.singleshot_data[seq][1][i][j] == 0:
                    data_01[seq][i-11][j] = 1
                
                elif ds.singleshot_data[seq][0][i][j] == 0 and ds.singleshot_data[seq][1][i][j] == 1:
                    data_10[seq][i-11][j] = 1
                
                elif ds.singleshot_data[seq][0][i][j] == 0 and ds.singleshot_data[seq][1][i][j] == 0:
                    data_00[seq][i-11][j] = 1
               
    
    average_11 = data_11.mean(axis = 2)
    average_01 = data_01.mean(axis = 2)
    average_10 = data_10.mean(axis = 2)
    average_00 = data_00.mean(axis = 2)
    
    average_normalized = np.zeros(shape = (seq_num, fitting_num,))
    
    for seq in range(seq_num):
        
        P2_q2 = ds.probability_data[seq][0][-2]
        P3_q2 = ds.probability_data[seq][0][-1]
        P2_q1 = ds.probability_data[seq][1][-2]
        P3_q1 = ds.probability_data[seq][1][-1]
        
        a_q2 = np.array([[-P_init_q2, 1-P_init_q2], [P_init_q2-1, P_init_q2]])
        
        b_q2 = np.array([P2_q2-P_init_q2, P3_q2/P_pi_q2-(1-P_init_q2)])
        
        F_R_q2 = np.linalg.solve(a_q2, b_q2)
        
        a_q1 = np.array([[-P_init_q1, 1-P_init_q1], [P_init_q1-1, P_init_q1]])
        
        b_q1 = np.array([P2_q1-P_init_q1, P3_q1/P_pi_q1-(1-P_init_q1)])
        
        F_R_q1 = np.linalg.solve(a_q1, b_q1)
        
#        Fidelity_read[seq][0][0] = F_R_q2[0]
#        Fidelity_read[seq][0][1] = F_R_q2[1]
#        Fidelity_read[seq][1][0] = F_R_q1[0]
#        Fidelity_read[seq][1][1] = F_R_q1[1]

        F_R_q2[0] = 1-P2_q2
        F_R_q2[1] = P3_q2/P_pi_q2
        F_R_q1[0] = 1-P2_q1
        F_R_q1[1] = P3_q1/P_pi_q1
        
        Fidelity_1 = np.array([[F_R_q1[0], 1-F_R_q1[1]],[1-F_R_q1[0], F_R_q1[1]]])
        Fidelity_2 = np.array([[F_R_q2[0], 1-F_R_q2[1]],[1-F_R_q2[0], F_R_q2[1]]])
        
        print('Fidelity1:', Fidelity_1)
        print('Fidelity2:', Fidelity_2)
        
        Fidelity = np.kron(Fidelity_1, Fidelity_2)
        
        for i in range(fitting_num):
            
            average_raw = np.array([[average_00[seq][i]], [average_01[seq][i]], [average_10[seq][i]], [average_11[seq][i]]])
            
            average_normalized[seq][i] = np.linalg.multi_dot([np.linalg.inv(Fidelity), average_raw])[-1]
    
    print('average shape:', average_normalized.shape)
    average_11 = average_normalized.mean(axis = 0)
    
    return average_11


#%%

#DS = load_data(location = location_xx, io = IO, formatter = formatter)

#fitting_point = 10
def average_two_qubit(ds):
    seq_num = len(ds.singleshot_data)
    fitting_num = fitting_point
#    clifford_num = len(ds.singleshot_data[0][0])
    data = np.ndarray(shape = (seq_num, fitting_num, 100,))
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

#%%

DS = load_data(location = location_xx, io = IO, formatter = formatter)
fitting_point = 10

ds = DS
x = np.array([len(clifford_sets[0][i]) for i in range(fitting_point)])

average = average_two_qubit(ds)

#average = average_normalized_two_qubit(ds)
x[0] = 0
#x = x-1
start = 0
end = 10

x = x[start:end]
average = average[start:end]

pars, pcov = curve_fit(RB_Fidelity, x, average,
                       p0 = (0.7, 0.3, 0.2), 
                       bounds = ((0.2, 0, 0),(0.9, 2, 0.8)))
fidelity = 1-(1-pars[0])*3/4
print('fidelity is: ', fidelity)
pt = MatPlot()
pt.add(x = x, y = average, fmt = 'bp',xlabel = 'Clifford Numbers', ylabel = '$P_{|11>}$', xunit = 'N', yunit = '%')
pt.add(x = x, y = RB_Fidelity(x,pars[0],pars[1],pars[2]), fmt = 'r--',)



#%%

#pt = MatPlot()
#pt.add(x = DS.sweep)
