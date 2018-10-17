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


location_new2 = '2018-07-25/01-51-34/RB_experiment2AllXY_sequence'


'''
interleaved character benchmarking
'''

'''
not bad, second points seems deviated
'''
location_new2 = '2018-08-28/01-01-21/RB_experimentAllXY_sequence'   # 73% (0) 74.4% (1) Clifford + CZ

#location_new2 = '2018-08-31/01-15-09/RB_experimentAllXY_sequence'   # 69% (0) 69 % (1) Clifford + CZ

'''
bad data
'''
#location_new2 = '2018-08-31/20-11-02/RB_experimentAllXY_sequence'   # 75% (0) 81% (1) Clifford + CZ

'''
this one not bad
'''
location_new2 = '2018-09-01/02-20-56/RB_experimentAllXY_sequence'   # 69% (0) 77% (1) Clifford + CZ

'''
this one is good
'''
location_new2 = '2018-08-30/01-15-32/RB_experimentAllXY_sequence'   # 86% (start from 1) 81.25% (from 0) Clifford + CZ

#location_new2 = '2018-08-29/17-04-46/RB_experimentAllXY_sequence'   #  72% (0) 78% (1) Clifford + CZ

#location_new2 = '2018-08-26/23-52-44/RB_experimentAllXY_sequence'   #  79% (0) 78.3% (1) Clifford + CZ

#location_new2 = '2018-08-28/23-20-23/RB_experimentAllXY_sequence'   #  67.9% (0) 79.8% (1) Clifford + CZ

#location_new2 = '2018-08-28/01-01-21/RB_experimentAllXY_sequence'   #  73.1% (0) 74.5% (1) Clifford + CZ

#location_new2 = '2018-09-02/01-19-46/RB_experimentAllXY_sequence'   #  76.85% (0) 80% (1) Clifford + CZ

#location_new2 = '2018-09-02/23-15-55/RB_experimentAllXY_sequence'   #  79.325% (0) 86.38% (1) Clifford + CZ

'''
super good data!!! in memory for the lunch
'''
location_new2 = '2018-09-03/12-45-49/RB_experimentAllXY_sequence'   #   Clifford + CZ

location_new2 = '2018-09-03/20-54-07/RB_experimentAllXY_sequence'   #   Clifford + CZ

location_new2 = '2018-09-05/13-09-22/RB_experimentAllXY_sequence'   #   Clifford + CZ

location_new2 = '2018-09-05/20-02-56/RB_experimentAllXY_sequence'   #   Clifford + CZ

location_new2 = '2018-09-06/02-35-11/RB_experimentAllXY_sequence'   #   Clifford + CZ

location_new2 = '2018-09-06/13-46-30/RB_experimentAllXY_sequence'   #   Clifford + CZ

'''
very high but seems like an even odd effect
'''
location_new2 = '2018-09-06/23-39-09/RB_experimentAllXY_sequence'   #   Clifford + CZ


'''
non_interleave new
'''
location_new2 = '2018-09-04/11-24-17/RB_experiment2AllXY_sequence'   #   Clifford without CZ


DS4 = load_data(location = location_new2, io = IO_K2, formatter = formatter)

#DS = load_data(location = location_new2, io = IO_K, formatter = formatter)

#%% character benchmarking without normalization

ds = DS
fitting_points = 18
#seq_rep_num = 20
seq_rep_num = int(DS.sequence_number_set.shape[0]/16)
sequence_number = 16*seq_rep_num
#sequence_number = 112
repetition = 100
init_state = ['00', '01', '10', '11',]

x = np.array([len(clifford_sets[0][i]) for i in range(fitting_points)])

#def average_two_qubit(ds)
data = {}
average = {}
for init in init_state:
    data[init] = np.ndarray(shape = (int(sequence_number/4), fitting_points, repetition,))

#%%
for j in range(sequence_number):
#    j_new = j%seq_rep_num
    
    j_new = (j//16)*4 + j%4
    
#    Pauli_index = j//seq_rep_num
    Pauli_index = j%16
    
    init_index = Pauli_index//4
#    print(init_index)
#    print(j)
    for i in range(11, 11+fitting_points):
        for k in range(repetition):
#            if i + k == 11:
#                print(init_index)
            if ds.singleshot_data[j][0][i][k] == 0 and ds.singleshot_data[j][1][i][k] == 0:
                data[init_state[init_index]][j_new][i-11][k] = int(1)
            else:
                data[init_state[init_index]][j_new][i-11][k] = int(0)
                
#for init in init_state:
#    average_probability = data[init].mean(axis = 2)
#    average[init] = average_probability.mean(axis = 0)
    
for init in init_state:
    average_probability = data[init].mean(axis = 2)
    average[init] = average_probability.mean(axis = 0)
    
P1 = average['00'] - average['01'] + average['10'] - average['11'] 
P2 = average['00'] + average['01'] - average['10'] - average['11']
P3 = average['00'] - average['01'] - average['10'] + average['11']

#%%
start = 1

pars1, pcov1 = curve_fit(RB_Fidelity, x[start:], P1[start:], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1, 0.001)))
pars2, pcov2 = curve_fit(RB_Fidelity, x[start:], P2[start:], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1, 0.001)))
pars3, pcov3 = curve_fit(RB_Fidelity, x[start:], P3[start:], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1, 0.001)))
P = 3/15 * (pars1[0] + pars2[0]) + 9/15 * pars3[0]
Fidelity = 1- (1-P)*3/4
print('Fidelity:', Fidelity)
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

P_pi_q1 = 0.98
P_pi_q2 = 0.97

P_init_q1 = 1#0.99
P_init_q2 = 1#0.99

fitting_point = 24

def average_normalized_two_qubit(ds):
    
    seq_num = len(ds.sequence_number_set)   # sequence_num e.g. Count, RB sequences
    
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
    
    average_normalized_q1 = np.zeros(shape = (seq_num, fitting_num,))
    average_normalized_q2 = np.zeros(shape = (seq_num, fitting_num,))
    average_normalized = np.zeros(shape = (seq_num, fitting_num,))
    
    for seq in range(seq_num):
        
        P2_q2 = ds.probability_data[seq][0][-2]
        P3_q2 = ds.probability_data[seq][0][-1]
        P2_q1 = ds.probability_data[seq][1][-2]
        P3_q1 = ds.probability_data[seq][1][-1]
        
#        P2_q2 = ds.probability_data[seq][0][1:].min()
#        P3_q2 = ds.probability_data[seq][0][1:].max()
#        P2_q1 = ds.probability_data[seq][1][1:].min()
#        P3_q1 = ds.probability_data[seq][1][1:].max()
        
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

        F_R_q2[0] = 1-P2_q2                 # readout Q2
        F_R_q2[1] = P3_q2/P_pi_q2
        F_R_q1[0] = 1-P2_q1
        F_R_q1[1] = P3_q1/P_pi_q1
        
        Fidelity_1 = np.array([[F_R_q1[0], 1-F_R_q1[1]],[1-F_R_q1[0], F_R_q1[1]]])
        Fidelity_2 = np.array([[F_R_q2[0], 1-F_R_q2[1]],[1-F_R_q2[0], F_R_q2[1]]])
        
#        print('Fidelity1:', Fidelity_1)
#        print('Fidelity2:', Fidelity_2)
        
        Fidelity = np.kron(Fidelity_1, Fidelity_2)      # Fidelity Q1 and Q2
        
        for i in range(fitting_num):
            
            average_raw_q1 = np.array([[ds.probability_data[seq][1][i]], [ds.probability_data[seq][1][i]]])
            
            average_raw_q2 = np.array([[ds.probability_data[seq][0][i]], [ds.probability_data[seq][0][i]]])
            
            average_raw = np.array([[average_00[seq][i]], [average_01[seq][i]], [average_10[seq][i]], [average_11[seq][i]]])
            
#            average_normalized_q1[seq][i] = np.linalg.multi_dot([np.linalg.inv(Fidelity_1), average_raw_q1])
            
#            average_normalized_q2[seq][i] = np.linalg.multi_dot([np.linalg.inv(Fidelity_2), average_raw_q2])
            
            average_normalized[seq][i] = np.linalg.multi_dot([np.linalg.inv(Fidelity), average_raw])[0]
    
    return average_normalized, average_normalized_q1, average_normalized_q2

#%%
first_sequence = 32
sequence_number = 128

ds = DS

average_normalized, average_normalized_q1, average_normalized_q2 = average_normalized_two_qubit(ds)

data = {}
average = {}
for init in init_state:
    data[init] = np.ndarray(shape = (int(sequence_number/4), fitting_point))

for j in range(sequence_number):
    
    j_new = (j//16)*4 + j%4
    
    Pauli_index = j%16
        
    init_index = Pauli_index//4
    
    data[init_state[init_index]][j_new] = average_normalized[j]

for init in init_state:
    average[init] = data[init].mean(axis = 0)

P1 = average['00'] - average['01'] + average['10'] - average['11'] 
P2 = average['00'] + average['01'] - average['10'] - average['11']
P3 = average['00'] - average['01'] - average['10'] + average['11']
#%%
start = 1
end = 18

pars1, pcov1 = curve_fit(RB_Fidelity, x[start:end], P1[start:end], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1.2, 0.005)))
pars2, pcov2 = curve_fit(RB_Fidelity, x[start:end], P2[start:end], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1.2, 0.005)))
pars3, pcov3 = curve_fit(RB_Fidelity, x[start:end], P3[start:end], p0 = (0.9, 0.2, 0), bounds = ((0.5, 0, 0),(1, 1.2, 0.005)))
P = 3/15 * (pars1[0] + pars2[0]) + 9/15 * pars3[0]
Fidelity = 1- (1-P)*3/4
print('P:', P)
print('Fidelity:', Fidelity)

    

#%%

import matplotlib as mpl

#plot_point = fitting_points
start_point = 1
plot_points = fitting_point

pt = MatPlot()
pt.add(x = x[start_point:plot_points],y = P1[start_point:plot_points], fmt = 'rp',xlabel = 'Clifford Numbers', ylabel = '$P_{|00>}$', xunit = 'N', yunit = '%')
pt.add(x = x[start_point:plot_points],y = P2[start_point:plot_points], fmt = 'bp',xlabel = 'Clifford Numbers', ylabel = '$P_{|00>}$', xunit = 'N', yunit = '%')
pt.add(x = x[start_point:plot_points],y = P3[start_point:plot_points], fmt = 'gp',xlabel = 'Clifford Numbers', ylabel = '$P_{|00>}$', xunit = 'N', yunit = '%')
pt.add(x = x[start_point:plot_points], y = RB_Fidelity(x,pars1[0],pars1[1],pars1[2])[start_point:plot_points],fmt = 'r--', )
pt.add(x = x[start_point:plot_points], y = RB_Fidelity(x,pars2[0],pars2[1],pars2[2])[start_point:plot_points],fmt = 'b--', )
pt.add(x = x[start_point:plot_points], y = RB_Fidelity(x,pars3[0],pars3[1],pars3[2])[start_point:plot_points],fmt = 'g--', )

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
