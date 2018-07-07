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
location12 = '2017-11-07/21-21-27/RB_experimentAllXY_sequence'      # project CZ on Q1 92.15%

location13 = '2017-11-08/12-29-49/RB_experimentAllXY_sequence'
location14 = '2017-11-08/13-30-25/RB_experimentAllXY_sequence'      # project CZ on Q2  85%
location15 = '2017-11-08/17-51-57/RB_experimentAllXY_sequence'      # project CZ on Q2  85%



location16 = '2017-11-09/10-30-58/RB_experimentAllXY_sequence'
location16 = '2017-11-09/14-50-11/RB_experimentAllXY_sequence'      # # project CZ on Q1  92.74%   ctrl at |1>
location17 = '2017-11-09/16-02-47/RB_experimentAllXY_sequence'       # project CZ on Q1  93.44%
location18 = '2017-11-09/17-35-19/RB_experimentAllXY_sequence'      # project CZ on Q1  93.61%
location19 = '2017-11-09/19-37-27/RB_experimentAllXY_sequence'      # project CZ on Q1  95.30%   ctrl at |0>

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
location_xyy1 = '2017-11-26/03-27-08/RB_experimentAllXY_sequence' #80%

location_te = '2017-12-12/12-02-59_DAC_V_sweep'


location_a = '2018-02-28/17-30-30/RB_experimentAllXY_sequence' #interleave Q1 with Q2 down
location_b = '2018-03-01/15-06-19/RB_experimentAllXY_sequence' #interleave Q1 with Q2 up

location_c = '2018-03-03/10-35-16/RB_experimentAllXY_sequence' #interleave Q2 with Q1 down
location_d = '2018-03-03/13-51-20/RB_experimentAllXY_sequence' #interleave Q2 with Q1 up


#%%


IO_K = DiskIO(base_location = 'K:\\ns\\qt\\spin-qubits\\data\\b059_data\\2018 data\\Data\\RB_experiment')

'''
qubit 2
'''

location_new = '2018-05-30/12-05-27/RB_experimentAllXY_sequence'
location_new2 = '2018-05-30/14-16-31/RB_experimentAllXY_sequence'
location_new3 = '2018-05-30/18-26-14/RB_experimentAllXY_sequence'
location_new4 = '2018-05-31/11-27-16/RB_experimentAllXY_sequence'

location_dc = '2018-05-29/15-05-16_DAC_V_sweep'


'''
qubit 1
'''
location_new = '2018-06-01/17-40-40/RB_experimentAllXY_sequence'

'''
simultaneous
'''

location_new = '2018-06-04/18-00-34/RB_experimentAllXY_sequence'

location_new2 = '2018-06-05/11-45-29/RB_experimentAllXY_sequence'

location_new2 = '2018-06-20/13-08-22/RB_experimentAllXY_sequence'

location_new2 = '2018-06-20/16-17-29/RB_experimentAllXY_sequence'

location_new2 = '2018-06-22/18-36-27/RB_experimentAllXY_sequence'


location_new2 = '2018-06-27/16-28-13/RB_experimentAllXY_sequence'

location_new2 = '2018-06-29/16-09-21/RB_experimentAllXY_sequence'
location_new2 = '2018-06-29/19-33-12/RB_experimentAllXY_sequence'
location_new2 = '2018-06-29/21-42-22/RB_experimentAllXY_sequence'
location_new2 = '2018-06-30/01-22-14/RB_experimentAllXY_sequence'
#location_new2 = '2018-06-30/03-37-05/RB_experimentAllXY_sequence'
#location_new2 = '2018-06-30/11-51-09/RB_experimentAllXY_sequence'
#location_new2 = '2018-06-30/13-20-51/RB_experimentAllXY_sequence'
#location_new2 = '2018-06-30/16-25-38/RB_experimentAllXY_sequence'
#location_new2 = '2018-07-01/14-36-37/RB_experimentAllXY_sequence'
#location_new2 = '2018-07-01/17-29-32/RB_experimentAllXY_sequence'
#location_new2 = '2018-07-01/18-10-34/RB_experimentAllXY_sequence'
#location_new2 = '2018-07-01/18-49-56/RB_experimentAllXY_sequence'
#location_new2 = '2018-07-01/19-40-32/RB_experimentAllXY_sequence'
location_new2 = '2018-07-02/01-31-40/RB_experimentAllXY_sequence'
location_new2 = '2018-07-02/11-49-51/RB_experimentAllXY_sequence'
location_new2 = '2018-07-02/16-28-41/RB_experimentAllXY_sequence'
location_new2 = '2018-07-02/17-24-54/RB_experimentAllXY_sequence'
location_new2 = '2018-07-02/19-10-11/RB_experimentAllXY_sequence'

location_new2 = '2018-07-05/13-25-05/RB_experimentAllXY_sequence'


DS = load_data(location = location_new2, io = IO_K, formatter = formatter)

#%%
NewIO = DiskIO(base_location = 'D:\\Data\\RB_experiment')

DS = load_data(location = location_dc, io = NewIO)

DS = load_data(location = location_4, io = IO, formatter = formatter)
DS2 = load_data(location = location_b, io = IO, formatter = formatter)
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
