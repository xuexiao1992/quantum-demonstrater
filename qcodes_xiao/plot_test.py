# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:29:18 2017

@author: X.X
"""

from pycqed.measurement.waveform_control.viewer import *
from qcodes.data.data_set import load_data, new_data, DataSet
from qcodes.data.data_array import DataArray

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.io import DiskIO



NewIO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')

datalocation = 'data/2017-07-21/#003_T1_10-55-59'

datalocation2 = 'data/test/T1'
#formatter = HDF5Format()

#data = load_data(location=datalocation, formatter=None, io=NewIO)

original_data = np.genfromtxt('C:/Users/LocalAdmin/Documents/data/test/T1.dat', delimiter='\t', dtype=np.float)

data = original_data[:,(0,2)]


#%% Plot
import scipy
from qtt.algorithms.functions import exp_function

datax=data.T.reshape( (2,8,-1))
datax=np.transpose(datax, (0,2,1) )

plt.figure(10); plt.clf()
for ii in range(8):
    dd=datax[:,:,ii]
    
    plt.plot(dd[0],dd[1], '.-')

#%% Fitting

def fitT1(xdata, ydata, verbose=1, fig=None):
    """ Determine T1 time by fitting exponential

    Arguments:
        xdata, ydata (array): independent and dependent variable data

    Returns:
        p (array): fitted function parameters
        pp (dict): extra fitting data

    .. seealso:: exp_function
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    timescale = np.percentile(xdata, 99)- np.percentile(xdata, 1)
    # initial values
    p0 = [np.percentile(ydata, 1), np.percentile(ydata, 99)-np.percentile(ydata, 1), 1./timescale]

    if verbose:
        print('fitT1: initial parameters %s' % (p0, ))

    # fit
    pp = scipy.optimize.curve_fit(exp_function, xdata, ydata, p0=p0)
    p = pp[0]

    if fig is not None:
        y = exp_function(xdata, *list(p))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='data')
        plt.plot(xdata, y, 'm-', label='fitted exponential')
        plt.legend(numpoints=1)
    return p, dict({'pp': pp, 'p0': p0})


xdata=dd[0]
ydata=dd[1]
p, r=fitT1(xdata, ydata, verbose=1, fig=100)
#==============================================================================
# 
#==============================================================================
#%%
plt.figure(10); plt.clf()
for ii in range(8):
    dd=datax[:,:,ii]
    xdata=dd[0]
    ydata=dd[1]
    p, r=fitT1(xdata, ydata, verbose=0, fig=None)
    yfit = exp_function(xdata, *list(p))

    print('fitted: %s, T1 %.3f [ms]' % (p, 1e3*1./p[2]))

    plt.plot(dd[0],dd[1], '.')
    plt.plot(dd[0],yfit, '-')
    
#%%

data1 = DataArray(preset_data = data)

#file = open('C:/Users/LocalAdmin/Documents/data/test/T1.dat')

#datanew = new_data(io = NewIO, name = 'testest', arrays = data1)
#dataset = DataSet(arrays = data1, io = NewIO, location = 'data/test/xiao')




#datata = np.loadtxt(name = )
Npoints = 100


#
#plt.ion()
#plt.figure()
#
#
#plt.pcolor(np.linspace(1, Npoints, num=Npoints), data.arrays[data.action_id_map[()]], res_avg)