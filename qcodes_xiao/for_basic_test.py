# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:20:20 2017

@author: think
"""

import numpy as np
import qcodes as qc
from qcodes.loops import Loop, ActiveLoop
import numpy as np

from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.io import DiskIO
from qcodes.instrument.parameter import ManualParameter, StandardParameter, ArrayParameter
from qcodes.utils.validators import Numbers
from functools import partial

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



#def add_T1exp_metadata(data):
#        
#        data.metadata['Parameters'] = {'Nrep': 10, 't_empty': 2, 't_load': 2.4, 't_read': 2.2}
#        data.write(write_metadata=True)
#
#
#add_T1exp_metadata(data)

#datatata = LP.run(background=False)




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