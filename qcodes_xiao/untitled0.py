# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:03:19 2017

@author: think
"""

import pyvisa

rm = pyvisa.ResourceManager()

print(rm.list_resources())