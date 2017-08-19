# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:21:04 2017

@author: X.X
"""

from copy import copy, deepcopy
#import stationF006
class ac():
    def __init__(self, a,b,):
        self.a = a
        self.b = b
        print(self.a,self.b)
    def __call__(self, **kw):
        self.a = kw.pop('a',self.a),
        self.b = kw.pop('b',self.b)
        print(self.b,self.a)
        
        
        
a = ac(a = 5, b = 12)

