# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:02:26 2017

@author: twatson
"""

from qcodes.instrument.base import Instrument

class dummy_M4i(Instrument):
    def __init__(self, name):
        super().__init__(name,)

        self.add_parameter('sample_rate',
                           label='sample rate',
                           get_cmd=self.get_sample_rate,
                           unit='Hz',
                           set_cmd=self.set_sample_rate,
                           docstring='write the sample rate for internal sample generation or read rate nearest to desired')
        self.add_parameter('clock_mode',
                           label='clock mode',
                           get_cmd = self.get_clock_mode,
                           set_cmd=self.set_clock_mode,
                           docstring='defines the used clock mode or reads out the actual selected one')
                
        
    def get_sample_rate(self,):
        return 100
    
    def set_sample_rate(self,val):
        return val
    
    def get_clock_mode(self,):
        return 100
    
    def set_clock_mode(self,val):
        return val