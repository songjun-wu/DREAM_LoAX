# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:40:32 2016

@author: Erin
"""

class Model():
    
    def __init__(self, likelihood, sampled_parameters):
        self.likelihood = likelihood
        if type(sampled_parameters) is list:
            self.sampled_parameters = sampled_parameters
        else:
            self.sampled_parameters = [sampled_parameters]
        
    def total_logp(self, q0, chainID, limits, weights, obs_all):


        like = self.likelihood(q0, chainID, limits, weights, obs_all)

        return like
    
        
        