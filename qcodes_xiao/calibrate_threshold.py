# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:50:58 2018

@author: LocalAdmin
"""

import scipy

def double_gaussian(signal_amp, params):
    """ A model for the sum of two Gaussian distributions. """
    [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up] = params
    gauss_dn = A_dn*np.exp(-(signal_amp-mean_dn)**2/(2*sigma_dn**2))
    gauss_up = A_up*np.exp(-(signal_amp-mean_up)**2/(2*sigma_up**2))
    double_gauss = gauss_dn + gauss_up
    return double_gauss
 
def fit_double_gaussian(signal_amp, counts, par_guess=None):
    """ Fitting of double gaussian. """
    func = lambda signal_amp, A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up: double_gaussian(signal_amp, [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up])
    if par_guess is None:
        A_dn = np.max(counts)
        A_up = np.max(counts)
        sigma_dn = 1/3*(np.max(signal_amp) - np.min(signal_amp))
        sigma_up = 1/3*(np.max(signal_amp) - np.min(signal_amp))
        mean_dn = np.min(signal_amp) + 1/3*(np.max(signal_amp) - np.min(signal_amp))
        mean_up = np.min(signal_amp) + 2/3*(np.max(signal_amp) - np.min(signal_amp))
        par_guess = np.array([A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up])
    par_fit, par_cov = scipy.optimize.curve_fit(func, signal_amp, counts, p0 = par_guess, bounds = ([0,0,0,0,-10,-10],[2*np.max(counts),2*np.max(counts),1,1,10,10]))
    return par_fit, par_guess
 
def calibrate_threshold(data):
    hist_values = []
    for i in range(1,10):
        for j in range(10,20):
            hist_values.append(min(data.raw_data[0,0,i,j,:]))
 
#    plt.figure()
#    h = plt.hist(hist_values, bins = 45)
    h = np.histogram(hist_values, bins = 45)
 
    signal_amp = (h[1][:-1] + h[1][1:])/2
    mean_dn = h[1][0] - (h[1][0]-h[1][-1])/3
    mean_up = h[1][0] - 2.2*(h[1][0]-h[1][-1])/3
    fit_params,_ = fit_double_gaussian(signal_amp,h[0])#,[300,400,0.022,0.026,mean_dn,mean_up])
   
#    plt.plot(signal_amp,double_gaussian(signal_amp,fit_params))
   
    visibility = []
    for s in signal_amp:
        ee_t = scipy.stats.norm.cdf(s, fit_params[5], fit_params[3])
        ee_s = 1 - scipy.stats.norm.cdf(s, fit_params[4], fit_params[2])
        visibility.append(1 - ee_t - ee_s)
    th_ind = visibility.index(max(visibility))
    threshold = signal_amp[th_ind]
   
    return threshold