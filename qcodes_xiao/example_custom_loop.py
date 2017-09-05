# -*- coding: utf-8 -*-
""" Measurements with a custom loop in qcodes

We show how to do a custom loop resulting in a 1D dataset.

https://github.com/QCoDeS/Qcodes
https://github.com/VandersypenQutech/qtt/

@author: eendebakpt
"""

#%% Load packages
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import qcodes

from qcodes import ManualParameter, QtPlot, MatPlot

import qtt
from qtt.data import makeDataSet1D
from qtt.measurements.scans import delta_time, update_dictionary


#%% Setup a live plotting window
plotQ = QtPlot(window_title='Live plot', interval=.5)
plotQ.setGeometry(100, 100, 600, 400)
plotQ.update()

qtt.live.liveplotwindow = plotQ

#%% Mock instruments
#
# The experiment we simulate is a sequence running on the AWG that does: initialize,
# apply a pulse with a certain frequency, measure the charge sensor. The resulting
# data of the charge sensor is thresholded to obtain spin up or down

frequency = ManualParameter('frequency', initial_value=10e6, unit='Hz')
elzermann_threshold = ManualParameter(
    'threshold', initial_value=.85, unit='a.u.')


class digitizer_class(qcodes.Instrument):

    def __init__(self, name, frequency, ntraces=12, **kwargs):
        """ Dummy instrument

        Resonance frequency at 10 MHz
        """
        super().__init__(name, **kwargs)
        self.frequency = frequency
        self.ntraces = ntraces

    def measure(self):
        """ Dummy measure function """
        x = 1 * (np.random.rand(40 * self.ntraces) - .5)

        for ii in range(self.ntraces):
            x[40 * ii+4:40 * (ii) + 10] += 4  # init
            x[40 * ii + 15:40 * (ii) + 22] += -2 + (self.frequency.get()-10e6 )/1e2  # wait
            pspin = np.exp(-2 * (self.frequency.get() - 10e6)**2 / 1e6)
            if pspin > .15:
                x[40 * ii + 30:40 * (ii) + 30 +
                  int(10 * pspin)] += 1  # readout
            else:
                x[40 * ii + 30:40 * (ii) + 40] += 0  # readout
        return x


digitizer = digitizer_class('digitizer', frequency)


plt.figure(10)
plt.clf()
for ii in range(10):
    digitizer.frequency.set(10e6 + ii * 100)
    x = digitizer.measure()
    plt.plot(x, label='x%d' % ii)
plt.xlabel('Time')
plt.ylabel('Raw data values')


def parse_trace(x):
    """ Convert data trace to spin-up probability """
    nn = int(x.size / 40)
    s = np.zeros(nn)
    for ii in range(nn):
        m = x[40 * ii + 30:40 * (ii) + 40]
        s[ii] = np.mean(m) > elzermann_threshold.get()
    return np.mean(s)


#%% Create station

station = qtt.Station(digitizer)


#%% Custom loop function
#
# For other examples of loop functions see qtt.measurements.scans
#

# setup the scanjob, e.g. a structure describing the scan
scanjob = {'sweepdata': {'param': frequency,
                         'start': 10e6 - 500, 'end': 10e6 + 1000, 'step': 20}}
scanjob['sweepdata']['wait_time'] = .2  # make scan slow
scanjob['minstrument'] = digitizer.name


def myscan(station, scanjob, location=None, liveplotwindow=None, verbose=1):
    """Make a scan 

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan
        location (None or str)
        liveplotwindow (None or QtPlot)
        verbose (int)

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """

    sweepdata = scanjob['sweepdata']
    wait_time_sweep = sweepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 1.)

    mname = 'spinprobability'
    start = sweepdata['start']
    step = sweepdata['step']
    end = sweepdata['end']
    sweepvalues = sweepdata['param'][start:end:step]
    alldata = makeDataSet1D(sweepvalues, yname=mname, location=location,)

    mdevice = getattr(station, scanjob['minstrument'])
    t0 = qtt.time.time()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=mname))

    time.sleep(wait_time_startscan)
    tprev = time.time()
    for ix, x in enumerate(sweepvalues):
        if verbose:
            qtt.pgeometry.tprint('myscan: %d/%d: time %.1f: setting %s to %.3f' %
                                 (ix, len(sweepvalues), time.time() - t0, sweepvalues.name, x), dt=1.5)
        # set
        sweepvalues.set(x)
        qtt.time.sleep(wait_time_sweep)

        # measure
        x = mdevice.measure()
        p = parse_trace(x)
        alldata.arrays[mname].ndarray[ix] = p

        if ix == len(sweepvalues) - 1 or ix % 5 == 0:
            delta, tprev, update = delta_time(tprev, thr=.2)
            if update and liveplotwindow:
                liveplotwindow.update_plot()

        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break
    dt = qtt.time.time() - t0

    if liveplotwindow:
        liveplotwindow.update_plot()

    update_dictionary(alldata.metadata, scanjob=scanjob,
                      dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(
        datetime.datetime.now()), )

    alldata.write(write_metadata=True)

    return alldata


# scan!
alldata = myscan(station, scanjob, location=None,
                 liveplotwindow=None, verbose=1)


# show results
print(alldata)

MatPlot(alldata.default_parameter_array())


#%% Scan again
elzermann_threshold.set(.96)
alldata = myscan(station, scanjob, location=None,
                 liveplotwindow=None, verbose=1)

#%% Extra: aborting measurements
#
# Create a GUI to abort measurements. For this redis needs to be installed, see
# https://github.com/VandersypenQutech/qtt/blob/master/INSTALL.md

mc = qtt.start_measurement_control()



