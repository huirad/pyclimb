#!/usr/bin/python3

##########################################################################
#
# Copyright (C) 2015, Helmut Schmidt
#
# License:
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
##########################################################################

"""Determine the climb rate from a GPS log with elevation data.

GPS data (latitude, longitude, elevation, time) is read from a GPX file.
The climb rate is calculated using a parametrizable Kalman filter.
Results are displayed as time-series plots using matplotlib.

Args:
    argv[1]: name of the gpx file
"""


import xml.etree.ElementTree as ET
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import numpy.matlib as ml

import sys


class Track:
    """Encapsulate a GPS track.

    The attributes of the trackpoints will be filled in flat lists.
    
    Attributes:
        timestamp: A list containing the timestamps of the trackpoints.
            Unit of measurement: seconds.
        lat: A list containing the latitude of the trackpoints.
            Unit of measurement: degrees.
        lon: A list containing the longitude of the trackpoints.
            Unit of measurement: degrees.
        ele: A list containing the elevation of the trackpoints.
            Unit of measurement: meter
        climb: A list containing the calculated climb rate.
            Unit of measurement: meters per hour.
    """

    def __init__(self):
        """Initialize all attributes as empty lists."""
        self.timestamp = []
        self.lat   = []
        self.lon   = []
        self.ele   = []
        self.climb = []

    def from_gpx_file(self, file, time_relative = False):
        """Read a GPS track from a .gpx file.

        The following attributes are read from each trackpoint: 
            timestamp, latitude, longitude, elevation

        Args:
            file: A string containing the name of the .gpx file.
            time_relative: If True, set timestamps will relative to 
                the first timestamp.
        """
        first_trackpt = True
        tree = ET.parse(file)
        GPX_NAMESPACE = "{http://www.topografix.com/GPX/1/1}"
        root = tree.getroot()
        for track in root.findall(GPX_NAMESPACE+'trk'):
            for trkseg in track.findall(GPX_NAMESPACE+'trkseg'):
                for trkpt in trkseg.findall(GPX_NAMESPACE+'trkpt'):
                    lat = float(trkpt.attrib.get('lat'))
                    self.lat.append(lat)
                    lon = float(trkpt.attrib.get('lon'))
                    self.lon.append(lon)
                    ele = float(trkpt.find(GPX_NAMESPACE+'ele').text)
                    self.ele.append(ele)
                    time = trkpt.find(GPX_NAMESPACE+'time').text
                    #2015-02-14T08:21:33Z
                    try:
                        utc = dt.datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.%fZ')
                    except:
                        utc = dt.datetime.strptime(time,'%Y-%m-%dT%H:%M:%SZ')
                    timestamp = utc.replace(tzinfo=dt.timezone.utc).timestamp()
                    if time_relative:
                        if first_trackpt:
                            first_timestamp = timestamp
                            first_trackpt = False
                        self.timestamp.append(timestamp - first_timestamp)
                    else:
                        self.timestamp.append(timestamp)


class ClimbKF:
    """ Kalman Filter for Climb Rate determination.

    All measurement units are according SI.

    Attributes:
        No public attributes - use getter functions to retrieve the current state 

    Internals:
        _count_updates: Number of processed elevation updates.
        _last_time: Timestamp of the last levation update.
        _q_climb: Process noise of the climb rate [(m/s)^2 / s].
        _q_ele: Process noise of the elevation  [(m)^2 / s].
        _r_ele: Measurement noise of the elevation [(m)^2].
        _x: state vector (elevation, climb).
        _P: error covariance matrix.
    """

    def __init__(self, q_ele, q_climb, r_ele):
        """ Initialize internal variables.

        Args:
            q_ele: Process noise of the elevation [(m)^2 / s]
            q_climb: Process noise of the climb rate [(m/s)^2 / s]
            r_ele: Measurement noise of the elevation [(m)^2]
        """

        #number of updates.
        self._count_updates = 0
        #last timestamp [s]
        self._last_time = 0

        #error parameters
        self._q_climb = q_climb
        self._q_ele = q_ele
        self._r_ele = r_ele
         
        #state vector x: elevation, climb - not initialized
        self._x = None
        #error covariance matrix P - not initialized
        self._P = None

    def _time_update(self, time):
        """ Private method for the time update/prediction step.

        Args:
            time: timestamp of measurement [s].   
        """
        #first construct the state propagation matrix
        dt = time - self._last_time
        A = np.matrix([[1, dt],[0, 1]], float)
        #state propagation
        self._x = A*self._x
        #covariance propagation: add process noise Q to P
        Q = np.matrix([[self._q_ele*dt, 0],[0, self._q_climb*dt]], float)
        self._P = A*self._P*A.T + Q

    def _measurement_update(self, ele):
        """ Private method for the measurement update/correction step.

        Args:
            ele: elevation measurement [m].
        """
        #measurement matrix
        H = np.matrix([[1,0]], float)
        #kalman gain
        K = self._P*H.T / (H*self._P*H.T + self._r_ele)
        #state update
        self._x = self._x + K*(ele - H*self._x)
        #covariance update
        I=ml.identity(2,float)
        self._P = (I - K*H)*self._P        
        

    def _first_update(self, ele):
        """ Private method for the first update.

        Args:
            ele: the elevation for the starting point.
        """
        self._x = np.matrix([[ele],[0]], float)
        self._P = np.matrix([[self._r_ele, 0],[0, 1000*1000]], float)


    def update(self, time, ele, track = None):
        """ Update the climb rate based on a new elevation measurement.

        Time Update and Measurement Update in one function.

        Args:
            time: timestamp of measurement [s].
            ele: elevation measurement [m].
            track: track object where to store the results.
        """

        if self._count_updates != 0:
            self._time_update(time)
            self._measurement_update(ele)
        else:
            self._first_update(ele)
        self._count_updates += 1
        self._last_time = time

        if (track):
            track.timestamp.append(time)
            track.ele.append(self._x[0,0])
            track.climb.append(self._x[1,0] * 3600) #convert unit to m/h

    def get_ele(self):
        """Access function to  the elevation.

        Returns:
            The current elevation.
        """
        return self._x[0,0]

    def get_climb(self):
        """Access function to the climb rate.

        Returns:
            The current climb rate.
        """
        return self._x[1,0]

    def get_time(self):
        """Convenience function to the last timestamp.
        
        Returns:
            The current timestamp.
        """
        return self._last_time


################################ MAIN #################################
if __name__ == "__main__":

    trk = Track()
    trk.from_gpx_file(sys.argv[1], True)

    #Kalman Filter tuning
    #   [Note: a typical climb rate is 0.1 m/s = 360 m/h]
    #   The "stiffness" of the estimated climb rate depends mainly on 
    #       q_ele/q_climb
    #   Useful stiffness for hiking: "averaging" over 15min-1h
    #
    #Example values:
    #   q_ele: Process noise of the elevation  [(m)^2 / s]
    #       q_ele = 1: elevation may deviate from estimation 
    #           by 1m in 1 s or 10m in 100s
    #       q_ele = 100: elevation may deviate from estimation 
    #           by 10m in 1 s or 100m in 100s
    #   q_climb: Process noise of the climb rate [(m/s)^2 / s]
    #       q_climb = 0.1: climb rate may change 
    #           by 0.1 m/s in 1s or by 1m/s in 100s 
    #           ==> far too high
    #       q_climb = 0.001: climb rate may change 
    #           by 0.01 m/s in 1s or by 0.1m/s in 100s 
    #           ==> maybe OK but still a bit nervous
    #       q_climb = 0.00001: climb rate may change 
    #           by 0.0001 m/s in 1s or by 0.01m/s in 100s
    #       0.01m/s in 100s means 4m/h per 2min
    #   r_ele: Measurement noise of the elevation [(m)^2]
    #       r_ele = 0.1: elevation measurement is accurate to 0.1m
    #       r_ele = 1: elevation measurement is accurate to 1m
    #           ==> realistic for a barometric elevation measurement
    #       r_ele = 100: elevation measurement is accurate to 10m

    climbKF1 = ClimbKF(q_ele=10, q_climb=0.00001, r_ele=1)
    trackKF1 = Track()
    climbKF2 = ClimbKF(q_ele=1, q_climb=0.00001, r_ele=1)
    trackKF2 = Track()
    climbKF3 = ClimbKF(q_ele=100, q_climb=0.00001, r_ele=1)
    trackKF3 = Track()


    for i in range(0, len(trk.timestamp)):
        climbKF1.update(trk.timestamp[i], trk.ele[i], trackKF1)
        climbKF2.update(trk.timestamp[i], trk.ele[i], trackKF2)
        climbKF3.update(trk.timestamp[i], trk.ele[i], trackKF3)

    #plot the results
    mpl.rcParams.update({'font.size': 8}) #use a smaller font
    fig=plt.figure()
    fig.canvas.set_window_title("Climb Rate from elevation")

    #first plot the elevation
    ax1 = fig.add_subplot(211)
    ax1.set_title("Elevation")
    ax1.set_xlabel("Timestamp [s]")
    ax1.set_ylabel("Elevation [m]")
    ax1.plot(trk.timestamp, trk.ele, 
        color="black", linestyle="", marker=".", label="Ele")
    ax1.plot(trackKF1.timestamp, trackKF1.ele, 
        color="red",   linestyle="-", marker="", label="Ele1")
    ax1.plot(trackKF2.timestamp, trackKF2.ele, 
        color="green", linestyle="-", marker="", label="Ele2")
    ax1.plot(trackKF3.timestamp, trackKF3.ele, 
        color="blue", linestyle="-", marker="", label="Ele2")
    ax1.legend(loc="upper left")

    #then plot the calculated climb rate
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.set_title("Climb Rate")
    ax2.set_xlabel("Timestamp [s]")
    ax2.set_ylabel("Climb Rate [m/h]")
    ax2.plot(trackKF1.timestamp, trackKF1.climb, 
        color="red",  linestyle="-", marker="", label="Climb1")
    ax2.plot(trackKF2.timestamp, trackKF2.climb, 
        color="green", linestyle="-", marker="", label="Climb2")
    ax2.plot(trackKF3.timestamp, trackKF3.climb, 
        color="blue", linestyle="-", marker="", label="Climb3")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    #add a data cursor
    mpl.widgets.Cursor(ax1, useblit=True, color='gray', linewidth=1)
    mpl.widgets.Cursor(ax2, useblit=True, color='gray', linewidth=1)
    #alternatively: add a multi-cursor for both axes
    #mpl.widgets.MultiCursor(fig.canvas, (ax1, ax2), color='gray',
    #    lw=1, horizOn=True)


    #everything set ==> show the plot
    plt.show()
