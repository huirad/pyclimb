#!/usr/bin/python3

#######################################################################
#
# Copyright (C) 2015, Helmut Schmidt
#
# License:
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#######################################################################

"""Determine the climb rate from a GPS log with elevation data.

GPS data (latitude, longitude, elevation, time) is read from a GPX file.
The climb rate is calculated using a parametrizable Kalman filter.
Results are displayed as time-series plots using matplotlib.

Parameters
----------
argv[1]: string
    name of the gpx file
"""

__author__ = "Helmut Schmidt, https://github.com/huirad"
__version__ = "0.3"
__date__ = "2015-08-04"
__credits__ = "Copyright: Helmut Schmidt. License: MPLv2"

#######################################################################
#
# Module-History
#  Date         Author              Reason
#  02-Mar-2015  Helmut Schmidt      v0.1 Initial version
#  09-Jul-2015  Helmut Schmidt      v0.2 Improve source documentation
#  05-Aug-2015  Helmut Schmidt      v0.3 Remove pylint warnings
#
#######################################################################


import xml.etree.ElementTree as ET
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import numpy.matlib as ml

import sys

GPX_NAMESPACE = "{http://www.topografix.com/GPX/1/1}"

class Track:
    """Encapsulate a GPS track.

    The attributes of the trackpoints will be filled in flat lists.

    Attributes
    ----------
    timestamp: float[]
        A list containing the timestamps of the trackpoints.
        Unit of measurement: seconds.
    lat: float[]
        A list containing the latitude of the trackpoints.
        Unit of measurement: degrees.
    lon: float[]
        A list containing the longitude of the trackpoints.
        Unit of measurement: degrees.
    ele: float[]
        A list containing the elevation of the trackpoints.
        Unit of measurement: meter
    climb: float[]
        A list containing the calculated climb rate.
        Unit of measurement: meters per hour.
    """

    def __init__(self):
        """
        Parameters
        ----------
        none

        """
        self.timestamp = []
        self.lat = []
        self.lon = []
        self.ele = []
        self.climb = []
        self._first_timestamp = None

    def _append_gps_trkpt(self, trkpt, first_trackpt, time_relative):
        """Append data from a gpx trkpt to the lists.

        This is a private method

        Parameters
        ----------
        trkpt: An ET xml node representing a gpx trkpt

        """
        lat = float(trkpt.attrib.get('lat'))
        self.lat.append(lat)
        lon = float(trkpt.attrib.get('lon'))
        self.lon.append(lon)
        ele = float(trkpt.find(GPX_NAMESPACE+'ele').text)
        self.ele.append(ele)
        time = trkpt.find(GPX_NAMESPACE+'time').text
        #2015-02-14T08:21:33Z
        try:
            utc = dt.datetime.strptime(
                time,
                '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            utc = dt.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
        timestamp = utc.replace(tzinfo=dt.timezone.utc).timestamp()
        if time_relative:
            if first_trackpt:
                self._first_timestamp = timestamp
            self.timestamp.append(timestamp - self._first_timestamp)
        else:
            self.timestamp.append(timestamp)


    def from_gpx_file(self, file, time_relative=False):
        """Read a GPS track from a .gpx file.

        The following attributes are read from each trackpoint:
        timestamp, latitude, longitude, elevation

        Parameters
        ----------
        file: string
            The name of the .gpx file do read
        time_relative: bool
            If True, set timestamps will relative to
            the first timestamp.
        """
        first_trackpt = True
        tree = ET.parse(file)
        root = tree.getroot()
        for track in root.findall(GPX_NAMESPACE+'trk'):
            for trkseg in track.findall(GPX_NAMESPACE+'trkseg'):
                for trkpt in trkseg.findall(GPX_NAMESPACE+'trkpt'):
                    self._append_gps_trkpt(
                        trkpt,
                        first_trackpt,
                        time_relative)
                    first_trackpt = False



class ClimbKF:
    """ Kalman Filter for Climb Rate determination.

    All measurement units are according SI.

    Attributes
    ----------
    No public attributes.
    Use getter functions to retrieve the current state.

    Internals
    ---------
    _count_updates: int
        Number of processed elevation updates.
    _last_time: time
        Timestamp of the last levation update.
    _q_climb: float
        Process noise of the climb rate [(m/s)^2 / s].
    _q_ele: float
        Process noise of the elevation  [(m)^2 / s].
    _r_ele: float
        Measurement noise of the elevation [(m)^2].
    _state: float[]
        state vector (elevation, climb).
    _cov: float[][]
        error covariance matrix.
    """

    def __init__(self, q_ele, q_climb, r_ele):
        """ Initialize internal variables.

        Parameters
        ----------
        q_ele: float
            Process noise of the elevation [(m)^2 / s]
        q_climb: float
            Process noise of the climb rate [(m/s)^2 / s]
        r_ele: float
            Measurement noise of the elevation [(m)^2]
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
        self._state = None
        #error covariance matrix P - not initialized
        self._cov = None

    def _time_update(self, time):
        """ Private method for the time update/prediction step.

        Parameters
        ----------
        time: float
            timestamp of measurement [s].
        """
        #first construct the state propagation matrix
        time_diff = time - self._last_time
        mat_a = np.matrix([[1, time_diff], [0, 1]], float)
        #state propagation
        self._state = mat_a*self._state
        #covariance propagation: add process noise Q to P
        mat_q = np.matrix(
            [[self._q_ele*time_diff, 0],
             [0, self._q_climb*time_diff]], float)
        self._cov = mat_a*self._cov*mat_a.T + mat_q

    def _measurement_update(self, ele):
        """ Private method for the measurement update/correction step.

        Parameters
        ----------
        ele: float
            elevation measurement [m].
        """
        #measurement matrix
        mat_h = np.matrix([[1, 0]], float)
        #kalman gain
        mat_k = self._cov*mat_h.T / (mat_h*self._cov*mat_h.T + self._r_ele)
        #state update
        self._state = self._state + mat_k*(ele - mat_h*self._state)
        #covariance update
        mat_i = ml.identity(2, float)
        self._cov = (mat_i - mat_k*mat_h)*self._cov


    def _first_update(self, ele):
        """ Private method for the first update.

        Parameters
        ----------
        ele: float
            the elevation for the starting point.
        """
        self._state = np.matrix([[ele], [0]], float)
        self._cov = np.matrix([[self._r_ele, 0], [0, 1000*1000]], float)


    def update(self, time, ele, track=None):
        """ Update the climb rate based on a new elevation measurement.

        Time Update and Measurement Update in one function.

        Parameters
        ----------
        time: float
            timestamp of measurement [s].
        ele: float
            elevation measurement [m].
        track: Track
            track object where to store the results.
        """

        if self._count_updates != 0:
            self._time_update(time)
            self._measurement_update(ele)
        else:
            self._first_update(ele)
        self._count_updates += 1
        self._last_time = time

        if track:
            track.timestamp.append(time)
            track.ele.append(self._state[0, 0])
            track.climb.append(self._state[1, 0] * 3600) #convert unit to m/h

    def get_ele(self):
        """Access function to  the elevation.

        Returns
        -------
        float
            The current elevation.
        """
        return self._state[0, 0]

    def get_climb(self):
        """Access function to the climb rate.

        Returns
        -------
        float
            The current climb rate.
        """
        return self._state[1, 0]

    def get_time(self):
        """Convenience function to the last timestamp.

        Returns
        -------
        float
            The current timestamp.
        """
        return self._last_time


################################ MAIN #################################
def main():
    """ Main function

    (pylint does not like variables as moduel level - interprets them as
    constants and this causes "invalid constant name" warnings.
    By creating main() as separate function those warnings are avoided.)
    """

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

    climb_kf1 = ClimbKF(q_ele=10, q_climb=0.00001, r_ele=1)
    track_kf1 = Track()
    climb_kf2 = ClimbKF(q_ele=1, q_climb=0.00001, r_ele=1)
    track_kf2 = Track()
    climb_kf3 = ClimbKF(q_ele=100, q_climb=0.00001, r_ele=1)
    track_kf3 = Track()


    for i in range(0, len(trk.timestamp)):
        climb_kf1.update(trk.timestamp[i], trk.ele[i], track_kf1)
        climb_kf2.update(trk.timestamp[i], trk.ele[i], track_kf2)
        climb_kf3.update(trk.timestamp[i], trk.ele[i], track_kf3)

    #plot the results
    mpl.rcParams.update({'font.size': 8}) #use a smaller font
    fig = plt.figure()
    fig.canvas.set_window_title("Climb Rate from elevation")

    #first plot the elevation
    ax1 = fig.add_subplot(211)
    ax1.set_title("Elevation")
    ax1.set_xlabel("Timestamp [s]")
    ax1.set_ylabel("Elevation [m]")
    ax1.plot(
        trk.timestamp,
        trk.ele,
        color="black", linestyle="", marker=".", label="Ele")
    ax1.plot(
        track_kf1.timestamp,
        track_kf1.ele,
        color="red", linestyle="-", marker="", label="Ele1")
    ax1.plot(
        track_kf2.timestamp,
        track_kf2.ele,
        color="green", linestyle="-", marker="", label="Ele2")
    ax1.plot(
        track_kf3.timestamp,
        track_kf3.ele,
        color="blue", linestyle="-", marker="", label="Ele2")
    ax1.legend(loc="upper left")

    #then plot the calculated climb rate
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.set_title("Climb Rate")
    ax2.set_xlabel("Timestamp [s]")
    ax2.set_ylabel("Climb Rate [m/h]")
    ax2.plot(
        track_kf1.timestamp,
        track_kf1.climb,
        color="red", linestyle="-", marker="", label="Climb1")
    ax2.plot(
        track_kf2.timestamp,
        track_kf2.climb,
        color="green", linestyle="-", marker="", label="Climb2")
    ax2.plot(
        track_kf3.timestamp,
        track_kf3.climb,
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


if __name__ == "__main__":
    main()
