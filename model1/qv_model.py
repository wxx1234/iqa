# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:09:12 2017

@author: Administrator
"""

c23 = 0
c24 = 0.0346
c25 = 0.64
c26 = 0
c27 = 2.948
c28 = 0.54
c29 = 2.7
c30 = 0.9
c31 = 1.8
c32 = 0.15
c33 = 5.5
c34 = 1.05


def quality_of_live_session(V_IR, V_PLEF, V_AIRF, Duration, Frequency, Interval, Totalplaytime):
    temp = ((V_AIRF * V_IR / (c24)) ** c25) * ((V_PLEF / (c27)) ** c28)
    QBlocking = temp / (1 + temp)
    Qv_LiveInstant = 5 - QBlocking
    return Qv_LiveInstant


def quality_of_vod_session(V_IR, V_PLEF, V_AIRF, Duration, Frequency, Interval, Totalplaytime):
    temp = ((Duration / c29) ** c30) * ((Interval / c31) ** c32) * ((Frequency / c33) ** c34)
    QStalling = temp / (1 + temp)
    Qv_VodInstant = 5 - QStalling
    return Qv_VodInstant
