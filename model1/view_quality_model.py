# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:09:12 2017

@author: Administrator
"""
import os
import pandas as pd

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


def view_quality(csv_file):
    ret = {}
    os.chdir(os.path.dirname(csv_file))
    csv_data = pd.read_csv(os.path.basename(csv_file), names=['name', 'V_IR', 'V_PLEF', 'V_AIRF', 'Duration',
                                                              'Frequency', 'Interval', 'Totalplaytime'])
    names = csv_data['name']
    V_IRs = csv_data['V_IR']
    V_PLEFs = csv_data['V_PLEF']
    V_AIRFs = csv_data['V_AIRF']
    Durations = csv_data['Duration']
    Frequencys = csv_data['Frequency']
    Intervals = csv_data['Interval']
    Totalplaytimes = csv_data['Totalplaytime']
    for name, V_IR, V_PLEF, V_AIRF, Duration, Frequency, Interval, Totalplaytime in zip(names, V_IRs, V_PLEFs, V_AIRFs,
                                                                                        Durations, Frequencys,
                                                                                        Intervals, Totalplaytimes):
        ret[name[1:-1]] = {
            'qv_live_session': quality_of_live_session(V_IR, V_PLEF, V_AIRF, Duration, Frequency, Interval,
                                                       Totalplaytime),
            'qv_vod_session': quality_of_vod_session(V_IR, V_PLEF, V_AIRF, Duration, Frequency, Interval,
                                                     Totalplaytime)}
    # import pprint
    # pprint.pprint(ret)
    return ret


if __name__ == '__main__':
    res = view_quality('C:/Users/WXX/Desktop/数据集样例/view quality/view quality.csv')
