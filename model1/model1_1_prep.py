# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:09:54 2017

@author: Administrator
"""

import numpy as np
import pandas as pd


def model1_1_fbf_var(file_path=None, width=None, height=None, qp_list=None, frame_type_list=None, skip_ratio=None,
                     mv0=None, mv1=None, bit_size_list=None, qp_max_list=None, qp_min_list=None, frame_rate=None,
                     bit_rate=None):
    if file_path:
        csv_data = pd.read_csv(file_path, names=['frame', 'pkt_pts', 'pkt_size', 'width', 'height', 'pict_type',
                                                 'skip_ratio', 'qp_min', 'qp_max', 'qp_avg', 'mv0_min', 'mv0_max',
                                                 'mv0_avg', 'mv1_min', 'mv1_max', 'mv1_avg'])
        frame_type_list = csv_data['pict_type']
        bit_size_list = csv_data['pkt_size']
        width = csv_data['width']
        height = csv_data['height']
        qp_min_list = csv_data['qp_min']
        qp_max_list = csv_data['qp_max']
        qp_list = csv_data['qp_avg']
        skip_ratio = csv_data['skip_ratio']
        for i in range(len(skip_ratio)):
            skip_ratio.set_value(i, float(skip_ratio[i][:-1]) / 100)
        mv0 = csv_data['mv0_avg']
        csv_data = csv_data.dropna(axis=1, how='all')
        number_of_frames, number_of_params = csv_data.shape
        if number_of_params > 13:
            mv1 = csv_data['mv1_avg']
        else:
            mv1 = mv0
    # I-frame Flicker Detection
    i_intra_flicker = 0.
    ind = [i for i, a in enumerate(frame_type_list.tolist()) if a == "I"]
    for j in range(1, len(ind) - 1):
        if (qp_list[ind[j]] - qp_list[ind[j + 1]] > 5) and (qp_list[ind[j]] - qp_list[ind[j - 1]] > 5):
            i_intra_flicker = 1
            break
        else:
            continue
            # IPB ststistics

    # Temperal pooling for frame quality.
    # frame_quality = []; data_smooth = [];

    # QP list for different frame type

    # calculate the frame qulity for each frame type
    # currently only I/P frames will be counted.
    count_i = 0

    count_all = 0

    # caltulate the num of I/B/P
    i_location = np.array(ind.copy())
    count_ii = len(i_location)
    count_pp = len([i for i, a in enumerate(frame_type_list.tolist()) if a == "P"])
    nbr_between_two_i_pics = count_pp / count_ii
    kfr = frame_rate / nbr_between_two_i_pics

    frame_i_sts = []
    for i in range(len(qp_list)):
        frame_type = frame_type_list[i]
        if frame_type == 'I' and qp_list[i] > 2:
            count_i = count_i + 1
            count_all = count_all + 1
            frame_i_sts.append(bit_size_list[i])

    ans = [np.mean(qp_list), frame_rate, i_intra_flicker, np.mean(qp_max_list), np.mean(qp_min_list), bit_rate,
           np.mean(frame_i_sts), np.mean(skip_ratio),
           np.mean([max(mv0[i], mv1[i]) for i, a in enumerate(mv0.tolist())]), kfr, width[0] * height[0]]

    return ans


if __name__ == '__main__':
    print(
        model1_1_fbf_var(file_path="C:/Users/WXX/Desktop/model1_tool/Ori_data/ffprobe_csv_huawei/1.csv", frame_rate=25,
                         bit_rate=1188))
