from model1.model1_1 import model1_1
from model1.model1_2 import model1_2
import pandas as pd
import time
import os
from os import listdir
from os.path import isfile, join


def list_all_files(file_path):
    return [f for f in listdir(file_path) if isfile(join(file_path, f))]


def model1_single(filename):
    os.chdir(os.path.dirname(filename))
    csv_data = pd.read_csv(os.path.basename(filename),
                           names=['frame', 'pkt_pts', 'pkt_size', 'width', 'height', 'pict_type',
                                  'skip_ratio', 'qp_min', 'qp_max', 'qp_avg', 'mv0_min', 'mv0_max',
                                  'mv0_avg', 'mv1_min', 'mv1_max', 'mv1_avg', 'frame_rate', 'bit_rate'])

    width = csv_data['width']
    height = csv_data['height']
    qp_list = csv_data['qp_avg']
    frame_type_list = csv_data['pict_type']
    skip_ratio = csv_data['skip_ratio'] / 100
    mv0 = csv_data['mv0_avg']
    csv_data = csv_data.dropna(axis=1, how='all')
    mv1 = csv_data['mv1_avg']
    frame_rate = csv_data['frame_rate'][0]
    bit_rate = csv_data['bit_rate'][0]
    number_of_frames, number_of_params = csv_data.shape
    bit_size_list = csv_data['pkt_size']
    qp_min_list = csv_data['qp_min']
    qp_max_list = csv_data['qp_max']
    model1_1_mos = model1_1(width=width, height=height, qp_list=qp_list, frame_type_list=frame_type_list,
                            skip_ratio=skip_ratio, mv0=mv0, mv1=mv1, bit_size_list=bit_size_list,
                            qp_max_list=qp_max_list, qp_min_list=qp_min_list, frame_rate=frame_rate, bit_rate=bit_rate)
    model1_2_mos = model1_2(width=width, height=height, qp_list=qp_list, frame_type_list=frame_type_list,
                            skip_ratio=skip_ratio, mv0=mv0, mv1=mv1, number_of_frames=number_of_frames,
                            number_of_params=number_of_params, device='TV')
    mos_all = model1_2_mos - (5 - model1_1_mos) * (model1_2_mos - 4) / 100
    return mos_all


def model1_multi(dir_path):
    ret = {}
    for i in list_all_files(dir_path):
        try:
            mos = model1_single(f'{dir_path}/{i}')
            ret[i] = mos
            # print(i, mos)
        except:
            print(f'---process {i} fail---')
    return ret


if __name__ == '__main__':
    t = time.time()
    res = model1_multi('C:/Users/WXX/Desktop/数据集样例/video quality')
    t = time.time() - t
    print(f'total {t} seconds, averge {t/115} seconds')
