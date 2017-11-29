from model1.model1_1_prep import model1_1_fbf_var
import numpy as np

PARAMS = [6.263614795, -4.85845311, 11.35721636, -12.34011446, 0.006377201, -0.001129199, 0.272345205, 0.015391019,
          -1.3273304e-7, 1.225635116e-6, 2.430240304]


def model1_1(file_path=None, width=None, height=None, qp_list=None, frame_type_list=None, skip_ratio=None, mv0=None,
             mv1=None, bit_size_list=None, qp_max_list=None, qp_min_list=None, frame_rate=None, bit_rate=None):
    x_data = model1_1_fbf_var(file_path=file_path, width=width, height=height, qp_list=qp_list,
                              frame_type_list=frame_type_list, skip_ratio=skip_ratio, mv0=mv0, mv1=mv1,
                              bit_size_list=bit_size_list, qp_max_list=qp_max_list, qp_min_list=qp_min_list,
                              frame_rate=frame_rate, bit_rate=bit_rate)

    qp_fr_ipt = PARAMS[0] + PARAMS[1] * (x_data[0] / 51) ** PARAMS[2] + PARAMS[3] / x_data[1] + \
                PARAMS[4] * x_data[2] + PARAMS[5] * (x_data[3] - x_data[4])

    cpx_ipt = min(np.sqrt(x_data[5] / x_data[6]) + PARAMS[6] * x_data[7], 1)
    motion_ipt = PARAMS[7] * x_data[8] * (1 - x_data[1] / 30)
    kfr_ipt = PARAMS[8] * x_data[9] + PARAMS[9]
    q_cod = kfr_ipt * np.exp(PARAMS[10] * (qp_fr_ipt + cpx_ipt + motion_ipt))
    res = min(max(q_cod, 1), 5)
    return res


if __name__ == '__main__':
    mos = model1_1("C:/Users/WXX/Desktop/model1_tool/Ori_data/ffprobe_csv_huawei/1.csv", frame_rate=25, bit_rate=1188)
    print(mos)
