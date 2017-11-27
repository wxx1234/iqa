from model1.model1_1_prep import get_xdata
import numpy as np

PARAMS = [6.263614795, -4.85845311, 11.35721636, -12.34011446, 0.006377201, -0.001129199, 0.272345205, 0.015391019,
          -1.3273304e-7, 1.225635116e-6, 2.430240304]


def model1_1(filename, frame_rate, bit_rate):
    x_data = get_xdata(filename, frame_rate, bit_rate)

    qp_fr_ipt = PARAMS[0] + PARAMS[1] * (x_data[0] / 51) ** PARAMS[2] + PARAMS[3] / x_data[1] + \
                PARAMS[4] * x_data[2] + PARAMS[5] * (x_data[3] - x_data[4])

    cpx_ipt = min(np.sqrt(x_data[5] / x_data[6]) + PARAMS[6] * x_data[7], 1)
    motion_ipt = PARAMS[7] * x_data[8] * (1 - x_data[1] / 30)
    kfr_ipt = PARAMS[8] * x_data[9] + PARAMS[9]
    q_cod = kfr_ipt * np.exp(PARAMS[10] * (qp_fr_ipt + cpx_ipt + motion_ipt))
    res = min(max(q_cod, 1), 5)
    return res


if __name__ == '__main__':
    mos = model1_1("C:/Users/WXX/Desktop/model1_tool/Ori_data/ffprobe_csv_huawei/1.csv", 25, 1188)
    print(mos)
