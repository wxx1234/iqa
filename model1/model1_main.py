from model1.model1_1 import model1_1
from model1.model1_2 import model1_2


def model1(filename, frame_rate, bit_rate):
    model1_1_mos = model1_1(filename, frame_rate, bit_rate)
    model1_2_mos = model1_2(filename)
    mos_all = model1_2_mos - (5 - model1_1_mos) * (model1_2_mos - 4) / 100
    return mos_all


if __name__ == '__main__':
    mos = model1("C:/Users/WXX/Desktop/model1_tool/Ori_data/ffprobe_csv_huawei/1.csv", 25, 1188)
    print(mos)
