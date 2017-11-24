import math
import numpy as np
import pandas as pd

# VBR coefficient
HMEH264Squality = np.array([[4.427, 3.336, 41.198, 7.035],
                            [4.592, 3.470, 39.146, 5.913],
                            [4.684, 3.497, 37.141, 5.352],
                            [4.799, 3.935, 37.195, 5.143],
                            [4.839, 4.236, 36.362, 5.085],
                            [4.839, 4.236, 36.362, 5.085],
                            [4.592, 3.470, 39.146, 5.913]])
# PPI coefficient for TV
HMESqualityTVPPI = np.array([[4.8, 227.2, -0.696, 1.95, 1, 4.68, 1],
                             [4.8, 227.2, -0.696, 1.95, 1, 4.7, 1],
                             [4.8, 227.2, -0.696, 2.532, 1, 4.75, 1],
                             [4.8, 227.2, -0.696, 2, 1, 5, 1],
                             [4.8, 227.2, -0.696, 1.86, 1, 5, 1],
                             [4.8, 227.2, -0.696, 1.75, 1, 5, 1],
                             [4.8, 227.2, -0.735, 1.95, 1, 4.7, 1]])
# Coefficient for temporal pooling
HMESmooth = np.array([9.0, 13.0, 12.0, 6.0, 5.0, 35.0])

# VBR HEVC vs H264
HMEH2642H265 = np.array([-0.0644, 1.3794, -0.3111])

RESOLUTION_360P_480P_DIVIDE = (640 * 360 + 854 * 480) / 2
RESOLUTION_480P_720P_DIVIDE = (854 * 480 + 1280 * 720) / 2
RESOLUTION_720P_1080P_DIVIDE = (1280 * 720 + 1920 * 1080) / 2
RESOLUTION_1080P_2K_DIVIDE = (1920 * 1080 + 2560 * 1440) / 2
RESOLUTION_2K_4K_DIVIDE = (2560 * 1440 + 3840 * 2160) / 2

# normalized resolution
Resolution1 = np.array([[640, 360],
                        [853, 480],
                        [1280, 720],
                        [1920, 1080],
                        [2560, 1440],
                        [3840, 2160],
                        [720, 576]])


def model2_1(file_path, number_of_vidoe_sequence=1, device='TV', frame_rate=30):
    """
    % Calculate the video MOS with extracted parameters
    % ff_mos: the MOS calculated by the parameters extracted with FF_PROBE
    % ff_qp: the average qp extracted with FF_PROBE
    % hw_mow: the MOS calculated by the parameters extracted with Huawei decoder
    % hw_qp: the average qp extracted with Huawei decoder
    % sk_ratio: average skip ratio extracted with FF_PROBE
    % mv_avg: average motion vector extracted with FF_PROBE
    % rsl: normalized resolution, used to select coefficient

    """

    if device == 'Phone':
        screen_size = 6
    else:
        screen_size = 42

    resolution_item = ['360P', '480P', '720P', '1080P', '2K', '4K', 'PAL']
    codec_type = ['H264', 'H265']

    # resolution type
    p360 = 1
    p480 = 2
    p720 = 3
    p1080 = 4
    p2k = 5
    p4k = 6
    p576 = 7

    color_type = ['b', 'g', 'r', 'b', 'g', 'r', 'b']

    ff_mos = 0
    ff_qp = 0
    sk_ratio = 0
    mv_avg = 0
    rsl = 0
    csv_data = pd.read_csv(file_path, names=['frame', 'pkt_pts', 'pkt_size', 'width', 'height', 'pict_type',
                                             'skip_ratio', 'qp_min', 'qp_max', 'qp_avg', 'mv0_min', 'mv0_max',
                                             'mv0_avg', 'mv1_min', 'mv1_max', 'mv1_avg'])

    width = csv_data['width']
    height = csv_data['height']
    qb_list = csv_data['qp_avg']
    frame_type_list = csv_data['pict_type']
    skip_ratio = csv_data['skip_ratio']
    mv0 = csv_data['mv0_avg']
    csv_data = csv_data.dropna(axis=1, how='all')
    number_of_frames, number_of_params = csv_data.shape
    if number_of_params > 13:
        mv1 = csv_data['mv1_avg']
    else:
        mv1 = mv0
    pixels = height[0] * width[0]
    if height[0] == 576:
        ppi_v = p576
    elif pixels < RESOLUTION_360P_480P_DIVIDE:
        ppi_v = p360
    elif pixels < RESOLUTION_480P_720P_DIVIDE:
        ppi_v = p480
    elif pixels < RESOLUTION_720P_1080P_DIVIDE:
        ppi_v = p720
    elif pixels < RESOLUTION_1080P_2K_DIVIDE:
        ppi_v = p1080
    elif pixels < RESOLUTION_2K_4K_DIVIDE:
        ppi_v = p2k
    else:
        ppi_v = p4k

    count_i = 0
    count_p = 0
    count_b = 0
    count_all = 0
    count_key = 0
    # currently only I/P frames will be counted.

    # QP list for different frame type
    frame_qp = []
    frame_i_qp = []
    frame_p_qp = []
    frame_b_qp = []
    frame_skip_ratio = []

    # Temperal pooling for frame quality.
    frame_quality = []
    data_smooth = []
    for i in range(number_of_frames):
        # todo
        if frame_type_list[i] == 'I':
            # print('I')
            if qb_list[i] > 2:
                count_i += 1
                count_all += 1
                count_key += 1

            else:
                count_all += 1
        elif frame_type_list[i] == 'P':
            # print('P')
            if qb_list[i] > 2:
                pass
            else:
                count_all += 1
        elif frame_type_list[i] == 'B':
            # print('B')
            if qb_list[i] > 2:
                pass
            else:
                count_all += 1
        else:
            count_all += 1
            print('other')

    return ff_mos, ff_qp, sk_ratio, mv_avg, rsl


def get_squality_mos_vbr(media_width, media_height, screen_size, qp, ppi_v: list, v: list):
    """
    % calculated the Frame quality based on frmae qp.
    % squality is the frame quality
    % media_width, media_height, screen_size and qp get from the extract
    % parameters
    % ppi_v: PPI coefficient
    % v: coefficient

    """

    ppi = math.sqrt(media_width * media_width + media_height * media_height) / screen_size

    display_mos = ppi_v[0] * (1 - 1 / (1 + pow(ppi / (ppi_v[1] * pow(screen_size, ppi_v[2])), ppi_v[3])))
    display_mos = min([5, display_mos])
    display_mos = max([1, display_mos])

    squality = display_mos - (display_mos - 1) / (1 + math.exp((v[2] - qp) / v[3]))
    squality = min([squality, display_mos])
    squality = max([1, squality])
    return squality


if __name__ == '__main__':
    fpath = 'C:/Users/WXX/Desktop/model2_tool/Ori_data/ffprobe_csv_huawei/1.csv'
    model2_1(fpath)
