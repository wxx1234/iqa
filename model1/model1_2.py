import math


def model2_1():
    pass


def get_squality_mos_vbr(media_width, media_height, screen_size, qp, ppi_v: list, v: list):
    ppi = math.sqrt(media_width * media_width + media_height * media_height) / screen_size

    display_mos = ppi_v[0] * (1 - 1 / (1 + pow(ppi / (ppi_v[1] * pow(screen_size, ppi_v[2])), ppi_v[3])))
    display_mos = min([5, display_mos])
    display_mos = max([1, display_mos])

    squality = display_mos - (display_mos - 1) / (1 + math.exp((v[2] - qp) / v[3]))
    squality = min([squality, display_mos])
    squality = max([1, squality])
    return squality
