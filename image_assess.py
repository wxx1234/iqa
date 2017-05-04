# -*- coding:utf-8 -*-
"""
Usage:
    image_assess.py <video> [--frame=<frame>] [--errout=<format>] [--fast]

"""
import numpy
from numpy import *
import scipy.signal as signal
import datetime
from scipy import ndimage
from scipy.misc import imresize
import warnings
import cv2
from docopt import docopt
import os
import sys
import copy
import ctypes


def rgb2hsy(*rgb_tuple):  # inumpyut an image or its RGB components
    nargin = len(rgb_tuple)

    if nargin == 1:
        r = rgb_tuple[0]
        if (r.dtype == 'uint8'):
            r = (r + 0.0) / 255
        elif (r.dtype == 'uint16'):
            r = (r + 0.0) / 65535

    elif nargin == 3:
        r = rgb_tuple[0]
        g = rgb_tuple[1]
        b = rgb_tuple[2]
        if (r.dtype == 'uint8'):
            r = (r + 0.0) / 255
        elif (r.dtype == 'uint16'):
            r = (r + 0.0) / 65535

        if (g.dtype == 'uint8'):
            g = (g + 0.0) / 255
        elif (g.dtype == 'uint16'):
            g = (g + 0.0) / 65535

        if (b.dtype == 'uint8'):
            b = (b + 0.0) / 255
        elif (b.dtype == 'uint16'):
            b = (b + 0.0) / 65535

    else:
        raise Exception("Wrong number of inumpyut arguments.")

    threeD = (len(r.shape) == 3)  # Determine if inumpyut includes a 3-D array

    if threeD:
        g = r[:, :, 1]
        b = r[:, :, 2]
        r = r[:, :, 0]
        siz = r.shape
        r = r[:]
        g = g[:]
        b = b[:]

    elif nargin == 1:
        g = r[:, 1]
        b = r[:, 2]
        r = r[:, 0]
        siz = r.shape

    else:
        if not (r.shape == g.shape == b.shape):
            raise Exception("R,G,B must all be the same size.")
        siz = r.shape
        r = r.reshape(1, -1)
        g = g.reshape(1, -1)
        b = b.reshape(1, -1)

    # Here be the algorithm

    # luminance
    y = 0.2125 * r + 0.7154 * g + 0.0721 * b

    C1 = numpy.zeros(y.shape)
    C2 = numpy.zeros(y.shape)
    C = numpy.zeros(y.shape)

    # hue
    C1 = r - 0.5 * g - 0.5 * b
    C2 = -numpy.sqrt(3.0) / 2.0 * g + numpy.sqrt(3.0) / 2.0 * b

    C = numpy.sqrt(C1 ** 2 + C2 ** 2 * 1.0)

    s = numpy.zeros(y.shape)
    h = numpy.zeros(y.shape)
    h[C == 0] = 0

    # faster vectorized expressions which
    # may speed up total execution by a factor of approx 2.5
    # thomas knudsen, thk@kms.dk 2003-11-24
    indic = ((C != 0) & (C2 <= 0))

    h[indic] = numpy.arccos(C1[indic] * 1.0 / C[indic])

    indic = ((C != 0) & (C2 > 0))
    h[indic] = 2 * numpy.pi - numpy.arccos(C1[indic] * 1.0 / C[indic])

    #

    # saturation

    # k=h./(pi/3.0);
    # k=floor(k);
    # hstar=h - k.*(pi/3.0);


    # s=(2.0*C.*sin(-hstar + (2.0*pi/3.0)))/sqrt(3.0);

    # simpler saturation expression which produces the same as above
    # s=numpy.fmax(r,numpy.fmax(g,b)) + numpy.fmax(-r,fmax(-g,-b))
    s = numpy.fmax(numpy.fmax(g, b), r)
    s = numpy.fmax(numpy.fmax(-g, -b), -r) + s

    # convert h to degrees
    h = h / numpy.pi
    h = h * 180.0

    # here ends the algorithm
    if (threeD or nargin == 3):
        h = h.reshape(siz)
        s = s.reshape(siz)
        y = y.reshape(siz)

        h = numpy.dstack((h, s, y))


    else:
        h = numpy.column_stack((h, s, y))
        pass

    return h


def color_compute(img):
    def sa_w_circ_mean_var(hue, saturation, mode):
        hue = hue / 180 * numpy.pi
        if mode == 1:
            sat = saturation
            cosI1 = numpy.cos(hue)
            sinI1 = numpy.sin(hue)

            w_sum_cos = numpy.sum(sat * cosI1)

            w_sum_sin = numpy.sum(sat * sinI1)

            mu1 = numpy.arctan2(w_sum_sin, w_sum_cos)

            hue_mean_len1 = numpy.sqrt(w_sum_sin ** 2 + w_sum_cos ** 2) / (len(hue))
            var1 = 1 - hue_mean_len1
            if mu1 < 0:
                mu1 = mu1 + 2 * numpy.pi
            mu1 = mu1 / numpy.pi * 180
            var1 = var1 / numpy.pi * 180
        return mu1, hue_mean_len1, var1
        # Ref: Circular Statistics Applied to Colour Images -Allan Hanbury

    # declaration
    bin_v_s = numpy.array(list(range(1, 101))) / 100.0
    bin_h = numpy.array(list(range(0, 361, 20)))
    hueMean = numpy.zeros(len(bin_v_s) + 1)
    hueVar = numpy.zeros(len(bin_v_s) + 1)
    hue_mean_len = numpy.zeros(len(bin_v_s) + 1)

    # Convert the RGB to IHSL Space
    hsy = rgb2hsy(img)
    hue = hsy[:, :, 0]
    s = hsy[:, :, 1]
    v = hsy[:, :, 2]
    vecImg = numpy.column_stack((hue.reshape(-1, 1), s.reshape(-1, 1), v.reshape(-1, 1)))

    # sort the img hue and saturation according to Luminance
    ind = numpy.argsort(vecImg[:, 2])

    sortImg_v = vecImg[ind, :]
    group = sortImg_v[:, 2].copy() * 100
    group = numpy.floor(group)
    group = group.astype(int)

    vFrq = numpy.bincount(group)
    vFrq.resize(101)
    v_tick = numpy.zeros(101, int)
    v_tick[0] = vFrq[0]
    for i1 in range(len(vFrq) - 1):
        v_tick[i1 + 1] = v_tick[i1] + vFrq[i1 + 1]
    v_tick.dtype = "int"
    # extract the hue slice and saturation slice from certain luminance plate
    p = 0
    for j in range(len(v_tick)):
        if v_tick[j] == 0:
            p += 1
            continue
        elif j == 0:
            hueSlice = sortImg_v[0:v_tick[j], 0]
            saSlice = sortImg_v[0:v_tick[j], 1]
        elif v_tick[j] > v_tick[j - 1]:
            hueSlice = sortImg_v[v_tick[j - 1]:v_tick[j], 0]
            saSlice = sortImg_v[v_tick[j - 1]:v_tick[j], 1]

        else:
            p += 1
            continue

        hueMean[p], hue_mean_len[p], hueVar[p] = sa_w_circ_mean_var(hueSlice, saSlice, 1)
        p += 1

    hbin = (hueMean / 20).astype(int)

    hn = numpy.bincount(hbin)

    hn = numpy.zeros(18)
    for i1 in range(0, len(hbin)):
        hn[hbin[i1]] = hn[hbin[i1]] + hue_mean_len[i1]
    color_richness = 0

    for i1 in range(0, len(hn)):

        if hn[i1] != 0:
            color_richness = color_richness - hn[i1] * numpy.log(hn[i1] / 100)
    hueMean_overall = hueMean
    hueVar_overall = hueVar

    return color_richness


def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)


warnings.filterwarnings("ignore")

numpy.set_printoptions(threshold=numpy.nan)


def rgb2gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return around(numpy.array(0.29900 * r + 0.58700 * g + 0.11400 * b))


def mat2gray(mat):
    mmax = numpy.max(mat)
    mmin = numpy.min(mat)
    mat = (mat - mmin) / (mmax - mmin)
    return mat


def matlab_style_gauss2d(shape=(3, 3), sigma=1.0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = numpy.ogrid[-m:m + 1, -n:n + 1]
    h = numpy.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < numpy.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def tm_kernel(N, p):
    # N×N为图像的大小
    # p最高阶数
    T = numpy.mat(zeros((p + 1, N)))
    T[0, 0] = 1
    # 论文中的公式(18)求出每个阶数对应x=0处的函数值
    for n in range(1, p + 1):
        c0 = -numpy.sqrt((N - n) / (N + n) + 0j) * numpy.sqrt((2 * n + 1) / (2 * n - 1) + 0j)
        T[n, 0] = ((T[n - 1, 0] + 0j) * c0).real
        # print(c0, T[n - 1, 0] + 0j, T[n, 0])
    # 论文中的公式(19)求出每个阶数对应x=1处的函数值
    for n in range(p + 1):
        c1 = 1 + (n * (1 + n) / (1 - N))
        T[n, 1] = c1 * T[n, 0]
    # 下面利用论文中的公式(20)迭代求出其他x处的函数值
    for x in range(2, int(N / 2)):
        for n in range(p + 1):
            c3 = (-n * (n + 1) - (2 * x - 1) * (x - N - 1) - x) / (x * (N - x))
            c4 = (x - 1) * (x - N - 1) / (x * (N - x))
            T[n, x] = c3 * T[n, x - 1] + c4 * T[n, x - 2]
    # 下面利用对称关系(文中公式22)求右半部分的函数值
    for n in range(p + 1):
        for x in range(int(N / 2) + 1, N + 1):
            T[n, x - 1] = ((-1) ** n) * T[n, N - x]
    return T


def tm_kernel_blur(N, p):
    # N×N为图像的大小
    # p最高阶数
    T = numpy.mat(zeros((p + 1, N)))
    T[0, 0] = 1 / numpy.sqrt(N)
    # 论文中的公式(18)求出每个阶数对应x=0处的函数值
    for n in range(1, p + 1):
        c0 = -numpy.sqrt((N - n) / (N + n) + 0j) * numpy.sqrt((2 * n + 1) / (2 * n - 1) + 0j)
        T[n, 0] = ((T[n - 1, 0] + 0j) * c0).real
        # print(c0, T[n - 1, 0] + 0j, T[n, 0])
    # 论文中的公式(19)求出每个阶数对应x=1处的函数值
    for n in range(p + 1):
        c1 = 1 + (n * (1 + n) / (1 - N))
        T[n, 1] = c1 * T[n, 0]
    # 下面利用论文中的公式(20)迭代求出其他x处的函数值
    for x in range(2, int(N / 2)):
        for n in range(p + 1):
            c3 = (-n * (n + 1) - (2 * x - 1) * (x - N - 1) - x) / (x * (N - x))
            c4 = (x - 1) * (x - N - 1) / (x * (N - x))
            T[n, x] = c3 * T[n, x - 1] + c4 * T[n, x - 2]
    # 下面利用对称关系(文中公式22)求右半部分的函数值
    for n in range(p + 1):
        for x in range(int(N / 2) + 1, N + 1):
            T[n, x - 1] = ((-1) ** n) * T[n, N - x]
    return T


def tm_function(img, ord):
    nx, ny = img.shape
    n = ord
    m = ord
    tx = tm_kernel_blur(nx, n)
    ty = tm_kernel_blur(ny, m)
    mmt = numpy.dot(numpy.dot(ty, double(img)), numpy.transpose(tx))
    return mmt


def var2(mtx, mn=None):
    if not mn:
        mn = numpy.mean(mean(mtx))
    if isreal(mtx.all()):
        res = numpy.sum(numpy.sum(numpy.multiply(abs(mtx - mn), numpy.abs(mtx - mn)))) / max(
            (numpy.prod(mtx.shape) - 1), 1)
    else:
        res = complex(numpy.sum(numpy.sum(numpy.multiply(real(mtx - mn), real(mtx - mn)))),
                      numpy.sum(numpy.sum(numpy.multiply((mtx - mn).imag, (mtx - mn).imag))))
        res /= numpy.max((numpy.prod(mtx.shape) - 1), 1)
    return res


def block_func(img_dst, img_gradient, blkSZ):
    m, n = img_dst.shape
    img_dst = numpy.double(img_dst)
    img_gradient = numpy.double(img_gradient)
    # print(img_gradient)
    rb = blkSZ
    cb = blkSZ
    r = int(numpy.floor(m / rb))
    c = int(numpy.floor(n / cb))
    signal_block = numpy.mat(numpy.zeros((rb * cb, r * c)))
    var_block = numpy.mat(numpy.zeros((1, r * c)))
    k = 0
    # var_block_1 = mat(zeros((1, r * c)))
    # var_block_2 = mat(zeros((1, r * c)))
    # var_block_3 = mat(zeros((1, r * c)))
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            image_temp = img_gradient[rb * (i - 1):rb * i, cb * (j - 1):cb * j]
            var_temp = img_dst[rb * (i - 1):rb * i, cb * (j - 1):cb * j]
            signal_block[:, k] = numpy.transpose(image_temp).reshape((64, 1))
            var_block[:, k] = var2(var_temp)
            # endG_1 = matlab_style_gauss2d(shape=(3, 3), sigma=0.1)
            # Ig_1 = correlate(image_temp, endG_1, output=numpy.float64, mode='constant')
            # endG_2 = matlab_style_gauss2d(shape=(3, 3), sigma=0.25)
            # Ig_2 = correlate(image_temp, endG_2, output=numpy.float64, mode='constant')
            # endG_3 = matlab_style_gauss2d(shape=(3, 3), sigma=0.45)
            # Ig_3 = correlate(image_temp, endG_3, output=numpy.float64, mode='constant')
            # signal_block[:, k] = numpy.transpose(image_temp).reshape((64, 1))
            # var_block_1[:, k] = var2(Ig_1)
            # var_block_2[:, k] = var2(Ig_2)
            # var_block_3[:, k] = var2(Ig_3)
            k += 1
    return signal_block, var_block, r, c


def sdsp(image):
    sigmaF = 6.2
    omega0 = 0.002
    sigmaD = 114
    sigmaC = 0.25
    oriRows, oriCols, junk = image.shape
    dsImage = zeros((256, 256, junk))
    dsImage[:, :, 0] = imresize(image[:, :, 0], (256, 256), mode='F')
    dsImage[:, :, 1] = imresize(image[:, :, 1], (256, 256), mode='F')
    dsImage[:, :, 2] = imresize(image[:, :, 2], (256, 256), mode='F')

    lab = rgbtolab(dsImage)

    LChannel = lab[:, :, 0]
    AChannel = lab[:, :, 1]
    BChannel = lab[:, :, 2]

    LFFT = numpy.fft.fft2(double(LChannel))
    AFFT = numpy.fft.fft2(double(AChannel))
    BFFT = numpy.fft.fft2(double(BChannel))

    rows, cols, junk = dsImage.shape
    LG = logGabor(rows, cols, omega0, sigmaF)
    FinalLResult = numpy.real(numpy.fft.ifft2(numpy.multiply(LFFT, LG)))
    FinalAResult = numpy.real(numpy.fft.ifft2(numpy.multiply(AFFT, LG)))
    FinalBResult = numpy.real(numpy.fft.ifft2(numpy.multiply(BFFT, LG)))

    SFMap = numpy.sqrt(FinalLResult ** 2 + FinalAResult ** 2 + FinalBResult ** 2)

    coordinateMtx = numpy.zeros((rows, cols, 2))
    c0 = numpy.arange(1, rows + 1).reshape(256, 1)
    c1 = numpy.arange(1, cols + 1).reshape(1, 256)
    coordinateMtx[:, :, 0] = numpy.tile(c0, (1, cols))
    coordinateMtx[:, :, 1] = numpy.tile(c1, (rows, 1))

    centerY = rows / 2
    centerX = cols / 2
    centerMtx = numpy.zeros((rows, cols, 2))
    centerMtx[:, :, 0] = numpy.ones((rows, cols)) * centerY
    centerMtx[:, :, 1] = numpy.ones((rows, cols)) * centerX
    SDMap = numpy.exp(-numpy.sum((coordinateMtx - centerMtx) ** 2, 2) / sigmaD ** 2)
    maxA = numpy.max(AChannel)
    minA = numpy.min(AChannel)
    normalizedA = (AChannel - minA) / (maxA - minA)

    maxB = numpy.max(BChannel)
    minB = numpy.min(BChannel)
    normalizedB = (BChannel - minB) / (maxB - minB)

    labDistSquare = normalizedA ** 2 + normalizedB ** 2
    SCMap = 1 - numpy.exp(-labDistSquare / (sigmaC ** 2))
    VSMap = numpy.multiply(numpy.multiply(SFMap, SDMap), SCMap)
    VSMap = imresize(VSMap, (oriRows, oriCols), mode='F')
    VSMap = mat2gray(VSMap)
    return VSMap


def rgbtolab(image):
    normalizedR = image[:, :, 0] / 255
    normalizedG = image[:, :, 1] / 255
    normalizedB = image[:, :, 2] / 255

    RSmallerOrEqualto4045 = normalizedR <= 0.04045
    RGreaterThan4045 = 1 - RSmallerOrEqualto4045
    tmpR = numpy.multiply(normalizedR / 12.92, RSmallerOrEqualto4045)
    tmpR = tmpR + numpy.multiply(power((normalizedR + 0.055) / 1.055, 2.4), RGreaterThan4045)

    GSmallerOrEqualto4045 = normalizedG <= 0.04045
    GGreaterThan4045 = 1 - GSmallerOrEqualto4045
    tmpG = numpy.multiply(normalizedG / 12.92, GSmallerOrEqualto4045)
    tmpG = tmpG + numpy.multiply(power((normalizedG + 0.055) / 1.055, 2.4), GGreaterThan4045)

    BSmallerOrEqualto4045 = normalizedB <= 0.04045
    BGreaterThan4045 = 1 - BSmallerOrEqualto4045
    tmpB = numpy.multiply(normalizedB / 12.92, BSmallerOrEqualto4045)
    tmpB = tmpB + numpy.multiply(power((normalizedB + 0.055) / 1.055, 2.4), BGreaterThan4045)

    X = tmpR * 0.4124564 + tmpG * 0.3575761 + tmpB * 0.1804375
    Y = tmpR * 0.2126729 + tmpG * 0.7151522 + tmpB * 0.0721750
    Z = tmpR * 0.0193339 + tmpG * 0.1191920 + tmpB * 0.9503041

    epsilon = 0.008856
    kappa = 903.3

    Xr = 0.9642
    Yr = 1.0
    Zr = 0.8251

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    xrGreaterThanEpsilon = xr > epsilon
    xrSmallerOrEqualtoEpsilon = 1 - xrGreaterThanEpsilon
    fx = numpy.multiply(power(xr, 1.0 / 3.0), xrGreaterThanEpsilon)
    fx = fx + numpy.multiply((kappa * xr + 16.0) / 116.0, xrSmallerOrEqualtoEpsilon)

    yrGreaterThanEpsilon = yr > epsilon
    yrSmallerOrEqualtoEpsilon = 1 - yrGreaterThanEpsilon
    fy = numpy.multiply(power(yr, 1.0 / 3.0), yrGreaterThanEpsilon)
    fy = fy + numpy.multiply((kappa * yr + 16.0) / 116.0, yrSmallerOrEqualtoEpsilon)

    zrGreaterThanEpsilon = zr > epsilon
    zrSmallerOrEqualtoEpsilon = 1 - zrGreaterThanEpsilon
    fz = numpy.multiply(power(zr, 1.0 / 3.0), zrGreaterThanEpsilon)
    fz = fz + numpy.multiply((kappa * zr + 16.0) / 116.0, zrSmallerOrEqualtoEpsilon)

    rows, cols, junk = image.shape
    labImage = zeros((rows, cols, 3))
    labImage[:, :, 0] = 116.0 * fy - 16.0
    labImage[:, :, 1] = 500.0 * (fx - fy)
    labImage[:, :, 2] = 200.0 * (fy - fz)
    return labImage


def logGabor(rows, cols, omega0, sigmaF):
    tmpc = numpy.arange(1, cols + 1)
    tmpr = numpy.arange(1, rows + 1)
    u1, u2 = numpy.meshgrid((tmpc - (int(cols / 2) + 1)) / (cols - mod(cols, 2)),
                            (tmpr - (int(rows / 2) + 1)) / (rows - mod(rows, 2)))
    mask = ones((rows, cols))
    for row_index in range(rows):
        for col_index in range(cols):
            if u1[row_index, col_index] ** 2 + u2[row_index, col_index] ** 2 > 0.25:
                mask[row_index, col_index] = 0
    u1 = numpy.multiply(u1, mask)
    u2 = numpy.multiply(u2, mask)

    u1 = numpy.fft.ifftshift(u1)
    u2 = numpy.fft.ifftshift(u2)
    radius = numpy.sqrt(numpy.multiply(u1, u1) + numpy.multiply(u2, u2))
    radius[0, 0] = 1

    LG = numpy.exp((-(numpy.log(radius / omega0)) ** 2) / (2 * (sigmaF ** 2)))
    LG[0, 0] = 0
    return LG


def bible_func(img_dst, img_gradient, img_orgdst, order=None, blkSZ=8):
    if not order:
        order = blkSZ - 1

    energy = []
    # tt = datetime.datetime.now()
    img_block, var_gray, Rnum, Cnum = block_func(img_dst, img_gradient, blkSZ)
    # print('bb:{}'.format((datetime.datetime.now() - tt).total_seconds()))

    saliency_map = sdsp(img_orgdst)
    saliency_map = mat2gray(imresize(saliency_map, (Rnum, Cnum), mode='F'))

    # tt = datetime.datetime.now()
    # print(img_block.shape[1], img_block.shape, Rnum, Cnum)

    print(img_block.shape)
    # ttt = datetime.datetime.now()
    for k in range(img_block.shape[1]):
        block = transpose(img_block[:, k].reshape((blkSZ, blkSZ)))
        mmt = tm_function(block, order)
        mmt[0, 0] = 0
        energy.append(numpy.sum(numpy.multiply(mmt, mmt)))
    # print((datetime.datetime.now() - ttt).total_seconds())
    # print('bb1:{}'.format((datetime.datetime.now() - tt).total_seconds()))

    saliency_weight = reshape(saliency_map, (1, Rnum * Cnum))
    score_1 = numpy.sum(numpy.multiply(energy, saliency_weight)) / numpy.sum(numpy.multiply(var_gray, saliency_weight))
    # score_1 = numpy.sum(energy) / numpy.sum(var_gray)
    # energy_1 = sum(energy)
    # var_1 = sum(var_gray)

    return score_1


def lum_contrast(image, bit_depth):
    """

    :param image:
    :param bit_depth:
    TO DO :bit_depth not finished
    :return:
    """
    lmin = 0
    lmax = 2 ** bit_depth - 1
    x, f = numpy.unique(image, return_counts=True)
    f = f / sum(f)
    t, md, mb = compute_threshold(f, x, 1)
    md_center = md / sum(f[:t])
    mb_center = mb / sum(f[t:])
    m_center = (md + mb) / sum(f)
    ct = abs(mb_center - md_center) / m_center
    a = 1.5
    sigma = 0.05
    contrast = exp(a / (1 + sigma / ct)) / exp(a)
    return contrast


def compute_threshold(f_norm, x, eta):
    md = zeros(len(x))
    mb = zeros(len(x))
    for i in range(len(x)):
        md[i] = sum(numpy.multiply(f_norm[:i + 1], x[:i + 1]))
        mb[i] = sum(numpy.multiply(f_norm[i + 1:], x[i + 1:]))
    d_value = abs(md - eta * mb)
    numpy.where(d_value == min(d_value))
    t_position = numpy.where(d_value == min(d_value))
    return t_position[0][0], md[t_position][0], mb[t_position][0]


def color_dynamic(color_channel, bit_depth):
    pd_ch = log10((numpy.max(color_channel) + 1) / (numpy.min(color_channel) + 1)) / log10(2 ** bit_depth)
    return pd_ch


def sobel(img):
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1)
    mag = dx ** 2 + dy ** 2
    # mag = numpy.hypot(dx, dy)
    # mag *= 255.0 / numpy.max(mag)
    mag = sobel_thin(mag)
    return mag


def sobel_thin(b):
    cut_off = 4 * mean(b)
    m, n = b.shape
    eee = zeros((m, n))
    for r in range(m):
        for c in range(n):
            if c < 1:
                b1 = True
            else:
                b1 = b[r, c - 1] < b[r, c]
            if c + 2 > n:
                b2 = True
            else:
                b2 = b[r, c] > b[r, c + 1]
            if r < 1:
                b3 = True
            else:
                b3 = b[r - 1, c] < b[r, c]
            if r + 2 > m:
                b4 = True
            else:
                b4 = b[r, c] > b[r + 1, c]
            eee[r, c] = (b[r, c] > cut_off) and ((b1 and b2) or (b3 and b4))
    return eee


def compute_mmt(a, b, c):
    aa = copy.deepcopy(a)
    bb = copy.deepcopy(b)
    cc = copy.deepcopy(c)
    for i in range(8):
        bb[:, i] *= aa[0, i]
    for i in range(8):
        aa[0, i] = sum(bb[i, :])
    s = 0
    for i in range(8):
        s += aa[0, i] + cc[i, 0]
    return s


def cut_image(im, h, w, thres=500):
    if h > thres:
        r = im[(h - thres) / 2:(h + thres) / 2, :, 0]
        g = im[(h - thres) / 2:(h + thres) / 2, :, 1]
        b = im[(h - thres) / 2:(h + thres) / 2, :, 2]
    else:
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
    if w > thres:
        r = r[:(w - thres) / 2:(w + thres) / 2]
        g = g[:(w - thres) / 2:(w + thres) / 2]
        b = b[:(w - thres) / 2:(w + thres) / 2]
    else:
        r = r[:, :]
        g = g[:, :]
        b = b[:, :]
    he, wi = r.shape
    tmp = numpy.zeros((he, wi, 3))
    tmp[:, :, 0] = r
    tmp[:, :, 1] = g
    tmp[:, :, 2] = b
    return tmp


class ImageQualityAssess:
    def __init__(self, img_path=None, im=None):
        if im is None:
            self.im = numpy.asarray(cv2.imread(img_path))
        else:
            self.im = im
        self.gray = rgb2gray(self.im)
        self.im_double = self.gray / 255

        self.h, self.w = self.gray.shape
        # self.middle_im = self.im_double
        # if self.h > 512:
        #     self.middle_im = self.middle_im[int((self.h - 128) / 2):int((self.h + 128) / 2), :]
        # if self.w > 512:
        #     self.middle_im = self.middle_im[:, int((self.w - 128) / 2):int((self.w + 128) / 2)]
        self.clib = ctypes.WinDLL(os.getcwd() + '/clib.dll')

    def block(self):
        N, M = self.im_double.shape
        Tx = tm_kernel(8, 7)
        Ty = numpy.transpose(tm_kernel(8, 7))
        a = int(N / 8)
        b = int(M / 8)
        c = 0.00000001
        MMT1 = mat(zeros((8, 8)))
        MMT2 = mat(zeros((8, 8)))
        su1 = []
        su2 = []
        # tt = datetime.datetime.now()
        for i in range(1, a, 4):
            for j in range(1, b, 4):
                BK1 = numpy.mat(self.im_double[(i - 1) * 8:i * 8, (j - 1) * 8 + 4:j * 8 + 4])
                BK2 = numpy.mat(self.im_double[(i - 1) * 8 + 4:i * 8 + 4, (j - 1) * 8:j * 8])
                BK3 = numpy.mat(self.im_double[(i - 1) * 8:i * 8, (j - 1) * 8:j * 8])

                E = numpy.sum(BK3) / 64
                SSM = 0
                for u in range(1, 9):
                    for v in range(1, 9):
                        SSM += (BK3[u - 1, v - 1] - E) ** 2
                Tx = mat(Tx)
                Ty = mat(Ty)
                for s in range(8):
                    for t in range(8):
                        MMT1[s, t] = numpy.dot(numpy.dot(Tx[s, :], BK1), Ty[:, t])
                        MMT2[s, t] = numpy.dot(numpy.dot(Tx[s, :], BK2), Ty[:, t])

                q1 = numpy.sum(numpy.abs(MMT1[7, :])) / (numpy.sum(abs(MMT1)) - numpy.abs(MMT1[0, 0]) + c)
                q2 = numpy.sum(numpy.abs(MMT2[:, 7])) / (numpy.sum(abs(MMT2)) - numpy.abs(MMT2[0, 0]) + c)

                if q1 > 0.05:
                    q1 = 0.05
                if q2 > 0.05:
                    q2 = 0.05
                if SSM < numpy.float(5000.0 / (255.0 * 255.0)):
                    su1.append((q1 + q2) / 2)
                else:
                    su2.append((q1 + q2) / 2)
        # tt2 = datetime.datetime.now()
        # print((tt2 - tt).total_seconds())
        HP = numpy.mean(su1)
        HN = numpy.mean(su2)
        q = (HP * 0.8 + HN * 0.2)
        Q = log(1 - q) / log(0.95)
        return Q

    def blur(self, blkSZ=8, order=9, Tx=numpy.array([0.5, 0, -0.5])):
        if doc['--fast']:
            height, width = self.gray.shape
            w = ctypes.c_uint64(width)
            h = ctypes.c_uint64(height)
            arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
            arr = (ctypes.c_byte * len(arr))(*arr)
            blur = self.clib.blur
            blur.restype = ctypes.c_double
            blur_score = blur(w, h, ctypes.byref(arr))
            return blur_score
        else:
            tt = datetime.datetime.now()
            height, width = self.gray.shape
            thre = 800
            if height > thre:
                gray = self.gray[height / 2 - thre / 2:height / 2 + thre / 2, :]
            else:
                gray = self.gray
            if width > thre:
                gray = gray[:, width / 2 - thre / 2:width / 2 + thre / 2]
            else:
                gray = gray
            im = cut_image(self.im, height, width, thre)
            SZx, SZy = gray.shape
            # print(self.gray)
            Tx = Tx.reshape((3, 1))

            gx = signal.convolve(gray, numpy.transpose(Tx), 'same')
            gy = signal.convolve(gray, Tx, 'same')
            gx[:, 0] = gray[:, 1] - gray[:, 0]
            gx[:, SZy - 1] = gray[:, SZy - 1] - gray[:, (SZy - 2)]
            gy[0, :] = gray[1, :] - gray[0, :]
            gy[SZx - 1, :] = gray[SZx - 1, :] - gray[SZx - 2, :]

            img_gradient = (abs(gx) + abs(gy)) / 2
            return bible_func(gray, img_gradient, im, order, blkSZ)

    def contrast(self, bit_depth=8):
        global_contrast = lum_contrast(self.gray, bit_depth)
        rows, cols = self.gray.shape
        block_size = 64
        whole_block_rows = int(floor(rows / block_size))
        whole_block_cols = int(floor(cols / block_size))
        local_cra = []
        for i in range(whole_block_rows):
            for j in range(whole_block_cols):
                tmp = self.gray[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
                tmp_edge = sobel(tmp)
                if numpy.sum(tmp_edge) > 0.002 * block_size ** 2:
                    local_cra.append(lum_contrast(tmp, bit_depth))
        local_contrast = numpy.mean(local_cra)
        co = numpy.sqrt((global_contrast ** 2 + local_contrast ** 2) / 2)
        # print(co)
        pd_r = color_dynamic(self.im[:, :, 0], bit_depth)
        pd_g = color_dynamic(self.im[:, :, 1], bit_depth)
        pd_b = color_dynamic(self.im[:, :, 2], bit_depth)
        # print(pd_r,pd_g,pd_b)
        pd = 0.299 * pd_r + 0.587 * pd_g + 0.114 * pd_b
        # print(pd)
        total_contrast = co * pd
        # print(total_contrast)
        return total_contrast

    def exposure(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        exposure = self.clib.exposure
        exposure.restype = ctypes.c_double
        exposure_score = exposure(w, h, ctypes.byref(arr))
        return exposure_score

    def noise(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        noise = self.clib.noise
        noise.restype = ctypes.c_double
        noise_score = noise(w, h, ctypes.byref(arr))
        return noise_score

    def color(self):
        return color_compute(self.im)

    def blackout(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        blackout = self.clib.blackout
        blackout.restype = ctypes.c_bool
        blackout_score = blackout(w, h, ctypes.byref(arr))
        return blackout_score

    def blockloss(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        blockloss = self.clib.blockLoss
        blockloss.restype = ctypes.c_double
        blockloss_score = blockloss(w, h, ctypes.byref(arr))
        return blockloss_score

    def interlace(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        interlace = self.clib.interlace
        interlace.restype = ctypes.c_double
        interlace_score = interlace(w, h, ctypes.byref(arr))
        return interlace_score

    def pillarbox(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        pillarbox = self.clib.pillarbox
        pillarbox.restype = ctypes.c_double
        pillarbox_score = pillarbox(w, h, ctypes.byref(arr))
        return pillarbox_score

    def letterbox(self):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)
        letterbox = self.clib.letterbox
        letterbox.restype = ctypes.c_double
        letterbox_score = letterbox(w, h, ctypes.byref(arr))
        return letterbox_score

    def freezing(self, black_out, f_ps, p_gray):
        height, width = self.gray.shape
        w = ctypes.c_uint64(width)
        h = ctypes.c_uint64(height)
        arr = list(map(ctypes.c_byte, self.gray.reshape((width * height,)).astype(int)))
        arr = (ctypes.c_byte * len(arr))(*arr)

        pheight, pwidth = p_gray.shape
        pw = ctypes.c_uint64(pwidth)
        ph = ctypes.c_uint64(pheight)
        parr = list(map(ctypes.c_byte, p_gray.reshape((pwidth * pheight,)).astype(int)))
        parr = (ctypes.c_byte * len(arr))(*parr)

        freezing = self.clib.freezing
        freezing.restype = ctypes.c_bool
        freezing_score = freezing(w, h, ctypes.byref(arr), pw, ph, ctypes.byref(parr), ctypes.c_bool(black_out),
                                  ctypes.c_double(f_ps))
        return freezing_score


def is_error(blkout, blkloss, inlace, freez, blu, blk, nse):
    x = ''
    if blkout == 1:
        x += 'blackout;'
    if blkloss > 5:
        x += 'blockloss;'
    if inlace > 0.5:
        x += 'interlacing;'
    if freez == 1:
        x += 'freezing;'
    if blu < 30:
        x += 'bluriness;'
    if blk < 40:
        x += 'blockiness;'
    if nse > 35.0 / 3.0:
        x += 'noise;'
    if x == '':
        return x
    else:
        return x[:-1]


def openfile(fname):
    ext_line = 0
    if os.path.exists(fname):
        fi = open(fname, 'r')
        l = len(fi.readlines())
        if l > 2:
            fi.close()
            fi = open(fname, 'a')
            ext_line = l - 2
        else:
            fi.close()
            os.remove(fname)
    if ext_line == 0:
        fi = open(fname, 'a')
        fi.write(',level1,,,,,,,level2\n')
        fi.write(
            'No,blackout,blockloss,interlacing,freezing,blurriness,blockiness,noise,pillar-boxing,letter-boxing,exposure,contrast,color,timecost(s),err\n')
    return ext_line, fi


if __name__ == '__main__':
    doc = docopt(__doc__)
    videofile = doc['<video>']
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    iserrout = doc['--errout']
    if iserrout:
        if not os.path.exists(os.getcwd() + '/{}_err'.format(videofile)):
            os.mkdir(os.getcwd() + '/{}_err'.format(videofile))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frame number is:{}'.format(frame_num))
    # cap = cv2.VideoCapture('C:/Users/wangxh/Desktop/UHD_KPI_files/ttest/BambooWindow_Lossless.mp4')
    count = 0
    exist_line, f = openfile(os.getcwd() + '/{}log.csv'.format(videofile))
    err_line, err_f = openfile(os.getcwd() + '/{}errlog.csv'.format(videofile))

    # f.close()
    p_pic = None
    if doc['--frame']:
        exist_line = int(doc['--frame']) - 1
    print('{:<8}{:<10}{:<11}{:<11}{:<10}{:<12}{:<12}{:<7}{:<12}{}'.format('No', 'blackout', 'blockloss',
                                                                          'interlace', 'freezing',
                                                                          'blurriness', 'blockiness',
                                                                          'noise', 'timecost(s)', 'err'))
    while cap.isOpened():
        ret, frame = cap.read()
        if count != exist_line:
            count += 1
            continue
        if frame is None:
            break
        # cv2.imshow('window-name', frame)
        # cv2.imwrite("frame%d.jpg" % count, frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

        im = numpy.array(frame)
        im[:, :, 0] = frame[:, :, 2]
        im[:, :, 1] = frame[:, :, 1]
        im[:, :, 2] = frame[:, :, 0]

        iqa = ImageQualityAssess(im=im)
        # print('No.{}'.format(count + 1))
        # cv2.imshow('1', frame)
        # cv2.waitKey(0)

        t0 = datetime.datetime.now()
        blur = iqa.blur() * 10
        time_cost_blur = (datetime.datetime.now() - t0).total_seconds()
        print('blur={}     timecost:{}s'.format(round(blur, 3), time_cost_blur))

        t0 = datetime.datetime.now()
        block = iqa.block() * 100
        time_cost_block = (datetime.datetime.now() - t0).total_seconds()
        print('block={}     timecost:{}s'.format(round(block, 3), time_cost_block))

        t0 = datetime.datetime.now()
        contrast = iqa.contrast() * 100
        time_cost_contrast = (datetime.datetime.now() - t0).total_seconds()
        print('contrast={}     timecost:{}s'.format(round(contrast, 3), time_cost_contrast))

        t0 = datetime.datetime.now()
        exposure = iqa.exposure() / 2.55
        time_cost_exposure = (datetime.datetime.now() - t0).total_seconds()
        print('exposure={}     timecost:{}s'.format(round(exposure, 3), time_cost_exposure))

        t0 = datetime.datetime.now()
        noise = iqa.noise() * 10 / 3
        time_cost_noise = (datetime.datetime.now() - t0).total_seconds()
        print('noise={}     timecost:{}s'.format(round(noise, 3), time_cost_noise))

        t0 = datetime.datetime.now()
        color = iqa.color()
        time_cost_color = (datetime.datetime.now() - t0).total_seconds()
        print('color={}     timecost:{}s'.format(round(color, 3), time_cost_color))

        t0 = datetime.datetime.now()
        blackout = iqa.blackout()
        time_cost_blackout = (datetime.datetime.now() - t0).total_seconds()
        print('blackout={}     timecost:{}s'.format(round(blackout, 3), time_cost_blackout))

        t0 = datetime.datetime.now()
        blockloss = iqa.blockloss()
        time_cost_blockloss = (datetime.datetime.now() - t0).total_seconds()
        print('blockloss={}     timecost:{}s'.format(round(blockloss, 3), time_cost_blockloss))

        t0 = datetime.datetime.now()
        interlace = iqa.interlace()
        time_cost_interlace = (datetime.datetime.now() - t0).total_seconds()
        print('interlace={}     timecost:{}s'.format(round(interlace, 3), time_cost_interlace))

        t0 = datetime.datetime.now()
        pillarbox = iqa.pillarbox()
        time_cost_pillarbox = (datetime.datetime.now() - t0).total_seconds()
        print('pillarbox={}     timecost:{}s'.format(round(pillarbox, 3), time_cost_pillarbox))

        t0 = datetime.datetime.now()
        letterbox = iqa.letterbox()
        time_cost_letterbox = (datetime.datetime.now() - t0).total_seconds()
        print('letterbox={}     timecost:{}s'.format(round(letterbox, 3), time_cost_letterbox))

        t0 = datetime.datetime.now()
        if p_pic is not None:
            freezing = iqa.freezing(blackout, fps, p_pic)
        else:
            freezing = False
        time_cost_freezing = (datetime.datetime.now() - t0).total_seconds()
        print('freezing={}     timecost:{}s'.format(round(freezing, 3), time_cost_freezing))

        time_total = time_cost_block + time_cost_blur + time_cost_color + time_cost_contrast + time_cost_exposure + time_cost_noise + time_cost_blackout + time_cost_blockloss + time_cost_interlace + time_cost_pillarbox + time_cost_letterbox + time_cost_freezing

        # f = open(os.getcwd() + '/{}log.csv'.format(videofile), 'a')
        err = is_error(blackout, blockloss, interlace, freezing, blur, block, noise)
        f.write('%d,%d,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n' % (
            count + 1, blackout, blockloss, interlace, freezing, blur, block, noise, pillarbox, letterbox, exposure,
            contrast, color, time_total, err))
        f.close()
        f = open(os.getcwd() + '/{}log.csv'.format(videofile), 'a')
        print('{:<8}{:<10}{:<11}{:<11}{:<10}{:<12}{:<12}{:<7}{:<12}{}'.format(count + 1, blackout, round(blockloss, 3),
                                                                              round(interlace, 3), freezing,
                                                                              round(blur, 3), round(block, 3),
                                                                              round(noise, 3), round(time_total, 3),
                                                                              err))
        if err:
            err_f.write('%d,%d,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n' % (
                count + 1, blackout, blockloss, interlace, freezing, blur, block, noise, pillarbox, letterbox, exposure,
                contrast, color, time_total, err))
            err_f.close()
            err_f = open(os.getcwd() + '/{}errlog.csv'.format(videofile), 'a')
            if iserrout:
                cv2.imwrite(os.getcwd() + '/{}_err/{}_{}.{}'.format(videofile, count + 1, err, iserrout), frame)
        p_pic = iqa.gray
        exist_line += 1
        count += 1
    f.close()
    err_f.close()
    print('{} pictures scan finish!'.format(count))
    cap.release()
