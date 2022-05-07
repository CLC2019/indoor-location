import logging
import os
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("DeepSC")
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s %(filename)s: %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def GreedyDecoder(output, labels, collapse_repeated=True):
    #decode [B, 3]
    #target [B, 3]
    arg_maxes = torch.argmax(output, dim=1)
    decodes = []
    targets = []
    int_to_pos = 1 # 待定 a function
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(int_to_pos)
        for j, index in enumerate(args):
            decode.append(index.item())
        decodes.append(int_to_pos)
    return decodes, targets

def plt_cdf(data):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(0, max(data))
    y = ecdf(x)
    #设置标题，x，y标题
    plt.title("CDF")
    plt.xlabel("error m")
    plt.ylabel("probability")
    #设置坐标刻度
    yscale = np.arange(0, 1.1, 0.1)
    plt.yticks(yscale)
    xscale = np.arange(0, max(data), 5)
    plt.xticks(xscale)

    plt.plot(x, y)
    plt.grid()
    plt.savefig("1.jpg")

def out2pos(output, labelpos, pos, meter_error):
    values, prelabel = output.topk(3, dim=1)
    #prelabel = torch.argmax(output, dim=1)
    for i in range(prelabel.shape[0]):
        referL = labelpos[i, :]
        posx1 = (prelabel[i, 0] % 23 + 1) * 5
        posy1 = (prelabel[i, 0] // 23 + 1) * 5
        posx2 = (prelabel[i, 1] % 23 + 1) * 5
        posy2 = (prelabel[i, 1] // 23 + 1) * 5
        posx3 = (prelabel[i, 2] % 23 + 1) * 5
        posy3 = (prelabel[i, 2] // 23 + 1) * 5
        #posx = values[i, 0] * posx1 + values[i, 1] * posx2
        #posy = values[i, 0] * posy1 + values[i, 1] * posy2
        posx = posx1
        posy = posy1
        pos.append([posx, posy])
        preLerror = [posx - referL[0], posy - referL[1]]
        meter_error.append(math.sqrt(preLerror[0] * preLerror[0] + preLerror[1] * preLerror[1]))

    return meter_error, pos

def WIPout2pos(output1, output2, labelpos, pos, meter_error):
    values1, prelabel1 = output1.topk(1, dim=1)
    values2, prelabel2 = output2.topk(1, dim=1)
    #prelabel = torch.argmax(output, dim=1)
    for i in range(prelabel1.shape[0]):
        referL = labelpos[i, :]
        posx1 = (prelabel1[i, 0] % 23 + 1) * 5
        posy1 = (prelabel1[i, 0] // 23 + 1) * 5
        posx2 = (prelabel2[i, 0] % 23 + 1) * 5
        posy2 = (prelabel2[i, 0] // 23 + 1) * 5
        posx = (0.99/(0.99+0.79)) * posx1 + (0.79/(0.99+0.79)) * posx2
        posy = (0.99/(0.99+0.79)) * posy1 + (0.79/(0.99+0.79)) * posy2
        pos.append([posx, posy])
        preLerror = [posx - referL[0], posy - referL[1]]
        meter_error.append(math.sqrt(preLerror[0] * preLerror[0] + preLerror[1] * preLerror[1]))

    return meter_error, pos
