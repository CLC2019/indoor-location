import logging
import os
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

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

    plt.plot(x, y)
    plt.grid()
    plt.savefig("1.jpg")