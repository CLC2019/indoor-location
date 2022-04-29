import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import CNNmodel, DNNmodel, fp_posmodel
from utils import logger_configuration, plt_cdf
import torch
import torch.nn as nn
from config import fp_pos_config as config
from dataloader import get_loader
import torch.nn.functional as F


device = config.device

def train(model, train_loader, test_loader, postestloader, config, criterion, optimizer, scheduler):
    logger = logger_configuration(config, save_log=True)

    data_len = len(train_loader.dataset)
    iteration = 0
    for epoch in range(config.epochs):
        model.train()
        for batch_idx, _data in enumerate(train_loader):
            TOA, labeltag, labelpos, phase = _data
            optimizer.zero_grad()
            ##summary(model, spectrograms)   #显示参数
            output = model(phase)  #
            loss = criterion(output, labeltag)
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            iteration += 1

            if iteration % config.print_step == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(TOA), data_len,
                           100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        if (epoch+1) % config.test_step == 0:
            test(model, test_loader, criterion, logger)
        if (epoch+1) % config.save_step == 0:
            torch.save(model.state_dict(), config.models + '/Ep{}.pth'.format(epoch+1))
            #test(model, test_loader, criterion, logger)
        '''
        if (epoch+1) % config.test_step == 0:
            positiontest(model, postestloader, criterion, logger=None)
        '''


def positiontest(model, test_loader, criterion, logger=None):
    if logger is None:
        logger = logger_configuration(config, save_log=False)
    print('\nevaluating…')
    model.eval()

    test_loss = 0
    pretruesum = 0
    prealllabel = 0
    meter_error = []
    pos = []
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            TOA, labeltag, labelpos, phase = _data
            output = model(phase)  #
            #loss = criterion(output, labeltag)
            prelabel = torch.argmax(output, dim=1)
            for i in range(len(prelabel)):
                referL = labelpos[i, :]
                posx = (prelabel[i] % 23 + 1) * 5
                posy = (prelabel[i] // 23 + 1) * 5
                pos.append([posx, posy])
                preLerror = [posx - referL[0],  posy - referL[1]]
                meter_error.append(math.sqrt(preLerror[0]*preLerror[0] + preLerror[1]*preLerror[1]))
    plt_cdf(meter_error)
    meter_error = torch.tensor(meter_error)

    print(meter_error)
    mean_meter_error = torch.mean(meter_error)
    logger.info('Test set: avg_meter_error: {:.4f}\n'.format(mean_meter_error))

def test(model, test_loader, criterion, logger=None):
    if logger is None:
        logger = logger_configuration(config, save_log=False)
    print('\nevaluating…')
    model.eval()

    test_loss = 0
    pretruesum = 0
    prealllabel = 0
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            TOA, labeltag, labelpos, phase = _data
            output = model(phase)  #
            loss = criterion(output, labeltag)
            argmax = torch.argmax(output, dim=1)
            for i in range(len(argmax)):
                m = labeltag[i] - argmax[i]
                prealllabel += 1
                if m==0:
                    pretruesum += 1
            test_loss += loss.item() / len(test_loader)

    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    logger.info('Test set: Average loss: {:.4f}, acc: {:.4f}%\n'.format(test_loss, 100 * pretruesum/prealllabel))

if __name__ == "__main__":
    train_loader, test_loader, postest_loader = get_loader(config)
    model = fp_posmodel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    #model_path = './history/phase/models/Ep35.pth'
    #pre_dict = torch.load(model_path)
    #model.load_state_dict(pre_dict, strict=False)
    train(model, train_loader, test_loader, postest_loader, config, criterion, optimizer, scheduler)
    test(model, test_loader, criterion, logger=None)
    #positiontest(model, postest_loader, criterion, logger=None)