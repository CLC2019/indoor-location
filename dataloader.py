import torch
import os
import numpy as np
from numpy import vstack
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class fp_pos_data(Dataset):
    def __init__(self, datapath):
        super(fp_pos_data, self).__init__()
        dataset = os.walk(datapath)
        self.TOA_output = []
        self.labeltag_output = []
        self.labels_pos_output = []
        self.phased_output = []
        first = 1
        for dirpath, dirnames, filenames in dataset:
            # print(filenames)
            for file in filenames:
                filename = os.path.join(dirpath, file)
                print(filename)
                output_data = []
                # read data from file
                df = pd.read_excel(filename)
                input_data = df.values
                if input_data.shape[0] < 2484:
                    toadd = df.columns.values
                    tomodify1 = toadd[3]
                    tomodify1 = int(tomodify1.split('.')[0])
                    tomodify2 = int(toadd[5].split('.')[0])
                    toadd[3] = tomodify1
                    toadd[5] = tomodify2
                    input_data = np.row_stack((toadd, input_data))
                    # print(input_data[0,:])
                if input_data.shape[0] == 3599:
                    toadd = df.columns.values
                    tomodify1 = toadd[3]
                    tomodify1 = int(tomodify1.split('.')[0])
                    #tomodify2 = int(toadd[5].split('.')[0])
                    toadd[3] = tomodify1
                    #toadd[5] = tomodify2
                    input_data = np.row_stack((toadd, input_data))

                # input_data = input_data[:, 4:]
                # append to array
                anchor = input_data[:, 1:3]
                anchor = anchor.astype(np.int8)
                # **********TOA数据***********
                TOA = input_data[:, 8]
                TOA = TOA.reshape(-1, 12)
                TOA = TOA.astype(np.float32)
                tags = TOA.shape[0]
                # **********tag标号***********
                labeltag = input_data[:, 3].reshape(-1, 12)[:, 0].astype(np.int32)
                # **********tag位置***********
                labels = input_data[:, 4:6]
                labels = labels.reshape(-1, 12, 2)
                labels = labels[:, 0, :]
                labels = labels.astype(np.float32)
                # **********天线phase***********
                phase1 = input_data[:, 11]
                phase2 = input_data[:, 12]
                phased = phase1 - phase2
                phased = phased.reshape(-1, 12)
                phase1 = phase1.reshape(-1, 12)
                phase2 = phase2.reshape(-1, 12)
                phased = np.expand_dims(phased.astype(np.float32), axis=1)
                phase2 = np.expand_dims(phase2.astype(np.float32), axis=1)
                phase1 = np.expand_dims(phase1.astype(np.float32), axis=1)
                phaseall = np.concatenate((phase1, phase2, phased), axis=1)



                if first > 0:
                    first = 0
                    self.TOA_output = TOA
                    self.labeltag_output = labeltag
                    self.labels_pos_output = labels
                    self.phased_output = phaseall
                else:
                    self.TOA_output = vstack((self.TOA_output, TOA))
                    self.labeltag_output = np.concatenate((self.labeltag_output, labeltag))
                    self.labels_pos_output = vstack((self.labels_pos_output, labels))
                    self.phased_output = vstack((self.phased_output, phaseall))
                    #print(TOA_output.shape)
                    #print(labeltag_output.shape)
                    #print(labels_pos_output.shape)
        # min-max归一化 ******TOA*******
        TOA_max = np.max(self.TOA_output)
        print(TOA_max)
        TOA_min = np.min(self.TOA_output)
        print(TOA_min)
        self.TOA_output = self.TOA_output.reshape(-1)
        print(self.TOA_output.shape)

        for i in range(self.TOA_output.shape[0]):
            self.TOA_output[i] = (self.TOA_output[i] - TOA_min) / (TOA_max - TOA_min)
        self.TOA_output = self.TOA_output.reshape(-1, 12)
        # print(TOA_output)

        # min-max归一化 ******phased*******
        phase_max = np.max(self.phased_output)
        phase_min = np.min(self.phased_output)
        #self.phased_output = self.phased_output.reshape(-1)
        print(self.phased_output.shape)
        for i in range(self.phased_output.shape[0]):
            for j in range(self.phased_output.shape[1]):
                for k in range(self.phased_output.shape[2]):
                    self.phased_output[i, j, k] = (self.phased_output[i, j, k] - phase_min) / (phase_max - phase_min)
        #self.phased_output = self.phased_output.reshape(-1, 12)

        #labeltag 减去1
        for i in range(self.labeltag_output.shape[0]):
            self.labeltag_output[i] = self.labeltag_output[i] -1
        # **********转tensor***********
        self.TOA_output = torch.tensor(self.TOA_output)
        self.labeltag_output = torch.tensor(self.labeltag_output).long()
        self.labels_pos_output = torch.tensor(self.labels_pos_output)
        self.phased_output = torch.tensor(self.phased_output)
        self.phased_output = self.phased_output.reshape(-1, 36)

    def __len__(self):
        return len(self.TOA_output)

    def __getitem__(self, item):
        return self.TOA_output[item], self.labeltag_output[item], self.labels_pos_output[item], self.phased_output[item]

class fp_pos_data_1m(Dataset):
    def __init__(self, datapath):
        super(fp_pos_data_1m, self).__init__()
        dataset = os.walk(datapath)
        self.TOA_output = []
        self.labeltag_output = []
        self.labels_pos_output = []
        self.phased_output = []
        first = 1
        for dirpath, dirnames, filenames in dataset:
            # print(filenames)
            for file in filenames:
                filename = os.path.join(dirpath, file)
                print(filename)
                output_data = []
                # read data from file
                df = pd.read_csv(filename)
                input_data = df.values
                if input_data.shape[0] < 2484:
                    toadd = df.columns.values
                    tomodify1 = toadd[3]
                    tomodify1 = int(tomodify1.split('.')[0])
                    tomodify2 = int(toadd[5].split('.')[0])
                    toadd[3] = tomodify1
                    toadd[5] = tomodify2
                    input_data = np.row_stack((toadd, input_data))
                    # print(input_data[0,:])
                if input_data.shape[0] == 3599:
                    toadd = df.columns.values
                    tomodify1 = toadd[3]
                    tomodify1 = int(tomodify1.split('.')[0])
                    #tomodify2 = int(toadd[5].split('.')[0])
                    toadd[3] = tomodify1
                    #toadd[5] = tomodify2
                    input_data = np.row_stack((toadd, input_data))

                if input_data.shape[0] == 69971:
                    toadd = df.columns.values
                    tomodify1 = toadd[3]
                    tomodify1 = int(tomodify1.split('.')[0])
                    tomodify2 = int(toadd[4].split('.')[0])
                    tomodify3 = int(toadd[5].split('.')[0])
                    toadd[3] = tomodify1
                    toadd[4] = tomodify2
                    toadd[5] = tomodify3
                    toadd = toadd.astype(np.float32)
                    input_data = np.row_stack((toadd, input_data))

                # input_data = input_data[:, 4:]
                # append to array
                anchor = input_data[:, 1:3]
                anchor = anchor.astype(np.int8)
                # **********TOA数据***********
                TOA = input_data[:, 8]
                TOA = TOA.reshape(-1, 12)
                TOA = TOA.astype(np.float32)
                tags = TOA.shape[0]
                # **********tag标号***********
                labeltag = input_data[:, 3].reshape(-1, 12)[:, 0].astype(np.int32)
                # **********tag位置***********
                labels = input_data[:, 4:6]
                labels = labels.reshape(-1, 12, 2)
                labels = labels[:, 0, :]
                labels = labels.astype(np.float32)
                # **********天线phase***********
                phase1 = input_data[:, 11]
                phase2 = input_data[:, 12]
                phased = phase1 - phase2
                phased = phased.reshape(-1, 12)
                phase1 = phase1.reshape(-1, 12)
                phase2 = phase2.reshape(-1, 12)
                phased = np.expand_dims(phased.astype(np.float32), axis=1)
                phase2 = np.expand_dims(phase2.astype(np.float32), axis=1)
                phase1 = np.expand_dims(phase1.astype(np.float32), axis=1)
                phaseall = np.concatenate((phase1, phase2, phased), axis=1)



                if first > 0:
                    first = 0
                    self.TOA_output = TOA
                    self.labeltag_output = labeltag
                    self.labels_pos_output = labels
                    self.phased_output = phaseall
                else:
                    self.TOA_output = vstack((self.TOA_output, TOA))
                    self.labeltag_output = np.concatenate((self.labeltag_output, labeltag))
                    self.labels_pos_output = vstack((self.labels_pos_output, labels))
                    self.phased_output = vstack((self.phased_output, phaseall))
                    #print(TOA_output.shape)
                    #print(labeltag_output.shape)
                    #print(labels_pos_output.shape)
        # min-max归一化 ******TOA*******
        TOA_max = np.max(self.TOA_output)
        print(TOA_max)
        TOA_min = np.min(self.TOA_output)
        print(TOA_min)
        self.TOA_output = self.TOA_output.reshape(-1)
        print(self.TOA_output.shape)

        for i in range(self.TOA_output.shape[0]):
            self.TOA_output[i] = (self.TOA_output[i] - TOA_min) / (TOA_max - TOA_min)
        self.TOA_output = self.TOA_output.reshape(-1, 12)
        # print(TOA_output)

        # min-max归一化 ******phased*******
        phase_max = np.max(self.phased_output)
        phase_min = np.min(self.phased_output)
        #self.phased_output = self.phased_output.reshape(-1)
        print(self.phased_output.shape)
        for i in range(self.phased_output.shape[0]):
            for j in range(self.phased_output.shape[1]):
                for k in range(self.phased_output.shape[2]):
                    self.phased_output[i, j, k] = (self.phased_output[i, j, k] - phase_min) / (phase_max - phase_min)
        #self.phased_output = self.phased_output.reshape(-1, 12)

        #labeltag 减去1
        for i in range(self.labeltag_output.shape[0]):
            self.labeltag_output[i] = self.labeltag_output[i] -1


        #labelpos归一化
        self.labels_pos_output[:, 0] = self.labels_pos_output[:, 0] / 120
        self.labels_pos_output[:, 1] = self.labels_pos_output[:, 1] / 50
        # **********转tensor***********
        self.TOA_output = torch.tensor(self.TOA_output)
        self.labeltag_output = torch.tensor(self.labeltag_output).long()
        self.labels_pos_output = torch.tensor(self.labels_pos_output)
        self.phased_output = torch.tensor(self.phased_output)
        self.phased_output = self.phased_output.reshape(-1, 36)

    def __len__(self):
        return len(self.TOA_output)

    def __getitem__(self, item):
        return self.TOA_output[item], self.labeltag_output[item], self.labels_pos_output[item], self.phased_output[item]


def get_loader(config):
    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)


    train_dir = './data/train/train_1m'
    #test_dir = './dataset/test_dataset'
    train_dataset = fp_pos_data_1m(train_dir)
    #test_dataset = fp_pos_data(test_dir)
    print('#trainDataset', len(train_dataset))
    #print('#testDataset', len(test_dataset))
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               num_workers=0,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=8,
                                              shuffle=True)
    # test_loader = None
    positiontestdir = './dataset/test_dataset'
    positiontestdataset = fp_pos_data(positiontestdir)

    positiontest_loader = torch.utils.data.DataLoader(dataset=positiontestdataset,
                                              batch_size=8,
                                              shuffle=True)
    return train_loader, test_loader, positiontest_loader

if __name__ == '__main__':
    class config:
        batch_size = 32
    get_loader(config=config)
    print("hello")