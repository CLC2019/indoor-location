import os
import pandas as pd
import numpy as np
from numpy import vstack

def datapre(datapath):
    datapath = datapath
    dataset = os.walk(datapath)
    TOA_output = []
    labeltag_output = []
    labels_pos_output = []
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
                #print(input_data[0,:])

            #input_data = input_data[:, 4:]
            # append to array
            anchor = input_data[:, 1:3]
            anchor = anchor.astype(np.int8)
            TOA = input_data[:, 8]
            TOA = TOA.reshape(-1, 12)
            TOA = TOA.astype(np.float32)
            tags = TOA.shape[0]
            labeltag = input_data[:, 3].reshape(-1, 12)[:, 0].astype(np.int32)
            labels = input_data[:, 4:6]
            labels = labels.reshape(-1, 12, 2)
            labels = labels[:, 0, :]
            labels = labels.astype(np.float32)

            if first > 0:
                first = 0
                TOA_output = TOA
                labeltag_output = labeltag
                labels_pos_output = labels
            else:
                TOA_output = vstack((TOA_output, TOA))
                labeltag_output = np.concatenate((labeltag_output, labeltag))
                labels_pos_output = vstack((labels_pos_output, labels))
                print(TOA_output.shape)
                print(labeltag_output.shape)
                print(labels_pos_output.shape)
    #min-max归一化 ******TOA*******
    TOA_max = np.max(TOA_output)
    print(TOA_max)
    TOA_min = np.min(TOA_output)
    print(TOA_min)
    TOA_output = TOA_output.reshape(-1)
    print(TOA_output.shape)
    for i in range(TOA_output.shape[0]):
        TOA_output[i] = (TOA_output[i] - TOA_min) / (TOA_max - TOA_min)
    TOA_output = TOA_output.reshape(-1, 12)
    #print(TOA_output)

    #**********转tensor***********
    TOA_output = TOA_output
    labeltag_output = labeltag_output
    labels_pos_output = labels_pos_output
    #print(TOA_output)
    return TOA_output, labeltag_output, labels_pos_output

if __name__ == '__main__':
    traindir = './dataset/train_dataset'
    testdir = './dataset/test_dataset'
    datapre(traindir)
    print("hello")