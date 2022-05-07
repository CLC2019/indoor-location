import torch.nn as nn
import torch
import torch.nn.functional as F

class CNNmodel(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel=3, stride=1, dropout=0):
        super(CNNmodel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        #self.linear1 = nn.Linear()


    def forward(self, x):
        #x为输入的CIR
        x = x.unsqueeze(1)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)

        return x

class DNNmodel(nn.Module):
    def __init__(self, tags=207, TRPs=12):
        super(DNNmodel, self).__init__()
        self.linear1 = nn.Linear(TRPs, 80)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(80, 60)
        self.linear3 = nn.Linear(60, 30)
        self.linear4 = nn.Linear(30, tags)

    def forward(self, x):
        #x [B, TRPs]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = F.softmax(x, dim=1)
        return x

class rescnn(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel=3, stride=1):
        super(rescnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel, stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += res
        x = self.relu(x)
        return x

class fp_posmodel(nn.Module):
    def __init__(self, n_cnnlayers=12):
        super(fp_posmodel, self).__init__()
        self.cnn = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.rescnn_layers = nn.Sequential(*[
            rescnn(32, 32, kernel=3, stride=1)
            for _ in range(n_cnnlayers)
        ])
        self.linear1 = nn.Linear(32*12, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 207)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bn1(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.reshape(sizes[0], -1)
        x = self.linear1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x

class fp_pos_phasemodel(nn.Module):
    def __init__(self, n_cnnlayers=12):
        super(fp_pos_phasemodel, self).__init__()
        self.cnn = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.rescnn_layers = nn.Sequential(*[
            rescnn(32, 32, kernel=3, stride=1)
            for _ in range(n_cnnlayers)
        ])
        self.linear1 = nn.Linear(32 * 36, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 207)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.bn1(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.reshape(sizes[0], -1)
        x = self.linear1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x
