import argparse

import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from math import pi
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from process import load_all_files

import copy
import modeloperations as mo
import time
import os

# rescale the x-axis for data
scale = 50


# load printer data
class PtData(Dataset):
    def __init__(self, dim=1):
        data_dir = r"Data"
        res_dict = load_all_files(rdir=data_dir)
        self.data = []
        for printer in res_dict.keys():
            x = np.array([])
            y = np.array([])
            x_test = np.array([])
            y_test = np.array([])
            for speed in res_dict[printer].keys():
                addx = res_dict[printer][speed]['data'][:, 0]
                addy = np.sqrt(
                    res_dict[printer][speed]['data'][:, 1] ** 2 + res_dict[printer][speed]['data'][:, 2] ** 2)
                # train-test split
                if not speed == 30:
                    x = np.concatenate((x, addx))
                    y = np.concatenate((y, addy))
                else:
                    x_test = np.concatenate((x_test, addx))
                    y_test = np.concatenate((y_test, addy))
            self.data.append((torch.tensor(y).unsqueeze(1).to(torch.float32),
                              torch.tensor(x / scale).unsqueeze(1).to(torch.float32)))
            self.data.append((torch.tensor(y_test).unsqueeze(1).to(torch.float32),
                              torch.tensor(x_test / scale).unsqueeze(1).to(torch.float32)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Neural Network encoder function
class Encoder(nn.Module):
    def __init__(self, x_dim=1, h_dim=64, feature_dim=1):
        super(Encoder, self).__init__()
        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, feature_dim)]
        self.input_to_hidden = nn.Sequential(*layers)
        self.sc = nn.Linear(x_dim, feature_dim)

    def forward(self, x):
        return self.sc(x) + self.input_to_hidden(x)


# Sigmoid encoder function
class Encoder_sigmoid(nn.Module):
    def __init__(self, x_dim=1, feature_dim=1):
        super(Encoder_sigmoid, self).__init__()
        '''
        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, feature_dim)]
        '''
        self.sc = nn.Linear(x_dim, feature_dim)  # nn.Sequential(*layers)
        # self.sc = nn.Linear(x_dim, feature_dim)

    def forward(self, x):
        return self.sc(x)  # + self.input_to_hidden(x)


# Gaussian kernel encoder function
class Encoder_kernel(nn.Module):
    def __init__(self, x_dim=1, feature_dim=1):
        super(Encoder_kernel, self).__init__()
        self.sc = nn.Linear(x_dim, feature_dim)  # nn.Sequential(*layers)

    def forward(self, x):
        return self.sc(x)


# Neural network decoder function
class Decoder(nn.Module):
    def __init__(self, feature_dim=1, middle_dim=128, z_dim=1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(feature_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, middle_dim)
        self.fc3 = nn.Linear(middle_dim, middle_dim)
        self.fc4 = nn.Linear(middle_dim, middle_dim)
        self.fc5 = nn.Linear(middle_dim, z_dim)
        # self.l = nn.Parameter(torch.zeros((1)) - 2)

    def forward(self, r):
        sy = torch.relu(self.fc1(r))
        sy = torch.relu(self.fc2(sy))
        sy = torch.relu(self.fc3(sy))
        sy = torch.relu(self.fc4(sy))
        sy = self.fc5(sy)
        return sy


# Sigmoid decoder function
class Decoder_sigmoid(nn.Module):
    def __init__(self, feature_dim=1, middle_dim=128, z_dim=1):
        super(Decoder_sigmoid, self).__init__()
        # self.fc1 = nn.Linear(feature_dim, z_dim)

        self.fc2 = nn.Linear(feature_dim, z_dim)

    def forward(self, r):
        sy = torch.sigmoid(r)  # self.fc1(r))
        sy = self.fc2(sy)
        return sy


# Gaussian kernel decoder function
class Decoder_kernel(nn.Module):
    def __init__(self, h_dim=128, z_dim=1, device='cuda:0'):
        super(Decoder_kernel, self).__init__()
        # self.fc1 = nn.Linear(feature_dim, z_dim)
        self.h_dim = h_dim
        self.centers = nn.Parameter(torch.rand(h_dim, requires_grad=True))
        self.stds = nn.Parameter(torch.ones(h_dim, requires_grad=True))  # .to(device)


        self.fc2 = nn.Linear(h_dim, z_dim)

        middle_dim = 1024
        self.fc2n = nn.Linear(h_dim, middle_dim)
        self.fcn2 = nn.Linear(middle_dim, z_dim)

    def forward(self, x):
        n = len(x)
        mumat = torch.tile(self.centers, (n, 1))
        lmat = torch.tile(self.stds, (n, 1))

        xmat = torch.transpose(torch.stack([x for i in range(self.h_dim)])[:, :, 0], 0,
                               1)  # torch.transpose(torch.tile(x, (self.h_dim)), 0, 1)

        Phi = mumat - xmat
        Phi = torch.exp(-Phi ** 2 * (0.5 * lmat ** 2))  # * self.magnify

        sy = self.fcn2(self.fc2n(Phi))
        return sy


def train(dataset, encoder, encoders, decoder, decoders, epochs, n_task, feature_dim, device, args={}):
    steps = 0
    inner_epoch = 10
    epoch_loss_history = []
    learning_rate = 3e-3
    wd = 3e-4
    centers = torch.zeros((n_task, feature_dim)).to(device)
    # centeredlf = copy.deepcopy(lf)
    factor2 = 0.01
    for epoch in range(epochs):
        factor2 *= 1.03
        start_time = time.time()
        clientdecoder = [copy.deepcopy(decoder) for i in range(len(encoders))]
        clientencoder = [copy.deepcopy(encoder) for i in range(len(encoders))]
        epoch_loss = 0.
        epoch_clloss = 0.
        if epoch % 5 == 4:
            if args['space'] in {'sigmoid', 'kernel'}:
                learning_rate *= 0.99
            else:
                learning_rate *= 0.9
        for i in range(len(dataset) // 2):
            data_i = dataset[2 * i]
            data = data_i[0].to(device), data_i[1].to(device)
            if args['fed'] in {'Ditto', 'indiv'}:
                optimizer = torch.optim.Adam(
                    list(clientencoder[i].parameters()) + list(
                        clientdecoder[i].parameters()), lr=learning_rate, weight_decay=wd)
                optimizer_l = torch.optim.Adam(list(decoders[i].parameters()) + list(encoders[i].parameters()),
                                               lr=learning_rate, weight_decay=wd)

            else:
                optimizer = torch.optim.Adam(
                    list(encoders[i].parameters()) + list(
                        clientdecoder[i].parameters()), lr=learning_rate, weight_decay=wd)
                optimizer_l = torch.optim.Adam(decoders[i].parameters(), lr=learning_rate, weight_decay=wd)
            for j in range(inner_epoch):
                optimizer.zero_grad()
                # Create context and target points and apply neural process
                x, y = data
                if args['fed'] in {'Ditto', 'indiv'}:
                    feature_x = clientencoder[i](x)
                else:
                    feature_x = encoders[i](x)

                ci = feature_x.mean(dim=0)
                y_pred = clientdecoder[i](feature_x)

                # regularize the features is essential for the performance of nn
                clloss = nn.MSELoss()(centers.mean(dim=0), ci)
                if args['space'] in {'nn'}:
                    factor = 0.01
                else:
                    factor = 0.0

                # empirical loss
                loss = nn.MSELoss()(y_pred, y) + factor * clloss  # + 0.1*nn.MSELoss()(sudo_y, sudo_y_global)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(x)
                epoch_clloss += clloss.item()

                steps += 1
                closs = torch.zeros(1)

                optimizer_l.zero_grad()
                # Create context and target points and apply neural process
                x, y = data
                if args['fed'] in {'Ditto', 'indiv'}:
                    feature_x = encoders[i](x)
                else:
                    with torch.no_grad():
                        feature_x = encoders[i](x)
                y_pred = decoders[i](feature_x)
                loss = nn.MSELoss()(y_pred, y)
                loss.backward()
                optimizer_l.step()
                if not args['fed'] == 'indiv':
                    if args['fed'] == 'SDA':
                        beta = 1
                    else:
                        beta = 0.9  # 0.001
                    coeffs = [beta, 1 - beta]
                    decoders[i].load_state_dict(mo.scalar_mul(coeffs, [decoder, decoders[i]]).state_dict())
                    if args['fed'] == 'Ditto':
                        clientencoder[i].load_state_dict(
                            mo.scalar_mul(coeffs, [encoder, clientencoder[i]]).state_dict())

        decoder.load_state_dict(mo.scalar_mul(torch.ones(len(encoders)) / len(encoders), clientdecoder).state_dict())
        if args['fed'] == 'Ditto':
            encoder.load_state_dict(
                mo.scalar_mul(torch.ones(len(encoders)) / len(encoders), clientencoder).state_dict())

        end_time = time.time()
        print("Epoch: {}, Avg_loss: {}, clloss: {}, dis loss: {}, time {}".format(epoch, (
            epoch_loss) / len(dataset), epoch_clloss / len(dataset), closs.item(),
                                                                                  end_time - start_time))
        epoch_loss_history.append(epoch_loss / len(dataset))
        trainl = test_loss(dataset, encoder, encoders, decoder, decoders, n_task, device, args, test=0)

        tl = test_loss(dataset, encoder, encoders, decoder, decoders, n_task, device, args)

    return tl


def test_loss(dataset, encoder, encoders, decoder, decoders, n_task, device, args, test=1):
    epoch_loss = 0
    epoch_length = 0
    for i in range(len(dataset) // 2):
        data_i = dataset[2 * i + test]
        data = data_i[0].to(device), data_i[1].to(device)
        # print('data size')
        # print(len(data),data[0].size())
        with torch.no_grad():
            x, y = data
            # print(x.size())
            # assert False
            feature_x = encoders[i](x)
            y_pred = decoders[i](feature_x)
            loss = nn.MSELoss()(y_pred, y)

            epoch_loss += loss.item() * len(x)
            epoch_length += len(x)
    if test == 1:
        print(" Avg test loss: {}, ".format(epoch_loss / epoch_length))
    else:
        print(" Avg train loss: {}, ".format(epoch_loss / epoch_length))
    return epoch_loss / epoch_length


def one_experinemt(output_dir, args):
    n_task = 6
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = PtData()

    feature_dim = 1
    device = torch.device('cuda')

    if args['space'] == 'sigmoid':
        feature_dim_sigmoid = 1
        encoder = Encoder_sigmoid(feature_dim=feature_dim_sigmoid).to(device)
        encoders = [Encoder_sigmoid(feature_dim=feature_dim_sigmoid).to(device) for i in range(n_task)]
        decoder = Decoder_sigmoid(feature_dim=feature_dim_sigmoid).to(device)
        decoders = [Decoder_sigmoid(feature_dim=feature_dim_sigmoid).to(device) for i in range(n_task)]

    elif args['space'] == 'nn':
        feature_dim_nn = 128
        encoder = Encoder(feature_dim=feature_dim_nn).to(device)
        encoders = [Encoder(feature_dim=feature_dim_nn).to(device) for i in range(n_task)]
        decoder = Decoder(feature_dim=feature_dim_nn).to(device)
        decoders = [Decoder(feature_dim=feature_dim_nn).to(device) for i in range(n_task)]
    elif args['space'] == 'kernel':
        h_dim = 16
        encoder = Encoder_kernel().to(device)
        encoders = [Encoder_kernel().to(device) for i in range(n_task)]
        decoder = Decoder_kernel(h_dim=h_dim).to(device)
        decoders = [Decoder_kernel(h_dim=h_dim).to(device) for i in range(n_task)]

    if args['space'] in {'sigmoid'}:
        epochs = 200
    elif args['space'] in {'kernel'}:
        epochs = 200
    else:
        epochs = 20

    tl = train(dataset, encoder, encoders, decoder, decoders, epochs, n_task, feature_dim, device, args)
    # tl = 0
    x_target = torch.Tensor(np.linspace(0.3, 2.2, 50)).to(device)
    x_target = x_target.unsqueeze(1).unsqueeze(0)
    if args['space'] == 'kernel':
        x_target = x_target[0]

    fig, axs = plt.subplots(1, 1)
    idxlist = [0, 1, 2, 3, 4, 5]

    for index in range(n_task):
        with torch.no_grad():
            y_target = (decoders[index](encoders[index](x_target)))
        mu = y_target.detach()

        if args['space'] == 'kernel':
            axs.plot(x_target.cpu().numpy()[:, 0], mu.cpu().numpy()[:, 0] * scale,
                     alpha=0.8, label="printer {}".format(index + 1))
        else:
            axs.plot(x_target.cpu().numpy()[0, :, 0], mu.cpu().numpy()[0, :, 0] * scale,
                     alpha=0.8, label="printer {}".format(index + 1))

        x_context, y_context = dataset[2 * index]

        axs.scatter(x_context.cpu().numpy(), y_context.cpu().numpy() * scale, alpha=0.3, marker='+')

    # plt.show()
    plt.ylim((0.0 * scale, 1.0 * scale))
    plt.xlabel('Mean Square Acceleration', fontsize=20)
    plt.ylabel('Average Nominal Speed', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(output_dir, 'printerdata_{}_{}.png'.format(args['fed'], args['space'])))
    return tl


if __name__ == "__main__":
    from misc import Tee
    import time
    import sys
    import os

    output_dir = 'printer_outputs/'
    jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
    output_dir += jour
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))

    n_task = 6

    parser = argparse.ArgumentParser(description='Printer data experiment')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fed', type=str, default="PDA")
    parser.add_argument('--space', type=str, default="nn")

    args = vars(parser.parse_args())
    print(args)
    tstloss = one_experinemt(output_dir, args)
    print("{}, {}, test loss {}".format(args['fed'], args['space'], tstloss))

