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

import argparse

import copy
import modeloperations as mo
import time

fed_dict = {
    'Ditto': 'Ditto',
    'PDA': 'PFL-DA',
    'indiv': 'Indiv',
}
color_list = ["orange", "green", "blue", "pink", "grey", "red", "cyan"]

MD = {
        'PFL-DA':"o",
        'Ditto': "v",
        'TP': "s",
        'FedAverage': "*",
        'Simple DA':"p",
        'Pfedme':"1",
        'indiv':"+",
}

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. 
    """

    def __init__(self, amplitude_range=(0.8, 1.2), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100, sparse_middle=False, rci=10):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range

        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            # x = (torch.rand(num_points,1))
            frequency = 1  # + np.random.rand()
            if sparse_middle:
                for j in range(2):
                    if j == 0:
                        if i > rci:
                            x = torch.linspace(-0.2, 1.2, 100).unsqueeze(1)
                        else:
                            x = torch.linspace(0, 1, num_points).unsqueeze(1)
                    else:
                        x = torch.linspace(0, 1, 100).unsqueeze(1)
                    y = a * torch.sin(2 * pi * frequency * (x - b))  # *500
                    self.data.append((x, y))
            else:
                for j in range(2):
                    if j == 0:
                        x = torch.rand(num_points, 1)
                    else:
                        x = torch.linspace(0, 1, 100).unsqueeze(1)

                    y = a * torch.sin(2 * pi * frequency * (x - b))  # *500
                    self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


scale = 1

class Embeddingnet1(nn.Module):
    def __init__(self, x_dim=1, embed_dim=128, h_dim=128):
        super(Embeddingnet1, self).__init__()
        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  # nn.Linear(h_dim,h_dim),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  # nn.Linear(h_dim,h_dim),
                  # nn.ReLU(inplace=True),
                  nn.Linear(h_dim, embed_dim)]
        self.input_to_hidden = nn.Sequential(*layers)
        self.sc = nn.Linear(x_dim, embed_dim)

    def forward(self, x):
        return self.sc(x) + self.input_to_hidden(x)


class Encoder(nn.Module):
    def __init__(self, x_dim=1, h_dim=256, feature_dim=1):
        super(Encoder, self).__init__()

        self.embednet1 = Embeddingnet1(x_dim, embed_dim=feature_dim, h_dim=h_dim)
        # self.embednet12 = Embeddingnet1(x_dim, embed_dim=feature_dim, h_dim=h_dim)

    def forward(self, x):
        bd = len(x)
        uns = torch.ones(bd, 1).to(x.device)
        # uns = uns*0.5+x*0.5
        phi = x + self.embednet1(uns)
        return phi % 1 - 0.5  # *self.embednet12(uns)


class DecoderFunction(nn.Module):
    def __init__(self, feature_dim=1, middle_dim=128, z_dim=1):
        super(DecoderFunction, self).__init__()
        self.fc1 = nn.Linear(feature_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, middle_dim)
        self.fc3 = nn.Linear(middle_dim, middle_dim)
        self.fc4 = nn.Linear(middle_dim, middle_dim)

        self.fc5 = nn.Linear(middle_dim, z_dim)


    def forward(self, r):
        sy = torch.relu(self.fc1(r))
        sy = torch.relu(self.fc2(sy)) + sy
        sy = torch.relu(self.fc3(sy)) + sy
        sy = torch.relu(self.fc4(sy)) + sy
        sy = self.fc5(sy)
        return sy


def train(dataset, encoder, encoders, decoder, decoders, epochs, n_task, feature_dim, device, args={}):
    steps = 0
    inner_epoch = 150

    threshold_inner_epochs = 50
    second_threshold = 100
    trainl = 1000000

    epoch_loss_history = []
     
    learning_rate = 5e-3
    wd = 3e-7
    centers = torch.zeros((n_task, feature_dim)).to(device)

    real_epoch = epochs
    if args['fed'] in {'PDA'}:
        used_inner_epoch = inner_epoch
    else:
        used_inner_epoch = threshold_inner_epochs
    for epoch in range(real_epoch):
        start_time = time.time()
        clientdecoder = [copy.deepcopy(decoder) for i in range(len(encoders))]
        clientencoder = [copy.deepcopy(encoder) for i in range(len(encoders))]
        epoch_loss = 0.
        epoch_clloss = 0.
        # adjust learning rate
        if epoch % 5 == 4:
            learning_rate *= 0.9
        
        # local updates
        for i in range(len(dataset) // 2):
            data_i = dataset[2 * i]
            data = data_i[0].to(device), data_i[1].to(device)
            # print('data size')
            # print(len(data),data[0].size())
            if args['fed'] in {'Ditto', 'indiv'}:
                optimizer = torch.optim.SGD(
                    list(clientencoder[i].parameters()) + list(
                        clientdecoder[i].parameters()), lr=learning_rate, weight_decay=wd)
                optimizer_l = torch.optim.Adam(list(decoders[i].parameters()) + list(encoders[i].parameters()),
                                               lr=learning_rate, weight_decay=wd)
            else:
                optimizer = torch.optim.Adam(
                    list(encoders[i].parameters()) + list(
                        clientdecoder[i].parameters()), lr=learning_rate, weight_decay=wd)
                optimizer_l = torch.optim.Adam(decoders[i].parameters(), lr=learning_rate, weight_decay=wd)
            notreset = True
            for j in range(used_inner_epoch):
                optimizer.zero_grad()
                x, y = data
                if args['fed'] in {'Ditto', 'indiv'}:
                    feature = clientencoder[i](x)
                else:
                    feature = encoders[i](x)
                #ci = feature.mean(dim=0)
                y_pred = clientdecoder[i](feature)             

                

                loss = nn.MSELoss()(y_pred, y) #+ factor * clloss  
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x)
                #epoch_clloss += clloss.item()
                
                steps += 1
                closs = torch.zeros(1)

                optimizer_l.zero_grad()
                x, y = data
                if args['fed'] in {'Ditto', 'indiv'}:
                    feature = encoders[i](x)
                else:
                    with torch.no_grad():
                        feature = encoders[i](x)
                y_pred = decoders[i](feature)
                loss = nn.MSELoss()(y_pred, y)
                loss.backward()
                optimizer_l.step()
                if j == 0:
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
                if args['fed'] == "PDA":
                    if j > threshold_inner_epochs and loss.item() < trainl * 2:
                        break

                    if notreset and j > second_threshold and loss.item() > trainl * 2:
                        print('client {}, resetting at inner {}/{}, outer {}/{}'.format(i, j, inner_epoch, epoch,
                                                                                        real_epoch))

                        clientdecoder[i] = copy.deepcopy(decoder)
                        clientencoder[i] = copy.deepcopy(encoder)

                        for layer in encoders[i].children():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        optimizer = torch.optim.SGD(
                            list(clientencoder[i].parameters()) + list(
                                clientdecoder[i].parameters()), lr=learning_rate, weight_decay=wd)
                        optimizer_l = torch.optim.Adam(list(decoders[i].parameters()) + list(encoders[i].parameters()),
                                                       lr=learning_rate, weight_decay=wd)
                        notreset = False

                    if j == inner_epoch - 1:
                        print('client {}, unable to break at inner {}/{}, outer {}/{}'.format(i, j, inner_epoch, epoch,
                                                                                              real_epoch))

        decoder.load_state_dict(mo.scalar_mul(torch.ones(len(encoders)) / len(encoders), clientdecoder).state_dict())
        if args['fed'] == 'Ditto':
            encoder.load_state_dict(
                mo.scalar_mul(torch.ones(len(encoders)) / len(encoders), clientencoder).state_dict())

    
        end_time = time.time()
        print("Epoch: {}, Avg_loss: {},  time {}".format(epoch, (epoch_loss) / len(dataset),  end_time - start_time))
        epoch_loss_history.append(epoch_loss / len(dataset))
        trainl = test_loss(dataset, encoder, encoders, decoder, decoders, n_task, device, args, test=0)
        tl = test_loss(dataset, encoder, encoders, decoder, decoders, n_task, device, args, test=1)
    return tl


def test_loss(dataset, encoder, encoders, decoderfunction, decoderfunctions, n_task, device, args, test=1):
    epoch_loss = 0
    epoch_length = 0
    for i in range(len(dataset) // 2):
        if i > args['rich_client_id']:
            break
        data_i = dataset[2 * i + test]
        data = data_i[0].to(device), data_i[1].to(device)
        # print('data size')
        # print(len(data),data[0].size())
        with torch.no_grad():
            x, y = data
            # print(x.size())
            # assert False
            feature = encoders[i](x)
            y_pred  = decoderfunctions[i](feature)
            loss = nn.MSELoss()(y_pred, y)

            epoch_loss += loss.item() * len(x)
            epoch_length += len(x)
    if test == 1:
        print(" Avg test loss: {}, ".format(epoch_loss / epoch_length))
    else:
        print(" Avg train loss: {}, ".format(epoch_loss / epoch_length))
        return epoch_loss / epoch_length


def one_experinemt(args):
    n_task = 500  # args['num_client']
    seed = 0  # args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from misc import Tee
    import time
    import sys
    import os

    output_dir = 'sine_outputs/'
    jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
    output_dir += jour
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))

    print(args)

    dataset = SineData(amplitude_range=(0.8, 1.2), shift_range=(-2., 2.), num_samples=n_task, num_points=5,
                       sparse_middle=True, rci=args['rich_client_id'])

    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    feature_dim = 1  # 128
    device = torch.device('cuda')

    encoder = Encoder(feature_dim=feature_dim).to(device)
    encoders = [Encoder(feature_dim=feature_dim).to(device) for i in range(n_task)]
    decoderfunction = DecoderFunction(feature_dim=feature_dim).to(device)
    decoderfunctions = [DecoderFunction(feature_dim=feature_dim).to(device) for i in range(n_task)]


    num_epochs = args['num_epochs']
    tl = train(dataset, encoder, encoders, decoderfunction, decoderfunctions, num_epochs, n_task, feature_dim, device, args)

    # Create a set of target points corresponding to entire [-pi, pi] range
    x_target = torch.Tensor(np.linspace(0, 1, 100)).to(device)
    x_target = x_target.unsqueeze(1).unsqueeze(0)
    
    fig, axs = plt.subplots(1, 1)
    idxlist = [0, 1, 2, 3, 4, 5]
    for index in range(min(n_task, 3)):
        with torch.no_grad():

            y_target = (decoderfunctions[index](encoders[index](x_target)))
        mu = y_target.detach()

        
        if index == 0:
            axs.plot(x_target.cpu().numpy()[0, :, 0], mu.cpu().numpy()[0, :, 0] * scale, '--',
                         label=fed_dict[args['fed']], color=color_list[index])
        else:
            axs.plot(x_target.cpu().numpy()[0, :, 0], mu.cpu().numpy()[0, :, 0] * scale, '--',
                         color=color_list[index])

        x_train, y_train = dataset[2 * index]
        
        axs.scatter(x_train.cpu().numpy(), y_train.cpu().numpy() * scale, color=color_list[index], marker=MD.values()[index])
        x_train, y_train = dataset[2 * index + 1]
        if index == 0:
            axs.plot(x_train.cpu().numpy(), y_train.cpu().numpy() * scale, color=color_list[index], alpha=0.7,
                     label='Ground Truth',marker=MD.values()[index])
        else:
            axs.plot(x_train.cpu().numpy(), y_train.cpu().numpy() * scale, color=color_list[index], alpha=0.7,marker=MD.values()[index])


    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(output_dir, 'new_sine_{}_{}.png'.format(args['fed'], args['space'])), bbox_inches="tight")
    plt.show()
    if feature_dim == 1:
        fig, axs = plt.subplots(1, 1)
        for index in range(min(n_task, 5)):
            x_target = torch.Tensor(np.linspace(0, 1, 10)).to(device)
            x_target = x_target.unsqueeze(1).unsqueeze(0)
            with torch.no_grad():
                phi = encoders[index](x_target)
                y_pred = decoderfunctions[index](phi)
            if index == 0:
                axs.scatter(phi.cpu().numpy(), y_pred.cpu().numpy(), color=color_list[index], alpha=0.7)
            else:
                axs.scatter(phi.cpu().numpy(), y_pred.cpu().numpy(), color=color_list[index], alpha=0.7)
        plt.xlabel(r'$\Phi(x)$', fontsize=20)
        plt.ylabel('y', fontsize=20)
        # plt.legend(fontsize=20)
        plt.savefig(os.path.join(output_dir, 'feature_{}_{}.png'.format(args['fed'], args['space'])),
                    bbox_inches="tight")
        plt.show()
    return tl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--fed', type=str, default="PDA")
    parser.add_argument('--space', type=str, default="nn")
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--rich_client_id', type=int, default=10)
    
    args = vars(parser.parse_args())

    tstloss = one_experinemt(args)
