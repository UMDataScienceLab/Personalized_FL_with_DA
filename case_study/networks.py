# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from libs import wide_resnet
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

class MNIST_CNN_top(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN_top, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        #self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        #self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        #self.bn1 = nn.GroupNorm(8, 128)
        #self.bn2 = nn.GroupNorm(8, 128)
        #self.bn3 = nn.GroupNorm(8, 128)

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        #x = self.conv2(x)
        #x = F.relu(x)
        #x = self.bn1(x)

        #x = self.conv3(x)
        #x = F.relu(x)
        #x = self.bn2(x)

        #x = self.conv4(x)
        #x = F.relu(x)
        #x = self.bn3(x)

        #x = self.avgpool(x)
        #x = x.view(len(x), -1)
        return x

class MNIST_CNN_bottom(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, in_features, out_features):
        super(MNIST_CNN_bottom, self).__init__()
        #self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        #self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = torch.nn.Linear(in_features, out_features)
        
        '''
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
        '''

    def forward(self, x):
        #x = self.conv1(x)
        #x = F.relu(x)
        #x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.classifier(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)

class OnedEncoder(nn.Module):
    def __init__(self, x_dim=8, h_dim=256, sudo_x_dim=1):
        super(OnedEncoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        #self.bn1 = nn.LayerNorm(middle_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, h_dim)
        self.fc22 = nn.Linear(h_dim, h_dim)
        self.fc23 = nn.Linear(h_dim, h_dim)
        self.fc24 = nn.Linear(h_dim, h_dim)

        #self.bn2 = nn.LayerNorm(middle_dim)
        self.fc3 = nn.Linear(h_dim, sudo_x_dim)
        #self.input_to_hidden = nn.Sequential(*layers)
        self.sc = nn.Linear(x_dim, sudo_x_dim)
        self.n_outputs = sudo_x_dim

    def forward(self, x):
        batch, dimx = x.shape
        #print(batch, dimx)
        sy = torch.relu(self.fc1(x))
        #sy = self.bn1(sy)
        #sy += x[:,-1:]

        sy = torch.relu(self.fc2(sy))+sy
        sy = torch.relu(self.fc21(sy))+sy
        sy = torch.relu(self.fc22(sy))+sy
        sy = torch.relu(self.fc23(sy))+sy
        sy = torch.relu(self.fc24(sy))+sy

        #print(sy.shape)
        #assert False
        #sy += x[:,-1:]
        #sy = self.bn2(sy)
        sy = torch.relu(self.fc3(sy))+sy
        sy = self.sc(x) + sy
        #sy[:,-6:] += x[:,-6:]
        return sy

class OnedDecoder(nn.Module):
    def __init__(self, sudo_x_dim=1,  z_dim=1,middle_dim=256):
        super(OnedDecoder, self).__init__()
        self.fc1 = nn.Linear(sudo_x_dim, middle_dim)
        #self.bn1 = nn.LayerNorm(middle_dim)
        self.fc2 = nn.Linear(middle_dim, middle_dim)
        #self.bn2 = nn.LayerNorm(middle_dim)
        self.fc3 = nn.Linear(middle_dim, middle_dim)
        #self.bn3 = nn.LayerNorm(middle_dim)
        self.fc4 = nn.Linear(middle_dim, middle_dim)
        self.fc45 = nn.Linear(middle_dim, middle_dim)
        self.fc46 = nn.Linear(middle_dim, middle_dim)
        self.fc47 = nn.Linear(middle_dim, middle_dim)
        self.fc48 = nn.Linear(middle_dim, middle_dim)
        #self.bn4 = nn.LayerNorm(middle_dim)

        self.fc5 = nn.Linear(middle_dim, z_dim)
        #self.l = nn.Parameter(torch.zeros((1)) - 2)


    def forward(self, x):
        sy = torch.relu(self.fc1(x))
        #sy = self.bn1(sy)
        sy = torch.relu(self.fc2(sy))+sy
        #sy[:,-6:] += x[:,-6:]

        #sy = self.bn2(sy)
        sy = torch.relu(self.fc3(sy))+sy
        #sy[:,-6:] += x[:,-6:]

        #sy = self.bn3(sy)
        sy = torch.relu(self.fc4(sy))+sy
        #sy[:,-6:] += x[:,-6:]
        sy = torch.relu(self.fc45(sy))+sy
        sy = torch.relu(self.fc46(sy))+sy

        sy = torch.relu(self.fc47(sy))+sy

        sy = torch.relu(self.fc48(sy))+sy


        #sy = self.bn4(sy)
        sy = self.fc5(sy)
        return sy



def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    #print(input_shape)
    if 'printingdata' in hparams.keys():
        return OnedEncoder(sudo_x_dim=256)
    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        #print("using cnn")
        if "simple_featurizer" in hparams.keys():
            return MNIST_CNN_top(input_shape)
        else:
            return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        #print("using wide resnet")
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        #print("using resnet")
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False, hparams={}):
    if "simple_featurizer" in hparams.keys():
        #print("using simple featurizer")
        return MNIST_CNN_bottom(in_features, out_features)
    if 'printingdata' in hparams.keys():
        return OnedDecoder(sudo_x_dim=256)
    if is_nonlinear:
        #print("non linear classifier")
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        #print("linear classifier")
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)