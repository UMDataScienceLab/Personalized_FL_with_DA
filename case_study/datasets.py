# adopt from domainbed repository

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from tft import read_dir
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    "FEMNIST",
    # Big images
    "VLCS",
    "PACS",

]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 100           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 1           # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class FEMNIST(MultipleDomainDataset):
    N_WORKERS = 0
    def __init__(self, root, environments, hparams):
        super().__init__()
        self.N_STEPS = 100

        if root is None:
            raise ValueError('Data directory not specified!')
      
        
        tsfm = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(1.0,))])
        self.datasets = []
        src_dir = os.path.join(root, "leaf/data/femnist/data/all_data")
        clients_raw, groups_raw, data_raw = read_dir(src_dir, 500)
        print("data loaded")
        for clid, username in enumerate(data_raw.keys()):
            #for order, x in enumerate(data_raw[username]["x"]):
            images = data_raw[username]["x"]
            if len(images) == 0:
                continue
            images = np.array(images)
            images = [np.reshape(img,(28,28))for img in images]
            images = torch.stack([tsfm(img) for img in images], dim=0)
            labels = data_raw[username]["y"]
            #print(images.shape)
            #self.datasets.append(TensorDataset(torch.cat((*[tsfm(img) for img in images]), 0), labels))
            self.datasets.append(TensorDataset(images.float(), torch.tensor(labels,dtype=torch.long)))
            #if clid == 0:
            #    print(images[0].shape)
        hparams["num_client"] = len(data_raw.keys())
        hparams["local_updates"] = 5
        print("data transformed")        
        
        self.input_shape = (1, 28, 28,)
        self.num_classes = 62

        hparams["simple_featurizer"] = 1


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    def __init__(self, root, test_envs, hparams):
        self.N_STEPS = 4

        if not 'num_client' in hparams.keys():
            coloring = [(i+0.5)/10 for i in range(10)] #[0.1, 0.2, 0.9]
        else:
            coloring = [(i+0.5)/hparams['num_client'] for i in range(hparams['num_client'])]
        super(ColoredMNIST, self).__init__(root, coloring,
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        self.N_STEPS = 11
        degrees = [0, 15, 30, 45, 60, 75]
        hparams["num_client"] = len(degrees)
        super(RotatedMNIST, self).__init__(root, degrees,
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                #interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        if 'maxenv' in hparams.keys():
            environments = environments[:hparams['maxenv']]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            if environment[:4] == "VLCS":
                path = os.path.join(root, environment,'full/')
            else:
                 path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.N_STEPS = 4
        hparams['num_client'] = 4
        hparams['weighting_factor'] = 0.99
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.N_STEPS = 1
        self.dir = os.path.join(root, "PACS/")
        hparams['weighting_factor'] = 0.99

        hparams['num_client'] = 4
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
