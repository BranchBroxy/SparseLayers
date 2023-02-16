import torch

import torchvision
import torchvision.transforms as transforms


def load_mnist(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


import torch
import numpy as np


def add_noise_to_mnist_dataset(dataset, noise_level):
    noisy_dataset = []
    for data in dataset:
        image, label = data
        # Add noise to the image
        image = image + noise_level * torch.randn(image.size())
        # Clip the image to be between 0 and 1
        image = torch.clamp(image, 0, 1)
        # Add the noisy data to the new dataset
        noisy_dataset.append((image, label))
    return noisy_dataset


