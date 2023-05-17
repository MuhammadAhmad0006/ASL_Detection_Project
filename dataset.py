import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from typing import List

import csv


class SignLanguage(Dataset):
#this class focuses on preprocessing and spliting the data into train and test

    @staticmethod
    def get_label_mapping():
        # All labels from the CSV files are mapped to their corresponding numbers 
        #the data set excludes J and Z so those corrisponding numbers have been removed 
        mapping = list(range(25))
        mapping.pop(9)
        return mapping

    @staticmethod
    def read_label_samples_from_csv(path: str):
        # Extracts data that meets the requirements of being the same size and label
        mapping = SignLanguage.get_label_mapping()
        labels, samples = [], []
        with open(path) as f:
            _ = next(f) 
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self,
    #This is Where the images get converted to numpy arrays 
            path: str="data/sign_mnist_train.csv",
            mean: List[float]=[0.485],
            std: List[float]=[0.229]):

        labels, samples = SignLanguage.read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
#This preprossing steps skews the images to have different random orentations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }


def get_train_test_loaders(batch_size=32):
    #Train test split with loaders to use across different files
    trainset = SignLanguage('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = SignLanguage('data/sign_mnist_test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


if __name__ == '__main__':
    loader, _ = get_train_test_loaders(2)
    print(next(iter(loader)))