import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset
from utils.get_partial_dataset import partial_dataset
from utils.utils_algo import generate_uniform_cv_candidate_labels,generate_hierarchical_cv_candidate_labels
import cv2
import os

def prepare_cv_datasets(dataname, batch_size):
    train_transform = transforms.Compose([transforms.ToTensor()])
    if dataname == 'cifar10':
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        ordinary_train_dataset = dsets.CIFAR10(root='../../datasets', train=True, transform=train_transform, download=True)
        test_dataset = dsets.CIFAR10(root='../../datasets', train=False, transform=test_transform)
    elif dataname=='cifar100'or dataname=='cifar100-H':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        ordinary_train_dataset = dsets.CIFAR100(root='../../datasets', train=True, download=True,transform=train_transform)
        test_dataset = dsets.CIFAR100(root='../../datasets', train=False, transform=test_transform)
    elif dataname=='cub':
        image_size=224
        resize=int(image_size / 0.875)
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        test_transform = transforms.Compose([
            transforms.Resize([resize,resize]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        ordinary_train_dataset = CUB(path='../../datasets/CUB_200_2011/',train=True,transform=train_transform)
        test_dataset = CUB(path='../../datasets/CUB_200_2011/',train=False,transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_data=ordinary_train_dataset.data
    train_labels = torch.Tensor(ordinary_train_dataset.targets).long()
    class_name=ordinary_train_dataset.classes
    return train_data,train_labels, test_loader, class_name

def prepare_train_loaders_for_uniform_cv_candidate_labels(dataname,partial_rate, data,labels, batch_size):
    ######
    if dataname=='cifar100-H':
        partialY = generate_hierarchical_cv_candidate_labels(dataname,labels, partial_rate)
    else:
        partialY = generate_uniform_cv_candidate_labels(labels,partial_rate)
    #######
    if dataname=='cifar10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
          ])
    elif dataname=="cub":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else: ## cifar100 cifar100-H
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])
    partial_matrix_dataset = partial_dataset(data, partialY.float(), labels.float(),train_transform,dataname)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return partial_matrix_train_loader, partialY


class CUB(Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)
        self.classes=self.get_class_name()
        
    def get_class_name(self):
        class_name=[]
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            lines=f.readlines()
        for line in lines:
            name=line.split(".")[1][0:-2].replace("_"," ")
            class_name.append(name)
        return class_name
    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
