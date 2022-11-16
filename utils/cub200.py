import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
from torch.utils.data import Dataset
import pickle
import copy

def load_cub200(image_size, batch_size,partial_rate):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = CUB(root='../../datasets/CUB_200_2011/',
                                train=True,
                                transform=train_transform,
                                partial_rate=partial_rate)
    test_dataset = CUB(root='../../datasets/CUB_200_2011/',
                               train=False,
                               transform=test_transform,
                                partial_rate=partial_rate)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    partialY = train_dataset.train_labels
    return partial_matrix_train_loader,test_loader,partialY,train_dataset.class_name

class CUB(torch.utils.data.Dataset):
    def __init__(self, root, train=True,transform=None, partial_type='binomial', partial_rate=0.1):
        self.root = root
        self.train = train
        self.transform = transform
        self.partial_rate = partial_rate
        self.partial_type = partial_type
        if self.train:
            self.class_name=self.get_class_name()
            self.train_data, self.train_labels = pickle.load(open(os.path.join(self.root, 'train.pkl'), 'rb'))
            assert (len(self.train_data) == 5994 and len(self.train_labels) == 5994)
            self.train_labels = np.array(self.train_labels)
            self.true_labels = copy.deepcopy(self.train_labels)
            self.train_labels = torch.from_numpy(self.train_labels)
            self.train_labels = generate_uniform_cv_candidate_labels(self.train_labels, partial_rate)
            print('-- Average candidate num: ', self.train_labels.sum(1).mean(), self.partial_rate)
        else:
            self.test_data, self.test_labels = pickle.load(open(os.path.join(self.root, 'test.pkl'), 'rb'))
            assert (len(self.test_data) == 5794 and len(self.test_labels) == 5794)

    def __getitem__(self, index):
        if self.train:
            image, target = self.train_data[index], self.train_labels[index]
        else:
            image, target = self.test_data[index], self.test_labels[index]
        image = Image.fromarray(image)
        if self.train:
            each_true_label = self.true_labels[index]
            each_image1 = self.transform(image)
            each_image2 = self.transform(image)
            each_label = target
            return each_image1,each_image2,each_label,each_true_label,index
        else:
            image = self.transform(image)
            return image, target
    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.test_data)

    def get_class_name(self):
        class_name=[]
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            lines=f.readlines()
        for line in lines:
            name=line.split(".")[1][0:-2].replace("_"," ")
        return class_name


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY








