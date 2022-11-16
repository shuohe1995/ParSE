from torch.utils.data import Dataset


class partial_dataset(Dataset):
    def __init__(self,images,given_label_matrix,true_labels,transforms,dataset):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.train_transform=transforms
        self.dataset=dataset
    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.dataset=='cifar10': # no data augmentation
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            each_image = self.train_transform(self.images[index])
            return each_image, each_label, each_true_label, index
        else: # two random data augmentation
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            each_image=self.images[index]
            each_image1=self.train_transform(each_image)
            each_image2 = self.train_transform(each_image)
            return each_image1, each_image2, each_label, each_true_label, index





