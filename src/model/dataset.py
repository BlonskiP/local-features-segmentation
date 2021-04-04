import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from matplotlib import pyplot as plt

class VoronoiRegionDataset(Dataset):
    def __init__(self, root):
        assert os.path.exists(root)
        LABELS_PATH = os.path.join(root, "labels.csv")
        self.labels = pd.read_csv(LABELS_PATH)
        self.labels.drop('index', axis=1)
        self.root = root
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index, as_tensor=False):
        path = os.path.join(self.root, self.labels.iloc[index, 1])
        img = io.imread(path)
        if as_tensor:
            img = self.toTensor(img)
        label = torch.tensor(int(self.labels.iloc[index, 2]))
        return img, label
    def plot(self,index):
        img, label = self.__getitem__(index)
        filename = self.labels.iloc[index, 1]
        fig = plt.figure()
        fig.suptitle(f'{filename}', fontsize=20)
        plt.imshow(img)
        plt.show()
    def __getlabels__(self):
        return self.labels
    
