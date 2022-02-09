import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import PIL
import torchvision.transforms as transforms


class refine_dataset():
    def __init__(self):
        pass

    def refine_data(self, dataset_path):
        dataset_list = os.listdir(dataset_path)
        dataset_list = sorted([os.path.join(dataset_path, x) for x in dataset_list])
        return dataset_list


class Dataset(Dataset):
    def __init__(self, refined_dataset, patch_size):
        self.dataset_list = refined_dataset
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, item):
        # loading image
        img = PIL.Image.open(self.dataset_list[item])
        # expand dimension for batch
        img = np.expand_dims(np.array(img).astype(np.float32), axis=0) / 255.
        # previous code below:
        # img = cv2.imread(self.dataset_list[item], cv2.IMREAD_COLOR)
        # img = np.expand_dims(img.astype(np.float32), axis=0) / 255.
        return img