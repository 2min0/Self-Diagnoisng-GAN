import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from new_typing import Any, Callable, Optional
import PIL
from PIL import Image
import torchvision.transforms as transforms


class refine_dataset():
    def __init__(self):
        pass

    def refine_data(self, dataset_path):
        dataset_list = os.listdir(dataset_path)
        dataset_list = sorted([os.path.join(dataset_path, x) for x in dataset_list])
        return dataset_list


class Dataset(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download=False,
    ):

        # super(Dataset, self).__init__(root, transform=transform,
        #                               target_trasnform=target_trasnform)
        self.data = []
        # to avoid error
        self.train = train
        self.target_transform = target_transform
        #

        self.transform = transform
        self.data_refiner = refine_dataset()
        self.dataset_list = self.data_refiner.refine_data(root)

        for filename in self.dataset_list:
            img = cv2.imread(filename)
            # 2048x2048 to 32x32 (to capture feature of images)
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
            # convert cv2 to PIL image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.data.append(img)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # to avoid error
        if download:
            pass

        if self.train:
            pass
        #

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # to avoid error
        if self.target_transform is not None:
            pass
        #

        return img, 1

        # # loading image
        # img = PIL.Image.open(self.dataset_list[item])
        # # expand dimension for batch
        # img = np.expand_dims(np.array(img).astype(np.float32), axis=0) / 255.
        # # previous code below:
        # # img = cv2.imread(self.dataset_list[item], cv2.IMREAD_COLOR)
        # # img = np.expand_dims(img.astype(np.float32), axis=0) / 255.
        # return img