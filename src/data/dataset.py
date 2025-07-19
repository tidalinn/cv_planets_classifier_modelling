import os
from typing import List, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ClassificationDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, labels: List[List[int]], path: str, transform: Compose = None):
        super().__init__()
        self.dataframe = dataframe
        self.labels = labels
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        path_image = f'{self.dataframe.image_name[index]}.jpg'
        label = torch.tensor(self.labels[index]).float()

        image = cv.imread(os.path.join(self.path, path_image))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label
