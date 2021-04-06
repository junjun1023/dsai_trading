import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):

        def __init__(self, path, reference=100, preprocessing=None):
                

                self.path = path

                self.reference = reference
                self.dataframe = pd.read_csv(path, names=["open", "high", "low", "close"])

                self.preprocessing = preprocessing
                # self.ids = range(len(self.dataframe) - reference)
        

        def __len__(self):
                return len(self.dataframe) - self.reference
        
        def __getitem__(self, i):

                # read image
                to_date = i+self.reference
                data = self.dataframe[i: to_date]
                
                gt = self.dataframe[to_date: to_date+1]

                # Cvt 2 image and normalize
                # not 255
                data = data.to_numpy(dtype=np.uint8)
                data = data / 255

                gt = gt.to_numpy(dtype=np.uint8)
                gt = gt / 255

                # apply preprocessing
                if self.preprocessing:
                        sample = self.preprocessing(image=data)
        
                return data, gt
                
