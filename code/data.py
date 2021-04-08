import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):

        def __init__(self, path, forecast=30,reference=100, preprocessing=None):
                

                self.path = path
                self.forecast = forecast
                self.reference = reference
                self.dataframe = pd.read_csv(path, names=["open", "high", "low", "close"])

                self.preprocessing = preprocessing

                for _ in range(forecast-1):
                        self.dataframe = self.dataframe.append(
                                pd.Series({"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan}), 
                                ignore_index=True)

                # print(len(self.dataframe))

                # self.ids = range(len(self.dataframe) - reference)
        

        def __len__(self):
                return len(self.dataframe) - (self.forecast - 1) - self.reference - 1
        
        def __getitem__(self, i):

                # read image
                # ref_from = i
                ref_to = i+self.reference
                
                ref = self.dataframe[i: ref_to]

                gt = self.dataframe[ref_to: ref_to+self.forecast]
                

                # Cvt 2 image and normalize
                # not 255
                ref = ref.to_numpy(dtype=np.double)
                # data = data / 255

                gt = gt.to_numpy(dtype=np.double)
                # gt = gt / 255

                # apply preprocessing
                if self.preprocessing:
                        sample = self.preprocessing(image=ref)
        
                return ref, gt
                
