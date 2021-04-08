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
                # ref_from = i
                ref_to = i+self.reference
                ref = self.dataframe[i: ref_to]
                
                gt = self.dataframe[ref_to: ref_to+30]

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
                
