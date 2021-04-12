import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from tqdm import tqdm
import json

from code import model
from code.data import Dataset
from code import epoch
from code import evaluation

root = os.getcwd()
batch = 8
forecast = 10
samples = 5
dataset = Dataset(path=os.path.join(
    root, "training.csv"), forecast=forecast)

print(dataset[0])
print()
print(dataset[1])
