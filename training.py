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

# variable setting
root = os.getcwd()
batch = 32
forecast = 30
samples = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

datetime = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")

# dataset
dataset = Dataset(path=os.path.join(root, "training.csv"))

# model
encoder = model.Extractor(in_channels=1, out_channels=1,
                          use_batchnorm=True, maxpool=False)
decoder = model.Decoder(classes=forecast)  # forcast 30 days


predictor = model.Model(encoder=encoder, decoder=decoder).to(device)
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)

print(encoder, decoder)

#  dataloader
trainloader = DataLoader(dataset, batch_size=batch,
                         shuffle=False, num_workers=2)

x, y = dataset[0]
print(x.shape, y.shape)

# print(summary(predictor,  (1, 4, 100)))

# training
predictor.train()
train_info = {
    "kendal": []
}

kendal_max = 1
for e in range(500):

    train_loss = epoch.train_epoch(predictor, optimizer, trainloader, device)
    pr, gt = epoch.test_epoch(predictor, dataset, device)

    kendal = evaluation.normalised_kendall_tau_distance(gt, pr)
    train_info["kendal"].append(kendal)

    print("Epoch: {}, loss = {:.5f}, kendal = {:.5f}".format(
        e+1, train_loss, kendal))

    if kendal < kendal_max:
        checkpoint = {
            'model_stat': predictor.state_dict(),
            'optimizer_stat': optimizer.state_dict(),
        }

        print("model saved")
        torch.save(checkpoint, os.path.join(root,
                                            "code",
                                            "{}.pth".format(datetime)))
        kendal_max = kendal
    with open(os.path.join(root, "code", "{}.json".format(datetime)), 'w') as f:
        json.dump(train_info, f)
