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

from sklearn.preprocessing import normalize


@torch.no_grad()
def test(predictor, dataset):

    import math
    predictor.eval()

    predict = []
    truth = []
    forward = []

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=2)

    for index, data in tqdm(enumerate(dataloader)):

        ref, gt = data  # batch, 100, 4
        gt = gt[:, :, 0].to(device, dtype=torch.float)

        ref = torch.unsqueeze(ref, 1)  # batch, channels, 100, 4
        ref = torch.transpose(ref, 2, 3)  # batch, channels, 4, 100

        ref = ref.to(device, dtype=torch.float)

        _, pr = predictor(ref)

        pos = torch.isnan(gt)
        pos = ~ pos

        gt = gt[pos]
        pr = pr[pos]
#         gt = torch.squeeze(gt)
#         pr = torch.squeeze(pr)

        pr = pr.detach().cpu().numpy().tolist()
        gt = gt.detach().cpu().numpy().tolist()

        truth.append(gt[0])

        if len(forward) == 0:
            forward = pr
            forward = [p / math.pow(2, i) for i, p in enumerate(forward)]
        else:
            predict.append(forward[0])
            forward = forward[1:]  # pop the first element
            forward += [0]

            pr = [p / math.pow(2, i+1) for i, p in enumerate(pr)]

            forward = [sum(x) for x in zip(forward, pr)]

    predict += forward

    return predict, truth


if __name__ == '__main__':
    # parameter
    root = os.getcwd()
    batch = 8
    forecast = 10
    samples = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # dataset
    dataset = Dataset(path=os.path.join(
        root, "training.csv"), forecast=forecast)

    print(len(dataset))
    x, y = dataset[0]
    print(x.shape, y.shape)

    # load path
    model_path = 'code/2021-04-12_18-17.pth'

    encoder = model.Extractor(in_channels=1, out_channels=1,
                              use_batchnorm=True, maxpool=False)
    decoder = model.Decoder(classes=forecast)  # forcast 30 days

    predictor = model.Model(encoder=encoder, decoder=decoder).to(device)

    print(predictor)
    predictor.load_state_dict(torch.load(
        model_path, map_location=device)['model_stat'])

    predict, truth = test(predictor, dataset)

    # print(normalize([truth]))
    # print(normalize([predict]))

    # plt.plot(normalize([truth])[0], 'b')
    # plt.plot(normalize([predict])[0], 'orange')
    # print(predict)
    plt.grid()
    plt.plot(truth, 'b')
    plt.plot(predict, 'orange')
    plt.savefig('result.png')
    plt.show()
