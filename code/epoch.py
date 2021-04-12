import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


def train_epoch(predictor, optimizer, dataloader, device, sample=5, value_weigth=0.1, trend_weight=1000):

    predictor.train()

    epoch_loss = 0

    for index, data in tqdm(enumerate(dataloader)):

        optimizer.zero_grad()

        ref, gt = data  # batch, 100, 4
        gt = gt[:, :, 0].to(device, dtype=torch.float)

        ref = torch.unsqueeze(ref, 1)  # batch, channels, 100, 4
        ref = torch.transpose(ref, 2, 3)  # batch, channels, 4, 100
        ref = ref.to(device, dtype=torch.float)

        _, pr = predictor(ref)

        pos = torch.isnan(gt)
        pos = ~ pos

        ###########################################
        # edit by fang
        # remove sampling, and use softmax
        gt = gt[pos]
        pr = pr[pos]

        y_gt = nn.Softmax(dim=0)(gt)

        y_pr = nn.Softmax(dim=0)(pr)

        loss = nn.BCELoss()(y_pr, y_gt) * trend_weight + \
            nn.MSELoss()(pr.unsqueeze(0), gt.unsqueeze(0)) * value_weigth

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        ################################################

    epoch_loss = epoch_loss/(index+1)

    return epoch_loss


@torch.no_grad()
def test_epoch(predictor, dataset, device):

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


def online_trading(predictor, dataset, device):

    import math
    from numpy.random import choice

    predictor.eval()

    predict = []
    truth = []
    forward = []

    slot = 0
    profit = 0
    slots = []

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

        pr = pr.detach().cpu().numpy().tolist()
        gt = gt.detach().cpu().numpy().tolist()

        truth.append(gt[0])

        # if len(forward) == 0:
        #     forward = pr
        #     forward = [p / math.pow(2, i) for i, p in enumerate(forward)]

        # else:
        #     predict.append(forward[0])
        #     forward = forward[1:]  # pop the first element
        #     forward += [0]

        #     pr = [p / math.pow(2, i+1) for i, p in enumerate(pr)]

        #     forward = [sum(x) for x in zip(forward, pr)]

        forward = pr

        # print(forward)
        min_indices = np.argsort(np.array(forward))         # min to max
        # print(min_indices)
        
        max_indices = min_indices[::-1]         # max to min
        # print(max_indices)


        max_dist = [1/math.pow(2, i+1) for i in range(len(max_indices-1))]
        # print(max_dist)

        min_dist = [1/math.pow(2, i+1) for i in range(len(min_indices-1))]
        # print(min_dist)

        max_dist.append(1-sum(max_dist))
        min_dist.append(1-sum(min_dist))

        max_dist = [x for _, x in sorted(zip(max_indices, max_dist))]
        # print(max_dist)
        min_dist = [x for _, x in sorted(zip(min_indices, min_dist))]

        max_dist = np.array(max_dist)
        max_dist /= max_dist.sum()

        min_dist = np.array(min_dist)
        min_dist /= min_dist.sum()

        draw_max = choice(np.array(forward), 1, p=max_dist)
        draw_min = choice(np.array(forward), 1, p=min_dist)

        # print(draw_max)
        draw_max = forward.index(draw_max)
        draw_min = forward.index(draw_min)
        # print(draw_min)

        if slot == 0:
            if draw_min == 0:
                print("buy")
                slot = 1
                profit = profit - gt[0]
                slots.append(1)
            elif draw_max == 0:
                print("sell")
                slot = -1
                profit = profit + gt[0]
                slots.append(-1)
            else:
                print("no thing")
                slots.append(0)
        elif slot == 1:
            if draw_max == 0:
                print("sell")
                slot = 0
                profit = profit + gt[0]
                slots.append(-1)
            else:
                print("no thing")
                slots.append(0)
        else:
            if draw_min == 0:
                print("buy")
                slot = 1
                profit = profit - gt[0]
                slots.append(1)
            else:
                print("no thing")
                slots.append(0)

    # predict += forward
    # return predict, truth
    return slots
