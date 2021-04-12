import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_epoch(predictor, optimizer, dataloader, device, sample_point=5, value_weigth=1, trend_weight=1):

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
