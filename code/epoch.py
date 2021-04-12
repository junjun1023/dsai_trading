import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

def train_epoch(predictor, optimizer, dataloader, device, sample=5):

        predictor.train()

        epoch_loss = 0
        
        for index, data in tqdm(enumerate(dataloader)):

                optimizer.zero_grad()

                ref, gt = data # batch, 100, 4
                gt = gt[:, :, 0].to(device, dtype=torch.float)
        
                
                ref = torch.unsqueeze(ref, 1) # batch, channels, 100, 4
                ref = torch.transpose(ref, 2, 3) # batch, channels, 4, 100
                ref = ref.to(device, dtype=torch.float)

                _, pr = predictor(ref)

                
                pos = torch.isnan(gt)
                pos = ~ pos
                
                gt = gt[pos]
                pr = pr[pos]
                # gt = gt.view(batch, -1)
                # pr = pr.view(batch, -1)
                
                
                # loss = torch.tensor(0, dtype=torch.float).to(device)

                ### sampling
                # for sample in range(5):
                #     for _gt, _pr in zip(gt, pr):
                
                _gt = gt
                _pr = pr
                
                src = (torch.rand(_gt.size(0) * sample) * _gt.size(0)).long()
                det = (torch.rand(_gt.size(0) * sample) * _gt.size(0)).long()


                y_gt = _gt[det] - _gt[src]
                y_pr = _pr[det] - _pr[src]

                y_gt = torch.where(y_gt >= 0, torch.ones_like(y_gt), torch.zeros_like(y_gt))

                y_pr = nn.Sigmoid()(y_pr)


                loss = nn.BCELoss()(y_pr, y_gt)
                                
                loss.backward()
                optimizer.step()
        
                epoch_loss += loss.item()
        

        epoch_loss = epoch_loss/(index+1)

        return epoch_loss


@torch.no_grad()
def test_epoch(predictor, dataset, device):
        
        import math
        predictor.eval()

        predict = []
        truth = []
        forward = []
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        for index, data in tqdm(enumerate(dataloader)):

                ref, gt = data # batch, 100, 4
                gt = gt[:, :, 0].to(device, dtype=torch.float)

                ref = torch.unsqueeze(ref, 1) # batch, channels, 100, 4
                ref = torch.transpose(ref, 2, 3) # batch, channels, 4, 100

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
                        forward = forward[1:] # pop the first element
                        forward += [0]
                        
                        pr = [p / math.pow(2, i+1) for i, p in enumerate(pr)]
                        
                        forward = [sum(x) for x in zip(forward, pr)]

        predict += forward

        return predict, truth



def online_trading(predictor, dataset, device):
        
        import math
        from np.random import choice

        predictor.eval()

        predict = []
        truth = []
        forward = []

        slot = 0
        profit = 0
        slots = []

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        for index, data in tqdm(enumerate(dataloader)):

                ref, gt = data # batch, 100, 4
                gt = gt[:, :, 0].to(device, dtype=torch.float)

                ref = torch.unsqueeze(ref, 1) # batch, channels, 100, 4
                ref = torch.transpose(ref, 2, 3) # batch, channels, 4, 100

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
                        forward = forward[1:] # pop the first element
                        forward += [0]
                        
                        pr = [p / math.pow(2, i+1) for i, p in enumerate(pr)]
                        
                        forward = [sum(x) for x in zip(forward, pr)]


                min_indices = np.argsort(np.array(forward))         # min to max
                max_indices = min_indices[::-1]         # max to min
                max_dist = [1/math.pow(2, i+1) for i in range(len(max_indices))]
                min_dist = [1/math.pow(2, i+1) for i in range(len(min_indices))]

                max_dist = [x for _, x in sorted(zip(max_indices, max_dist))]
                min_dist = [x for _, x in sorted(zip(min_indices, min_dist))]

                draw_max = choice(np.array(forward), 1, p=max_dist)
                draw_min = choice(np.array(forward), 1, p=min_dist)

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