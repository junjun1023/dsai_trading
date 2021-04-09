import torch

def train_epoch(predictor, optimizer, dataloader):

        predictor.train()

        epoch_loss = 0
        
        for index, data in tqdm(enumerate(dataloader)):

                optimizer.zero_grad()

                ref, gt = data # batch, 100, 4
                gt = gt[:, :, 0].to(predictor.device, dtype=torch.float)
        
                
                ref = torch.unsqueeze(ref, 1) # batch, channels, 100, 4
                ref = torch.transpose(ref, 2, 3) # batch, channels, 4, 100
                ref = ref.to(predictor.device, dtype=torch.float)

                _, pr = predictor(ref)

                
                pos = torch.isnan(gt)
                pos = ~ pos
                
                gt = gt[pos]
                pr = pr[pos]
        #         gt = gt.view(batch, -1)
        #         pr = pr.view(batch, -1)
                
                
                loss = torch.tensor(0, dtype=torch.float).to(predictor.device)

                ### sampling
                for sample in range(samples):
        #             for _gt, _pr in zip(gt, pr):
                
                        _gt = gt
                        _pr = pr
                        
                        src = (torch.rand(_gt.size(0)) * _gt.size(0)).long()
                        det = (torch.rand(_gt.size(0)) * _gt.size(0)).long()


                        y_gt = _gt[det] - _gt[src]
                        y_pr = _pr[det] - _pr[src]

                        y_gt = torch.where(y_gt >= 0, torch.ones_like(y_gt), torch.zeros_like(y_gt))

                        y_pr = nn.Sigmoid()(y_pr)


                        loss += nn.BCELoss()(y_pr, y_gt)
                                
                        loss.backward()
                        optimizer.step()
                
                        epoch_loss += loss.item()
        

        epoch_loss = epoch_loss/(index+1)

        print("\nEpoch: {}, bce= {:.3f}".format(epoch+1, epoch_loss))

        return epoch_loss


@torch.no_grad()
def test_epoch(predictor, dataset):
        
        import math
        predictor.eval()

        predict = []
        truth = []
        forward = []
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        for index, data in tqdm(enumerate(dataloader)):

                ref, gt = data # batch, 100, 4
                gt = gt[:, :, 0].to(predictor.device, dtype=torch.float)

                ref = torch.unsqueeze(ref, 1) # batch, channels, 100, 4
                ref = torch.transpose(ref, 2, 3) # batch, channels, 4, 100

                ref = ref.to(predictor.device, dtype=torch.float)

                _, pr = predictor(ref)
                
                pos = torch.isnan(gt)
                pos = ~ pos
                
                gt = gt[pos]
                pr = pr[pos]
                gt = torch.squeeze(gt)
                pr = torch.squeeze(pr)

                pr = pr.detach().cpu().numpy().tolist()
                gt = gt.detach().cpu().numpy().tolist()
                
                if len(forward) == 0:
                        forward = pr
                        forward = [p / math.pow(2, i) for i, p in enumerate(forward)]
                        
                        truth.append(gt[0])
                else:
                        truth.append(gt[-1])
                        predict.append(forward[0])
                        forward = forward[1:] # pop the first element
                        forward += [0]
                        
                        pr = [p / math.pow(2, i+1) for i, p in enumerate(pr)]
                        
                        forward = [sum(x) for x in zip(forward, pr)]

        predict += forward

        return predict, truth