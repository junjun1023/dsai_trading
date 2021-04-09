import torch
import numpy as np

def kendal_tau_distance(score_gt, score_pr):
        from scipy import stats
        if isinstance(score_gt, torch.Tensor):
                score_gt = torch.reshape(score_gt, (-1, ))
                score_gt = score_gt.cpu().detach().numpy()
        elif isinstance(score_gt, list):
                score_gt = np.array(score_gt)
                
        if isinstance(score_pr, torch.Tensor):
                score_pr = torch.reshape(score_pr, (-1, ))
                score_pr = score_pr.cpu().detach().numpy()
        elif isinstance(score_pr, list):
                score_pr = np.array(score_pr)
        
        tau, p_value = stats.kendalltau(score_gt, score_pr)
        return tau