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

def normalised_kendall_tau_distance(values1, values2):
        """Compute the Kendall tau distance."""
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(values1)
        b = np.argsort(values2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        return ndisordered / (n * (n - 1))