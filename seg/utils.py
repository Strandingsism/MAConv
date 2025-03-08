import os
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(state, is_best, save_dir):
    """Save checkpoint to file"""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_filename)

class PolyLRScheduler(_LRScheduler):
    """Polynomial learning rate scheduler"""
    def __init__(self, optimizer, max_epochs, power=0.9, min_lr=1e-4, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super(PolyLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch/self.max_epochs)**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def get_model_size(model):
    """Calculate model parameters"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params
    }

def cross_entropy2d(input, target, weight=None, reduction='mean'):
    """2D cross entropy loss for segmentation"""
    n, c, h, w = input.size()
    
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    
    if weight is not None:
        weight = weight.view(-1)
    
    loss = torch.nn.functional.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=255
    )
    
    return loss 