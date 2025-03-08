import numpy as np
import torch

class SegmentationMetric:
    """Metric calculation for semantic segmentation"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        
    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist
    
    def update(self, label_preds, label_trues):
        """Update confusion matrix"""
        label_preds = label_preds.cpu().numpy()
        label_trues = label_trues.cpu().numpy()
        self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten())
    
    def get(self):
        """Calculate metrics"""
        hist = self.confusion_matrix
        
        # Calculate IoU for each class
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)  # mIoU
        
        # Calculate accuracy for each class
        acc = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        mean_acc = np.nanmean(acc)  # mAcc
        
        # Calculate global pixel accuracy
        acc_global = np.sum(np.diag(hist)) / (np.sum(hist) + 1e-10)
        
        return {
            'mIoU': mean_iu,
            'mAcc': mean_acc,
            'aAcc': acc_global,
            'IoU': iu,
            'Acc': acc
        }
    
    def reset(self):
        """Reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 