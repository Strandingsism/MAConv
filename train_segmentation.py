import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime

from datasets import CityscapesDataset, SegmentationTransform
from metrics import SegmentationMetric, AverageMeter
from utils import (
    setup_seed, save_checkpoint, PolyLRScheduler,
    get_model_size, cross_entropy2d
)

# Import all models
from MA_repvit_seg import ma_repvit_seg_m1_5
# from DCN_repvit_seg import dcn_repvit_seg_m1_5
# from WT_repvit_seg import wt_repvit_seg_m1_5
# from repvit_seg import repvit_seg_m1_5  

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation networks')
    
    # Dataset parameters
    parser.add_argument('--dataset-dir', type=str, default='city',
                        help='Cityscapes dataset directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2,
                        help='mini-batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['repvit', 'ma_repvit', 'dcn_repvit', 'wt_repvit'],
                        help='model to train')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save-dir', type=str, default='checkpoint',
                        help='directory to save checkpoints')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{args.model}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataset and data loaders
    train_transform = SegmentationTransform(split='train')
    val_transform = SegmentationTransform(split='val')
    
    train_dataset = CityscapesDataset(
        args.dataset_dir, split='train',
        transform=train_transform
    )
    val_dataset = CityscapesDataset(
        args.dataset_dir, split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )
    
    # Select model
    model_dict = {
        'repvit': repvit_seg_m1_5,
        'ma_repvit': ma_repvit_seg_m1_5,
        'dcn_repvit': dcn_repvit_seg_m1_5,
        'wt_repvit': wt_repvit_seg_m1_5
    }
    
    model = model_dict[args.model](num_classes=19)
    model = model.cuda()
    
    # Print model information
    model_info = get_model_size(model)
    print("Model Information:")
    for k, v in model_info.items():
        print(f"{k}: {v:,}")
    
    # Create optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    scheduler = PolyLRScheduler(optimizer, args.epochs)
    
    # Create evaluation metric
    metric = SegmentationMetric(num_classes=19)
    
    # Track best metrics
    best_miou = 0
    
    # Training loop
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss = train(train_loader, model, optimizer, epoch, args)
        
        # Validate
        val_loss, scores = validate(val_loader, model, metric, args)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = scores['mIoU'] > best_miou
        best_miou = max(scores['mIoU'], best_miou)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': best_miou,
        }, is_best, save_dir)
        
        # Log metrics
        print(f"Epoch: {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"mIoU: {scores['mIoU']:.4f}, mAcc: {scores['mAcc']:.4f}, aAcc: {scores['aAcc']:.4f}")
        
        # Save best metrics to JSON
        if is_best:
            best_metrics = {
                'mIoU': float(scores['mIoU']),
                'mAcc': float(scores['mAcc']),
                'aAcc': float(scores['aAcc']),
                'class_IoU': {name: float(iou) for name, iou in zip(CityscapesDataset.classes, scores['IoU'])},
                'class_Acc': {name: float(acc) for name, acc in zip(CityscapesDataset.classes, scores['Acc'])}
            }
            
            with open(os.path.join(save_dir, 'best_metrics.json'), 'w') as f:
                json.dump(best_metrics, f, indent=4)

def train(train_loader, model, optimizer, epoch, args):
    # Switch to train mode
    model.train()
    
    # Setup meter
    losses = AverageMeter()
    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):
        # Move data to CUDA
        images = images.cuda()
        targets = targets.cuda()
        
        # Compute output
        outputs = model(images)
        loss = cross_entropy2d(outputs, targets)
        
        # Record loss
        losses.update(loss.item(), images.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print statistics
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch+1}][{i}/{len(train_loader)}] '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Time: {time.time() - end:.2f}s')
            end = time.time()
    
    return losses.avg

def validate(val_loader, model, metric, args):
    # Switch to eval mode
    model.eval()
    
    # Setup meters
    losses = AverageMeter()
    metric.reset()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            # Move data to CUDA
            images = images.cuda()
            targets = targets.cuda()
            
            # Compute output
            outputs = model(images)
            loss = cross_entropy2d(outputs, targets)
            
            # Record loss
            losses.update(loss.item(), images.size(0))
            
            # Update metrics
            preds = outputs.argmax(1)
            metric.update(preds, targets)
            
            # Print progress
            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}] Loss: {losses.val:.4f} ({losses.avg:.4f})')
    
    scores = metric.get()
    print(f'Testing Results: mIoU: {scores["mIoU"]:.4f}, mAcc: {scores["mAcc"]:.4f}, aAcc: {scores["aAcc"]:.4f}')
    
    return losses.avg, scores

if __name__ == '__main__':
    main()
