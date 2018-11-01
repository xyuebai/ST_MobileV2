#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:53:44 2018

@author: yue
"""
import argparse
import time
import utils
import os
import sys
import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--param_path', default=None, help="Path to the folder having params.json")
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--random_sample', action='store_true', default=True, help='whether to sample the dataset with random sampler')
#parser.add_argument('--resume_path', default=None, help='Path to any previous saved checkpoint')
#parser.add_argument('--teacher_checkpoint', default=None, help='Full Path to a trained teacher model')



class STN_MobileNet(nn.Module):
    def __init__(self, use_dropout=False):
        super(STN_MobileNet, self).__init__()

        # Spatial transformer localization-network
        self.dropout = use_dropout 
        self.drop_prob = 0.5
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1 ),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Spatial transformer network forward function
        
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.mobile = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 100)
    
    def stn(self, x):

            xs = self.localization(x)
            xs = xs.view(-1, 32 * 3 * 3)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
    
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)
            return x
   
    def forward(self, x):
        x = self.stn(x)
        x = self.mobile(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
        #return F.log_softmax(x, dim=1)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, params):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    logging.info("Running epoch: {}/{}".format(epoch+1, params.num_epochs))
    for i, (input, target) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)

        if params.cuda:
            input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        error = f_mae(output, target)
        
        mae.update(error,input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % params.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae.val:.3f} ({mae.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, mae=mae))           

def validate(val_loader, model, criterion, params):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if params.cuda:
                input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            error = f_mae(output, target)
            losses.update(loss.item(), input.size(0))
            mae.update(error, input.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % params.print_freq == 0:
                logging.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae.val:.3f} ({mae.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae=mae))

        logging.info('MAE {mae.val:.3f} ({mae.avg:.3f})'
              .format(mae=mae))

    return mae.avg

def f_mae(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        values, indices = torch.max(output, 1)
        mae_error = torch.sum(torch.abs(indices-target))/batch_size
 
        return mae_error.item()


if __name__ == '__main__' :
    global args, best_prec1
    lowest_mae = 100
    args = parser.parse_args()
    params = utils.ParamParser(os.path.join(args.param_path, 'params.json'))

    params.cuda = torch.cuda.is_available()

    utils.setLogger(os.path.join(args.param_path, "train.log"))

    if params.cuda:
        model = STN_MobileNet().cuda()
        
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), params.learning_rate,
                                momentum=params.momentum,
                                weight_decay=1e-4)
 
    logging.info("Loading the datasets")    
    
    # Data loading code
    traindir = os.path.join(params.data, 'TRAIN')
    valdir = os.path.join('crop-cvpr2016', 'TEST')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    dataset_train = datasets.ImageFolder(traindir)                                                                         
                                                                                
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(
            dataset_train.imgs, 
            len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))                     
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, 
        shuffle = False, sampler = sampler, 
        num_workers=params.num_workers, pin_memory=True)

    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=params.batch_size, shuffle=False,
        num_workers=params.num_workers, pin_memory=True)

    logging.info("Datasets loaded, training starts")    
    if args.evaluate:
        validate(val_loader, model, criterion)
        sys.exit('validation done.')

    for epoch in range(params.start_epoch, params.num_epochs):
        
        adjust_learning_rate(optimizer, epoch, params.learning_rate)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, params)

        # evaluate on validation set
        avg_mae = validate(val_loader, model, criterion, params)

        # remember best prec@1 and save checkpoint
        is_best = avg_mae < lowest_mae
        lowest_mae = min(avg_mae, lowest_mae)
        utils.save_checkpoint({"epoch" : epoch + 1,
                               "state_dict":model.state_dict(),
                               'lowest_mae': lowest_mae,
                               "optimizer":optimizer.state_dict()}, 
        isBest=is_best, ckpt_dir=args.param_path)
        if is_best:
            logging.info("New best accuracy found!")

    
    