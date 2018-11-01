import argparse
import csv
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributions as tdist
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
from collections import OrderedDict

import flops_benchmark
from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
from model import STN_MobileNet2
from run import train, test, save_checkpoint, find_bounds_clr

parser = argparse.ArgumentParser(description='STN_MobileNetv2 training with PyTorch')
parser.add_argument('--dataroot', required=True, metavar='PATH',
                    help='Path to ImageNet train and val folders, preprocessed as described in '
                         'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Optimization options
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--b_weights', type=int, default=False, help='Balance weights of classes during training')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200, 300],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=5e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=1e-4, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=20,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpoints
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 224).')

# https://github.com/keras-team/keras/blob/fe066966b5afa96f2f6b9f71ec0c71158b44068d/keras/applications/mobilenetv2.py#L30
claimed_acc_top1 = {224: {1.4: 0.75, 1.3: 0.744, 1.0: 0.718, 0.75: 0.698, 0.5: 0.654, 0.35: 0.603},
                    192: {1.0: 0.707, 0.75: 0.687, 0.5: 0.639, 0.35: 0.582},
                    160: {1.0: 0.688, 0.75: 0.664, 0.5: 0.610, 0.35: 0.557},
                    128: {1.0: 0.653, 0.75: 0.632, 0.5: 0.577, 0.35: 0.508},
                    96: {1.0: 0.603, 0.75: 0.588, 0.5: 0.512, 0.35: 0.455},
                    }
claimed_acc_top5 = {224: {1.4: 0.925, 1.3: 0.921, 1.0: 0.910, 0.75: 0.896, 0.5: 0.864, 0.35: 0.829},
                    192: {1.0: 0.901, 0.75: 0.889, 0.5: 0.854, 0.35: 0.812},
                    160: {1.0: 0.890, 0.75: 0.873, 0.5: 0.832, 0.35: 0.791},
                    128: {1.0: 0.869, 0.75: 0.855, 0.5: 0.808, 0.35: 0.750},
                    96: {1.0: 0.832, 0.75: 0.816, 0.5: 0.758, 0.35: 0.704},
                    }


def main():
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = 'pretrain_stn_' + time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    model = STN_MobileNet2(input_size=args.input_size, scale=args.scaling)
    # print(model.stnmod.fc_loc[0].bias.data)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(model)
    print('number of parameters: {}'.format(num_parameters))
    print('FLOPs: {}'.format(
        flops_benchmark.count_flops(STN_MobileNet2,
                                    args.batch_size // len(args.gpus) if args.gpus is not None else args.batch_size,
                                    device, dtype, args.input_size, 3, args.scaling)))

    train_loader, val_loader, test_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size,
                                                        args.input_size,
                                                        args.workers, args.b_weights)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)

    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    if args.find_clr:
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=save_path)
        return

    if args.clr:
        print('Use CLR')
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        print('Use scheduler')
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    best_val = 500

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            # args.start_epoch = checkpoint['epoch'] - 1
            # best_val = checkpoint['best_prec1']
            # best_test = checkpoint['best_prec1']
            args.start_epoch = 0
            best_val = 500
            state_dict = checkpoint['state_dict']

            # if weights from imagenet

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                #     print(k, v.size())
                name = k
                if k == 'module.fc.bias':
                    new_state_dict[name] = torch.zeros(101)
                    continue
                elif k == 'module.fc.weight':
                    new_state_dict[name] = torch.ones(101, 1280)
                    continue
                else:
                    print('else:', name)
                    new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_val = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        loss, test_mae = test(model, test_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    # claimed_acc1 = None
    # claimed_acc5 = None
    # if args.input_size in claimed_acc_top1:
    #     if args.scaling in claimed_acc_top1[args.input_size]:
    #         claimed_acc1 = claimed_acc_top1[args.input_size][args.scaling]
    #         claimed_acc5 = claimed_acc_top5[args.input_size][args.scaling]
    #         csv_logger.write_text(
    #             'Claimed accuracies are: {:.2f}% top-1, {:.2f}% top-5'.format(claimed_acc1 * 100., claimed_acc5 * 100.))

    train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, test_loader, optimizer,
                  criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, best_val)


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, test_loader, optimizer, criterion,
                  device, dtype,
                  batch_size, log_interval, csv_logger, save_path, best_val):
    for epoch in trange(start_epoch, epochs + 1):
        if not isinstance(scheduler, CyclicLR):
            scheduler.step()
        train_loss, train_mae, = train(model, train_loader, epoch, optimizer, criterion, device,
                                       dtype, batch_size, log_interval, scheduler)
        val_loss, val_mae = test(model, val_loader, criterion, device, dtype)
        test_loss, test_mae = test(model, test_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'test_mae': test_mae,
                          'test_loss': test_loss, 'val_mae': val_mae,
                          'val_loss': val_loss, 'train_mae': train_mae,
                          'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_val,
                         'optimizer': optimizer.state_dict()}, val_mae < best_val, filepath=save_path)

        # csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        csv_logger.plot_progress()

        if val_mae < best_val:
            best_val = val_mae

    csv_logger.write_text('Lowest mae is {:.2f}'.format(best_val))


if __name__ == '__main__':
    main()
