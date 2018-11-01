import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from clr import CyclicLR


def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
    model.train()
    errors = 0
    losses = 0
    # correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        
#        for param_group in optimizer.param_groups:
#            print(param_group['lr'])

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        losses += loss.item()
        loss.backward()
        optimizer.step()

        # corr = correct(output, target, topk=(1, 5))
        # correct1 += corr[0]
        # correct5 += corr[1]
        cu_mae = f_mae(output, target)
        errors += cu_mae



        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'MAE: {:.2f}({:.2f}). '.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           cu_mae,
                                                           errors / (batch_idx + 1)))
        if batch_idx >= 1200:
            break
    return losses / (batch_idx+1), errors / (batch_idx+1)


def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    # correct1, correct5 = 0, 0
    errors = 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
        # correct1 += corr[0]
        # correct5 += corr[1]
            cu_mae = f_mae(output, target)
            errors += cu_mae

    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f}.'
        'MAE: {:.2f}.'.format(test_loss, errors / (batch_idx+1)))
    return test_loss, errors / (batch_idx+1)


def f_mae(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        values, indices = torch.max(output, 1)
        mae_error = torch.sum(torch.abs(indices - target)) / batch_size

        return mae_error.item()


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./result', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return
