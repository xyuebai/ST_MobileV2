from model import STN_MobileNet2
from data import get_loaders
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim
import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import os
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='STN_MobileNetv2 training with PyTorch')
parser.add_argument('--dataroot', metavar='PATH')
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 180],
                    help='Decrease learning rate at these epochs.')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 224).')


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    print(inp.size())
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(model, val, device):
    model.eval()
    with torch.no_grad():
        # Get a batch of training data
        for batch_idx, (data, target) in enumerate(tqdm(val)):

            # samples will be a 64 x D dimensional tensor
            # feed it to your neural network model
            input_tensor, target = data.cpu(), target.cpu()
            transformed_input_tensor = model.stnmod(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2, figsize=(20, 10))
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
            fig_path = 'results/pretrain_stn_2018-11-01_09-53-07/plot/'
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            f.savefig(fig_path + 'batch_' + str(batch_idx) + '.png')



def main():
    args = parser.parse_args()
    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'
    state_dict = torch.load('results/pretrain_stn_2018-11-01_09-53-07/model_best.pth.tar', map_location=device)
    model = STN_MobileNet2(input_size=args.input_size, scale=args.scaling)

    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    optimizer.load_state_dict(state_dict['optimizer'])
    # print("=> loaded checkpoint '{}' (epoch {})".format('results/2018-10-26_07-56-14model_best.pth.tar', state_dict['epoch']))
    train_loader, val_loader, test_loader = get_loaders('/home/baiy/workplace/dataset/cropped-cvpr2016-oa', args.batch_size, args.batch_size, args.input_size, args.workers)
    visualize_stn(model, val_loader, device)
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    main()
