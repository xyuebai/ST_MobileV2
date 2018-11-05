import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2)), # move center, adjust ratio, scaling
        transforms.RandomAffine(90, translate=(0.01, 0.01), shear=30, resample=False,
                                            fillcolor=0), # move center, rotation, shearing
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    # if scale_size != input_size:
    #     t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(augment=True, input_size=224):
    normalize = __imagenet_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return inception_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


def get_loaders(dataroot, val_batch_size, train_batch_size, input_size, workers, b_weights):
    test_data = datasets.ImageFolder(root=os.path.join(dataroot, 'test'), transform=get_transform(False, input_size))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False, num_workers=workers,
                                              pin_memory=True)

    val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=get_transform(False, input_size))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)

    train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'),
                                      transform=get_transform(input_size=input_size))
    if b_weights:
        weights = make_weights_for_balanced_classes(
            train_data.imgs,
            len(train_data.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False,
                                                   num_workers=workers, pin_memory=True, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader
