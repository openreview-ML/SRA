"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
from os.path import join as ospj
import random
import copy
import shutil
import sys
import yaml

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)

def Cor_CoeLoss( y_pred, y_target):
        x = y_pred
        y = y_target
        x_var = x - torch.mean(x)
        y_var = y - torch.mean(y)
        r_num = torch.sum(x_var * y_var)
        r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
        r = r_num / r_den

        # return 1 - r  # best are 0
        return 1 - r ** 2  # abslute constrain

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_norm_values(norm_family = 'imagenet'):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if norm_family == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std

def save_args(args, log_path, argfile):
    shutil.copy('train.py', log_path)
    modelfiles = ospj(log_path, 'models')
    try:
        shutil.copy(argfile, log_path)
    except:
        print('Config exists')
    try:
        shutil.copytree('models/', modelfiles)
    except:
        print('Already exists')
    with open(ospj(log_path,'args_all.yaml'),'w') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, 'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

class UnNormalizer:
    '''
    Unnormalize a given tensor using mean and std of a dataset family
    Inputs
        norm_family: String, dataset
        tensor: Torch tensor
    Outputs
        tensor: Unnormalized tensor
    '''
    def __init__(self, norm_family = 'imagenet'):
        self.mean, self.std = get_norm_values(norm_family=norm_family)
        self.mean, self.std = torch.Tensor(self.mean).view(1, 3, 1, 1), torch.Tensor(self.std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor * self.std) + self.mean

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)