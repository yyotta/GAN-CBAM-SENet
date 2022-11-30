import argparse, os
import torch
import utils, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import torch.nn.functional as F
from GAN_attention import generator, discriminator, GAN_attention

def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='gan',
                        choices=['gan', 'baseline'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'faces_anime'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=96, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--use_cudnn_optim', type=bool, default=True)
    parser.add_argument('--print_frequency', type=int, default=100, help='The frequency of print info while training')
    
    args = parser.parse_args()

    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    
    return args





if __name__ == '__main__':
    args = parse_args()
    
    if args is None:
        print("The argument is None! Check the argument you input!")
        exit()
    
    
    if args.use_cudnn_optim:
        torch.backends.cudnn.benchmark = True

    if args.gan_type == 'baseline':
        # Set baseline module here.
        pass
    elif args.gan_type == 'gan':
        net = GAN_attention(args)
    else:
        raise Exception("[X] There is not a model called : " + args.gan_type)

    net.train()
    print(' [√] Training Done!')


    net.save_imgs(args.epoch)
    print(" [√] Testing finished!")
