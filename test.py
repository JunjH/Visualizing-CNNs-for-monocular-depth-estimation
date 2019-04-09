import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from models import modules, net, resnet, densenet, senet
import net_mask
import loaddata
import util
import numpy as np

import os
import matplotlib
import matplotlib.image
matplotlib.rcParams['image.cmap'] = 'viridis'

import pdb

parser = argparse.ArgumentParser(description='single depth estimation')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--name', default='train2_2', type=str,
                    help='name of experiment')

def define_model(encoder='resnet'):
    if encoder is 'resnet':
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if encoder is 'densenet':
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if encoder is 'senet':
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    global args
    args = parser.parse_args()

    model_selection = 'resnet'
    model = define_model(encoder = model_selection)
    original_model2 = net_mask.drn_d_22(pretrained=True)
    model2 = net_mask.AutoED(original_model2)  
 
    model = torch.nn.DataParallel(model).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()

    model.load_state_dict(torch.load('./pretrained_model/model_' + model_selection))
    model2.load_state_dict(torch.load('./net_mask/mask_' + model_selection))

    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, model2,'mask_'+model_selection)

 



def test(train_loader, model, model2, dir):
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    model.eval()
    model2.eval()

    # if not os.path.exists(dir):
    #     os.mkdir(dir)

    for i, sample_batched in enumerate(train_loader):
        image, depth_ = sample_batched['image'], sample_batched['depth']

        image = torch.autograd.Variable(image, volatile=True).cuda()
        depth_ = torch.autograd.Variable(depth_, volatile=True).cuda(async=True)
 
        depth = model(image.clone())

        mask = model2(image)
        output = model(image*mask)

        batchSize = depth.size(0)
        errors = util.evaluateError(output,depth_)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        totalNumber = totalNumber + batchSize
        averageError = util.averageErrors(errorSum, totalNumber)

        # mask = mask.squeeze().view(228,304).data.cpu().float().numpy()
        # matplotlib.image.imsave(dir+'/mask'+str(i)+'.png', mask)
 
    print('rmse:',np.sqrt(averageError['MSE']))

 



if __name__ == '__main__':
    main()
