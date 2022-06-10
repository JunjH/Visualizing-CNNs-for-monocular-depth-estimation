import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from models import modules, net, resnet, densenet
import net_mask
import loaddata
import util
import numpy as np
import sobel
import net_mask
import os

import pdb

parser = argparse.ArgumentParser(description='single depth estimation')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


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
 
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        model2 = torch.nn.DataParallel(model2, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        model2 = torch.nn.DataParallel(model2, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = torch.nn.DataParallel(model).cuda()
        model2 = torch.nn.DataParallel(model2).cuda()
        batch_size = 8
    model.load_state_dict(torch.load('./pretrained_model/model_' + model_selection))

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model2.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size)
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, model2, optimizer, epoch)

    torch.save(model2.state_dict(), './net_mask/mask_'+model_selection)        



def train(train_loader, model, model2, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    model.eval()
    model2.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth_ = sample_batched['image'], sample_batched['depth']

        image = torch.autograd.Variable(image).cuda()
        depth_ = torch.autograd.Variable(depth_).cuda()
 
        ones = torch.ones(depth_.size(0), 1, depth_.size(2),depth_.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)

        depth = model(image.clone()).detach()

        optimizer.zero_grad()
        mask = model2(image)
        output = model(image*mask)

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        loss_rec = loss_depth + loss_normal + (loss_dx + loss_dy)
        loss_sparse = mask.mean()

        loss = loss_rec + loss_sparse*5

        losses.update(loss_sparse.data[0], image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
   
        batchSize = depth.size(0)

        errors = util.evaluateError(output,depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        totalNumber = totalNumber + batchSize
        averageError = util.averageErrors(errorSum, totalNumber)

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
    print('errors: ', averageError)
 
class AverageMeter(object):
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

if __name__ == '__main__':
    main()
