import torch
import torch.nn as nn
import numpy as np

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
        tv_kx = torch.from_numpy(np.array([[-1, 1],[0, 0]])).float().view(2, 2)
        tv_ky = torch.from_numpy(np.array([[-1, 0],[1, 0]])).float().view(2, 2)

        tv_kx.unsqueeze_(0)
        tv_kx.unsqueeze_(0)

        tv_ky.unsqueeze_(0)
        tv_ky.unsqueeze_(0)


        self.conv.weight = nn.Parameter(tv_kx.expand_as(self.conv.weight))
        self.conv2.weight = nn.Parameter(tv_ky.expand_as(self.conv2.weight))
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        tv_x = self.conv(x).contiguous()
        tv_y = self.conv2(x).contiguous()

        # tv = torch.sqrt(torch.pow(torch.abs(tv_x),2) + torch.pow(torch.abs(tv_y),2)).mean()
        tv = (torch.abs(tv_x) + torch.abs(tv_y)).mean()

        return tv