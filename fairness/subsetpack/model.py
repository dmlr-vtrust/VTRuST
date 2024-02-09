#Underlying model architecture

#In this case, ResNet18 with a lower kernel size,stride and padding appropriate for cifar10/cifar100 is defined
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms


class Network(nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(11,64) 
    self.activ1 = nn.ReLU()
    self.layer2 = nn.Linear(64, 32)
    self.activ2 = nn.ReLU()
    self.layer3 = nn.Linear(32,2)


  def forward(self, x):
    out = self.activ1(self.layer1(x))
    out = self.activ2(self.layer2(out))
    out = self.layer3(out)

    return out

class Model(object):

    def __init__(self):
        pass

    def netmodel(self):
        model = Network()
        return model
