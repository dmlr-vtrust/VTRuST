import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from skimage.util import random_noise
import torchvision
#import cv2
import torchvision.transforms as transforms
import numpy as np
import random
import os
import time
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()


class MyCustomDataset(Dataset):
    def __init__(self, X,y, height, width, transforms=None):
        self.data = X
        self.labels = y
        self.height = height
        self.width = width
        self.transforms = transforms

       
    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = self.data[index]
        img_as_img = Image.fromarray(img_as_np)
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg

############ Model Architecture ##########
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _resnet(block, layers):
    model = ResNet(block, layers)
    return model

def ResNet18():
    return _resnet(BasicBlock, [2, 2, 2, 2])

'''def cifar_clean_testloader():


    test_transform = transforms.Compose([transforms.ToTensor()])
    test_clean_set = torchvision.datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
    test_clean_loader = torch.utils.data.DataLoader(test_clean_set, batch_size=100, shuffle=False, num_workers=2)

    net1.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_clean_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net1(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print('Standard Accuracy')
    print(acc)'''

def cifar_c_testloader(corruption, data_dir, num_classes=10, 
    test_batch_size=100, num_workers=2):

    if num_classes==10:
        CIFAR = torchvision.datasets.CIFAR10
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C')
    elif num_classes==100:
        CIFAR = torchvision.datasets.CIFAR100
        base_c_path = os.path.join(data_dir, 'CIFAR-100-C')
    else:
        raise Exception('Wrong num_classes %d' % num_classes)
    
    # test set:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = CIFAR(data_dir, train=False, transform=test_transform, download=False)
    
    # replace clean data with corrupted data:
    test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))
    print(f"len of test_set.data  {len(test_set.data)}")
    test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    print('loader for %s ready' % corruption)
    test_c_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader




device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data

torch.manual_seed(123)
net1 = ResNet18()
net1 = net1.to(device)
net1.load_state_dict(torch.load('./checkpoint_sub/cifar6_60perc_vtrust.pth')['net']) #model load


# Model
torch.manual_seed(123)
print('==> Building model..')
criterion = nn.CrossEntropyLoss()
    
def test(epoch,model):

    CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur',  'zoom_blur', 'frost',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression']

    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_c_loader = cifar_c_testloader(corruption=corruption, data_dir='./', num_classes=10, 
            test_batch_size=100, num_workers=2)
        print(len(test_c_loader))
        test_seen_c_loader_list.append(test_c_loader)

    # val corruption:
    print('evaluating corruptions...')
    test_c_losses, test_c_accs = [], []
    for corruption, test_c_loader in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_batch_num = len(test_c_loader)
        print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
        batch_avg_meter = AverageMeter()
        test_batch_accs = []
        ts = time.time()
        test_c_loss_meter, test_c_acc_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_c_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # append loss:
                test_c_loss_meter.append(loss.item())
                test_c_acc_meter.append(acc.item())

                batch_avg_meter.append(acc.item())

                if (batch_idx+1)%100==0:
                    test_batch_accs.append(batch_avg_meter.avg)
                    batch_avg_meter = AverageMeter()

        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        # test loss and acc of each type of corruptions:
        test_c_losses.append(test_c_loss_meter.avg)
        test_c_accs.append(test_c_acc_meter.avg)

        # print
        corruption_str = '%s: %.4f' % (corruption, test_c_accs[-1])
        print(corruption_str)
        print('Severity levels (1-5)')
        print(test_batch_accs)
    # mean over 16 types of attacks:
    test_c_loss = np.mean(test_c_losses)
    test_c_acc = np.mean(test_c_accs)

    # print
    print("Corruption accuracies")
    print(test_c_accs)
    avg_str = 'corruption acc: (mean) %.4f' % (test_c_acc)
    print(avg_str)


for epoch in range(1):
    print("Testing")
    test(epoch,net1)
