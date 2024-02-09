import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from operator import itemgetter
import numpy as np
import random
import copy
import cv2
import os
import argparse
from torch.utils.data import Dataset
from PIL import Image
from subsetpack.noise import gaussian_noise,shot_noise,gaussian_blur,impulse_noise,speckle_noise,autocontrast,equalize, posterize, solarize,brightness,contrast,defocus_blur, zoom_blur, pixelate,elastic_transform,frost,jpeg_compression,glass_blur


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
    transforms.ToTensor()
])

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True)


ftrain = []
fatrain = []
ytrain = []
yatrain = []

aug_list = [gaussian_noise, shot_noise, impulse_noise, glass_blur, zoom_blur, frost, brightness, contrast, elastic_transform, pixelate,  jpeg_compression, defocus_blur]


num_classes = 10
num_augs = 12

class_img_list_dict = {}
for i in range(num_classes):
    class_img_list_dict[i] = []

image_id_tgt_im_dict = {}

for batchid, (ip, targ) in enumerate(zip(trainset.data, trainset.targets)):
    class_img_list_dict[targ].append(batchid)
    image_id_tgt_im_dict[batchid] = [targ, ip]

class_aug_dict = {}
for i in range(num_classes):
    class_aug_dict[i] = []


#Sampled Augmentation datasets from 3 rounds

#Round 1
file = open("1204_2041_code2_dict_data_r1_class_aug_dict.txt", "r")

for line in file:
    class_list = []
    class_num, data = line.split("!@")
    class_num = int(class_num)
    data = data.lstrip("[").rstrip("]\n").split(", ")
    for str_num in data:
        class_list.append(int(str_num))

    class_aug_dict[class_num].append(class_list)

convert_img = transforms.Compose([transforms.ToTensor(),transforms.ToPILImage()])

for key, val in class_aug_dict.items():
    for idx in range(num_augs):
        for im_id in val[idx]:
            targ = image_id_tgt_im_dict[im_id][0]
            ip = image_id_tgt_im_dict[im_id][1]

            ftrain.append(ip)
            op1 = aug_list[idx]

            fatrain.append(op1(convert_img(ip),3).astype(np.uint8))

            ytrain.append(targ)
            yatrain.append(targ)


class_aug_dict = {}
for i in range(num_classes):
    class_aug_dict[i] = []

#Round 2
file = open("2112_1204_2041_code2_dict_data_r2_class_aug_dict_mod.txt", "r")

for line in file:
    class_list = []
    class_num, data = line.split("!@")
    class_num = int(class_num)
    data = data.lstrip("[").rstrip("]\n").split(", ")
    for str_num in data:
        class_list.append(int(str_num))

    class_aug_dict[class_num].append(class_list)

file.close()

for key, val in class_aug_dict.items():
    print (f"Class {key}")

    for idx in range(num_augs):
        for im_id in val[idx]:
            targ = image_id_tgt_im_dict[im_id][0]
            ip = image_id_tgt_im_dict[im_id][1]
            op1 = aug_list[idx]
            fatrain.append(op1(convert_img(ip),3).astype(np.uint8))
            yatrain.append(targ)

class_aug_dict = {}
for i in range(num_classes):
    class_aug_dict[i] = []

#Round 3
file = open("2212_dict_data_r3_s10.txt", "r")

for line in file:
    class_list = []
    class_num, data = line.split("!@")
    class_num = int(class_num)
    data = data.lstrip("[").rstrip("]\n").split(", ")
    for str_num in data:
        class_list.append(int(str_num))

    class_aug_dict[class_num].append(class_list)

file.close()

for key, val in class_aug_dict.items():
    print (f"Class {key}")

    for idx in range(num_augs):
        for im_id in val[idx]:
            targ = image_id_tgt_im_dict[im_id][0]
            ip = image_id_tgt_im_dict[im_id][1]
            op1 = aug_list[idx]
            fatrain.append(op1(convert_img(ip),3).astype(np.uint8))
            yatrain.append(targ)

trainset = MyCustomDataset(np.asarray(ftrain),np.asarray(ytrain),32,32,transform_train)
train_augset = MyCustomDataset(np.asarray(fatrain),np.asarray(yatrain),32,32,transform_train)
trainsets = torch.utils.data.ConcatDataset([trainset, train_augset])


temp = np.load('influence_scores_robust.npy')
print(temp.shape)
values = []
for i in range(temp.shape[0]):
    if temp[i]!=0:
       values.append((temp[i],i))

sval = sorted(values,key=itemgetter(0),reverse=True)
subset_trindices = []
for v in sval:
    subset_trindices.append(v[1])

print(len(set(subset_trindices)))
subset_trindices = subset_trindices[:120000]
subset_train = torch.utils.data.Subset(trainsets, subset_trindices)

print(len(set(subset_trindices)))

trainloader = torch.utils.data.DataLoader(
    subset_train, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data/', train=False, download=True, transform=transform_test)

testloaderb = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print(len(trainloader))
# Model
torch.manual_seed(321)
print('==> Building model..')

net = ResNet18()
net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(train_loss/batch_idx)
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloaderb):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sub'):
            os.mkdir('checkpoint_sub')
        torch.save(state, './checkpoint_sub/cifar6_60perc_vtrust.pth')
        best_acc = acc

    print(acc)
    print("Best Standard Accuracy")
    print(best_acc)

for epoch in range(start_epoch,300): #Run till convergence
    print("Training")
    train(epoch)
    print("Testing")
    test(epoch)
    scheduler.step()
