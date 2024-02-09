#Defining dataset class

import torch
import os
import random
import numpy as np
from PIL import Image
from skimage.util import random_noise
from skimage.filters import gaussian
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from subsetpack.noise import gaussian_noise,shot_noise,gaussian_blur,impulse_noise,speckle_noise,autocontrast,equalize, posterize, solarize,brightness,contrast,defocus_blur, zoom_blur, pixelate,elastic_transform,frost,jpeg_compression,glass_blur
################# Creating training dataset class #################
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
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#Custom BatchSampler
class RandomBatchSampler(torch.utils.data.Sampler):

    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler

    def __iter__(self):
        randind = random.sample(list(self.batch_sampler),len(self.batch_sampler))
        return iter(randind)

    def __len__(self):
        return len(self.batch_sampler)


class Dataset(object):

    def __init__(self,confdata):

        #self.trainset = Dataset_train()
        self.transform_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True)

        self.transform_test = transforms.Compose([
                                            transforms.ToTensor(),
                                            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
        self.testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True,transform=self.transform_test)
        self.confdata = confdata

    def load_data(self):

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

        for batchid, (ip, targ) in enumerate(zip(self.trainset.data, self.trainset.targets)):
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

        #adding 20k aug images
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


        #adding 60% orig images to aug images
        for key, val in class_aug_dict.items():
            print (f"Class {key}")

            for idx in range(num_augs):
                # print (f"Curr aug : {aug_list[idx]}")
                
                for im_id in val[idx]:
                    targ = image_id_tgt_im_dict[im_id][0]
                    ip = image_id_tgt_im_dict[im_id][1]

                    # ftrain.append(ip)
                    op1 = aug_list[idx]
                    # print (op1, type(op1))
                    fatrain.append(op1(convert_img(ip),3).astype(np.uint8))

                    # ytrain.append(targ)
                    yatrain.append(targ)


        trainset1 = MyCustomDataset(np.asarray(ftrain),np.asarray(ytrain),32,32,self.transform_train)
        train_augset = MyCustomDataset(np.asarray(fatrain),np.asarray(yatrain),32,32,self.transform_train)
        self.trainset2 = torch.utils.data.ConcatDataset([trainset1, train_augset])

        sampler = RandomBatchSampler(torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SequentialSampler(self.trainset2),batch_size=self.confdata['trainbatch'],drop_last=False))
        trainloader = torch.utils.data.DataLoader(self.trainset2,batch_sampler=sampler)

        print("Length of trainloader: ")
        print(len(trainloader))

        ######## Subset of validation set indices to be used for computing value function #########
        subset_indices = [3931,  787, 9037, 2093, 3403, 6198, 4153, 9401, 8555, 6480, 1477, 7734,
        2200, 9095, 3217, 6508, 6787, 7774, 8753, 3195, 5433, 2385,  298, 8668,
         966, 5049, 1802, 1169, 5202, 1597, 1130,  223,   70, 7052, 7875, 5275,
        5638,  898, 5441, 7044, 3046, 6796, 3661, 3740, 6143, 9969,  371, 1996,
        3927, 3891, 4853, 3223, 8155, 1681, 9751, 3520, 4147, 5324, 9301, 8977,
        2007, 1369, 3681, 9870, 2452, 9935, 5743, 6333, 6042, 2601, 6683, 7609,
         478, 8845, 5886, 9830, 2925, 1575, 8978, 4339, 6462, 2027, 5059, 8897,
        8953, 2667,  677, 8906, 8159, 1424, 2475, 6730, 7178, 1158, 5098, 9011,
        7533, 9598, 3911, 3407, 9219, 8374, 5824, 3010, 2607, 2190, 8337, 9028,
        8100, 3218, 7372, 9884, 3883, 6699, 5944, 7498, 7340,  751, 6619, 4148,
        8956,  233, 2421, 3714, 1915, 4187, 7695, 1393, 2678, 1995, 1161, 8882,
        6136, 2884, 5472, 2721,   84, 7900, 1114, 1828, 4580, 4769, 3009, 8649,
        2196, 3674, 9535, 4281, 5238, 6972, 1809, 1718, 3559, 2098, 5140, 2457,
        8450, 2302, 4542, 7337, 1038, 1432, 1540, 9919, 7239, 6291, 1101, 5324,
        5136,   35, 7403, 7126, 5975, 2897, 8563,  276, 8893, 3334, 8461,  558,
         858, 3840,  393, 6594,  973, 3030,  163, 1671, 8345,   89, 9470, 5926,
        4830, 7840, 4200, 9457, 7081, 5307, 5075, 7901, 9334, 2244, 5877,  401,
         175, 9469, 7387, 4376, 8069, 3356, 1887, 4454, 8290, 1860, 8342, 7624,
        2334, 4033, 8871, 1738, 9796,  699, 2105, 8681, 9676, 1458, 8954, 5419,
         400, 3312, 9167, 3859, 7466, 5339, 6708, 1373, 8989, 6925, 9961, 9404,
        6573, 1892, 5266, 5121, 6344, 4770, 2506, 1398, 4477, 9519,  164, 4329,
        3396, 4508, 2879, 1613, 1631, 3894, 3897, 8814, 6983, 8458, 5328, 2295,
        8674, 7034, 4870, 4434, 9201, 8114, 3510,  977,  350, 7326, 9358, 6057,
        4952, 4188, 3081, 7216, 2060, 8913, 2059,  277, 1988, 1324,  618, 6239,
        5164, 8070, 6947, 6424, 3111, 7730, 9072, 4285, 2721, 5364,  736, 9366,
        7434, 5246, 2920, 9535, 5407, 7768, 6199, 8621, 3312, 2945, 9461, 5093,
        3298, 8497,  186, 8724, 8406, 6843, 9769, 9309, 1239, 7198, 1505, 9542,
        4403,  253, 1579,  282, 6906, 7894, 7616, 7287,  870,  694, 8208, 2572,
        4075,  435, 8718, 9967, 2828, 9941, 9298, 4536, 8576, 8074, 8659, 6814,
        4040, 5515, 7980, 8782,  742, 7119, 2759, 8520, 1757, 2868, 5859, 7830,
        2395, 4589, 1653, 6908, 1475, 8094, 5729, 2776, 2195, 8454, 3194, 9282,
        2740, 2810, 2775, 1635, 1291, 1413, 6260, 4176, 5144, 3344, 2763, 4866,
        4191, 9116, 5569, 8005,  779, 8885, 8645, 1299, 5861, 6932, 7442, 5353,
        5058, 1421, 5829, 4625, 5082, 4344, 8629, 8088, 4833, 1251,  736,  772,
        7405, 7700, 5690, 2472, 3275, 3714, 5092, 8290, 5569, 3738, 5447, 3842,
         611, 4640, 7876, 9941, 5255, 8458, 9888, 9740, 7067, 2642, 2918, 5709,
        2632, 2222, 9127, 4917, 9142, 7335, 8823, 5234, 9745, 5302, 4943, 8512,
        2309, 3181, 9778, 5101, 9286,  945, 1877, 6426, 5042, 6693, 5866, 6204,
        4630, 3082, 6596, 3786, 7820, 7690, 9825, 3403, 5803, 5448,  399, 7655,
         474, 4439, 8000, 9025,  119, 5972, 8293, 1810,  689, 9312, 5991, 8546,
        4006, 6478, 8707, 9881, 5626, 5389, 9469, 7737,  666, 3640, 7726, 5614,
         654, 9194,  531, 2425, 7944, 5630, 2854, 7977, 4797,  753, 7536, 6869,
        8740, 4418, 3525, 2810, 2971, 9519, 9346, 5006, 9415, 2206, 1251, 9253,
        8076, 7949, 1197, 6741, 9082, 1213, 2294, 6322, 4984, 7998, 3446, 7075,
        3663, 6907, 1160, 6927, 1614, 7869, 6938, 4049, 6983, 1690, 1726,  629,
        5728, 9530, 3058, 5600, 5108,  420, 2279, 6550, 7239, 2445, 2358,  535,
        2365, 4424, 6462, 3691, 6874, 2546, 6750, 8276, 5555, 8699,  404, 1220,
         303,  794, 2214, 1375, 6221, 5957, 3869,  230, 7070, 1789, 6805, 1537,
        2936, 7005, 6558, 6722, 6133, 5618, 4246, 1770,  417, 2766,  672, 5461,
        5519, 1629, 8978, 4205, 7698, 8404, 2052, 6490, 8007, 8646, 9671, 1882,
        9555, 4484, 4123, 8334, 4384, 9801, 1734, 8727, 5523, 8376, 7851,  280,
          67,  697, 7792, 1018, 1187, 4226, 3311, 1752, 5996, 8384,  421, 4822,
        4815, 9868, 5374,  433, 8967, 2833, 5386, 5421, 8851, 9471, 6794, 6850,
        8408, 3916, 4862, 6920, 5018, 6625, 2855, 6983, 6792, 1430, 9160, 9523,
        6100, 7110, 5331,  606, 3323, 8398, 4095, 5960, 9491, 2632, 9266, 7723,
        2774,  377, 4547, 1080, 4013, 2576, 9600,  168, 8602, 1214, 5474,  266,
        2754, 6240,  917, 8777, 2573, 5457, 5642, 9331, 3585,  759, 4979,  659,
        2467, 9321, 2837, 2492,  959, 8353,  444, 2748, 6760, 3189, 4766, 4911,
        1779, 5781, 4971, 3567, 9518, 8181, 2893, 8746, 2800, 4340,  892, 3822,
        7755, 8167, 8793,  757,  870, 6267,  452, 2773, 1406, 1787, 8478, 2457,
        4518, 5107, 7134, 9588, 9042, 8279, 1740, 8187, 7242, 6972, 7201, 1921,
        2892, 3495,  417, 7954, 7568, 1751, 7951, 6964, 8515, 5557, 6844, 4709,
         207, 5140,  985, 5353, 9499,  714, 3527, 8650, 6180, 5081,  650, 8424,
        2025, 9194, 7185, 1890,  321, 1820, 5006, 9469, 9458, 3474, 5839, 8193,
        8316, 2302, 7837, 9436, 2443, 9611, 9269, 8885, 6630, 3293, 7183, 4341,
        3104, 5923, 7546, 8417, 2966, 8432, 7323, 2523, 2392, 1869, 8470, 5781,
        5454, 2902, 6327, 6871, 9299, 8627,  375, 9219, 5296, 8261, 6970, 6440,
        9416, 7475, 1754, 2137, 8365,  181, 2917, 5660, 5369, 5955, 8289,  206,
        3450, 6610, 2115, 5530, 9779,  407, 3387, 4520,  197, 3585, 4576, 8411,
        9924, 9884, 3611, 1220, 6801, 3264, 1345, 4822, 8530, 9536, 5469, 3198,
        2094, 4712, 4345, 5896, 1133, 6915, 8677, 9905, 9037, 3460, 8398,  625,
         485, 8901, 6767,  660, 3465, 4707, 8034, 9951, 9979, 7635, 8221, 3198,
        4690, 4695, 4108, 5411, 1986, 3557, 3106, 3553,  462, 5581, 5048, 4177,
         119, 9608,  861, 9203, 5022, 9921, 4891, 5698,  285, 3909,    1, 3327,
        8060, 1754, 3228, 7108, 3632, 5388, 1165, 2799, 3193, 7903, 1819,  324,
        2760, 8348, 3843, 9110, 9310, 8593, 9014, 4412, 4615, 4290, 1803, 7076,
         630, 2054, 8183, 7630, 4213, 8118, 8109, 5619,   51, 7331, 8763, 8005,
        7934, 5920, 9288,  688, 9459, 7734, 4209,  802, 7661, 3584, 2262, 7451,
        6968, 1370, 7544, 6422, 7199, 8445, 3293, 8395, 2624, 2985, 2811, 8654,
        5169, 5724,  460,  356, 8258, 7928, 9368,  355, 4905, 2073,  267,  231,
        3340, 1558, 5377, 1992, 4370, 5017, 9984, 1916, 9055, 7130, 2212, 6744,
        4372, 6677, 1939, 8008, 7356, 1515, 7606, 5127,  563, 4740, 5212, 8827,
        4992, 8039, 2608, 9717, 2196, 7930,    1, 9271, 5828, 8779, 4722, 9631, 7058, 6197, 8390, 4385]

        subset_sev_indices = []
        for ind in subset_indices:
            subset_sev_indices.append(ind+20000) #severity 3
        print(len(subset_sev_indices))

        subset_test = torch.utils.data.Subset(self.testset, subset_indices)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.confdata['testbatch'], shuffle=False, num_workers=2)
        testsubloader = torch.utils.data.DataLoader(subset_test, batch_size=self.confdata['testbatch'], shuffle=False, num_workers=2)

        self.testset.data = np.load(os.path.join('../../CIFAR-10-C', 'gaussian_noise.npy'))
        self.testset.targets = torch.LongTensor(np.load(os.path.join('../../CIFAR-10-C', 'labels.npy')))

        subset_noise_test = torch.utils.data.Subset(self.testset, subset_sev_indices)
        testsub_noise_loader = torch.utils.data.DataLoader(subset_noise_test, batch_size=self.confdata['testbatch'], shuffle=False, num_workers=2)
        testloader_s = torch.utils.data.DataLoader(subset_noise_test, batch_size=1, shuffle=False, num_workers=2)

        self.testset.data = np.load(os.path.join('../../CIFAR-10-C', 'brightness.npy'))
        self.testset.targets = torch.LongTensor(np.load(os.path.join('../../CIFAR-10-C', 'labels.npy')))

        subset_noise2_test = torch.utils.data.Subset(self.testset, subset_sev_indices)
        testsub_noise2_loader = torch.utils.data.DataLoader(subset_noise2_test, batch_size=self.confdata['testbatch'], shuffle=False, num_workers=2)
        testloader_s2 = torch.utils.data.DataLoader(subset_noise2_test, batch_size=1, shuffle=False, num_workers=2)


        return trainloader, testsubloader, self.trainset2, self.testset, subset_test,testsub_noise_loader,testloader_s,testsub_noise2_loader,testloader_s2
