######### This class will compute values/importance weights for all data points using selected trajectories from CheckSel #############

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np
import copy
import os
import argparse
import time
import pickle


class DataValue(object):

    def __init__(self,train,testset,testloader,noiseloader,noise2loader,model,helperobj,confdata):

        self.trainset = train
        self.testset = testset
        self.testloader = testloader
        self.noise_loader = noiseloader
        self.noise2_loader = noise2loader
        self.model = model
        self.rootpath = confdata['root_dir']
        self.bsize = confdata['trainbatch']
        self.keeprem = confdata['retain_scores']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume = confdata['resume']
        self.helperobj = helperobj
        self.confdata = confdata


    ############ Computes values for datapoints assigned to the selected batches during training ###########

    def score_trajbatch(self):

        cpind = []
        alphaval = []
        indcp = np.load(self.rootpath+'trajp_value_indices.npy')
        for ind in range(indcp.shape[0]):
            cpind.append((indcp[ind][1],indcp[ind][2]))
            alphaval.append(indcp[ind][0])

        subset_trindices = []
        btlist = set()
        batches = os.listdir(self.rootpath+'trajp/')
        res = [ 0 for i in range(len(self.trainset)) ]
        cpv = [ [] for i in range(len(self.trainset)) ]
        repeat = [ 0 for i in range(len(self.trainset)) ]
        fea = {}
        testgr = {}
        dict_contrib = {}
        ind = 0

        modelparam = {}
        numind = []

        rescount = 0
        for batchv in batches:
                net = torch.load(self.rootpath+'trajp/'+batchv)
                subset_pickindices =[]
                ep = int(batchv.split('_')[1].split('_')[0])
                bt = int(batchv.split('_')[3].split('.')[0])
                alpha = alphaval[cpind.index((ep,bt))]
                start = ((self.bsize-1)*bt)+bt
                end = start + (self.bsize-1)
                test_gradb = None
                for testi,(data,datasub) in enumerate(zip(self.noise_loader,self.testloader),0):
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    inputs_sub_test, targets_sub_test = datasub
                    inputs_sub_test, targets_sub_test = inputs_sub_test.to(self.device), targets_sub_test.to(self.device)

                    predn_noise = net(inputs)
                    predn_clean = net(inputs_sub_test)

                    loss = self.helperobj.calc_loss_test(predn_clean, predn_noise, targets)

                    if test_gradb is None :

                        test_gradb = self.helperobj.get_grad_test(loss, net)
                        test_gradb = test_gradb.unsqueeze(0)
                    else:
                        test_gradb = torch.cat((test_gradb, self.helperobj.get_grad_test(loss, net).unsqueeze(0)), axis = 0)


                for testi,(data,datasub) in enumerate(zip(self.noise2_loader,self.testloader),0):
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    inputs_sub_test, targets_sub_test = datasub
                    inputs_sub_test, targets_sub_test = inputs_sub_test.to(self.device), targets_sub_test.to(self.device)

                    predn_noise = net(inputs)
                    predn_clean = net(inputs_sub_test)

                    loss = self.helperobj.calc_loss_test(predn_clean, predn_noise, targets)

                    if test_gradb is None :
                        test_gradb = self.helperobj.get_grad_test(loss, net)
                        test_gradb = test_gradb.unsqueeze(0)
                    else:
                        test_gradb = torch.cat((test_gradb, self.helperobj.get_grad_test(loss, net).unsqueeze(0)), axis = 0)

                #testgr[ckpt] = test_gradb

                for s in range(start,end+1):
                    if s>=len(self.trainset):
                        break
                    subset_trindices.append(s)
                    subset_pickindices.append(s)
                    cpv[s].append((ep,bt))


                subsetcp = torch.utils.data.Subset(self.trainset, subset_pickindices)
                trainsubloader = torch.utils.data.DataLoader(subsetcp, batch_size=1, shuffle=False, num_workers=2)

                
                for batch_idx, (inputs, targets) in enumerate(trainsubloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.squeeze(1)
                    ind = ind + 1
                    train_grad_single = self.helperobj.get_grad(inputs, targets, net).unsqueeze(0)
                    tempf = test_gradb*train_grad_single
                    temp = (tempf + 0.5*(tempf*tempf))/len(trainsubloader)  #Feature value for each datapoint
                    repeat[subset_pickindices[batch_idx]]+=1
                    res[subset_pickindices[batch_idx]] += (alpha*(torch.sum(temp).item()))/repeat[subset_pickindices[batch_idx]] #score
                    rescount = rescount + 1

        with open('./influence_scores_robust.npy', 'wb') as f:
            np.save(f, np.array(res))

        return res


    ############ If computed data values are not to be used, the existing ones are removed and calculated again using selected batches ##############

    def scorevalue(self):

        ########### If one has resumed, scores have to be recomputed. In order to avoid any eventuality, making retain_scores false explicitly############# 
        if self.resume:
            self.keeprem = False #If training is resumed from some point, data values/scores have to be recomputed
 
        if not self.keeprem:
            if os.path.exists('./influence_scores_robust.npy'):
                os.remove('./influence_scores_robust.npy')

            scores = self.score_trajbatch()

        else:
            if os.path.exists('influence_scores.npy'):
                scores = np.load('./influence_scores.npy')
            else:
                scores = None #retain_scores True even if no scores are kept recomputed
                
        return scores
