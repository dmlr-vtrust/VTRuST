######### This class will compute values/importance weights for all data points from the selected batches using VTruST #############
######### Features of the datapoints from a particular selected batch are computed using the respective model parameter and the learned coefficients ###########
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import time
import pickle


class DataValue(object):

    def __init__(self,train,testset,test,model,helperobj,confdata):

        self.trainset = train
        self.testset = testset
        self.testloaderb = test
        self.model = model
        self.confdata = confdata
        self.rootpath = confdata['root_dir']
        self.bsize = confdata['trainbatch']
        self.keeprem = confdata['retain_scores']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume = confdata['resume']
        self.helperobj = helperobj
        self.confdata = confdata


    def score_trajbatch(self):

        cpind = []
        alphaval = []
        indcp = np.load(self.rootpath+'trajp_value_indices.npy') #validation set

        for ind in range(indcp.shape[0]):
            cpind.append((indcp[ind][1],indcp[ind][2]))
            alphaval.append(indcp[ind][0]) #coefficient values
       
        subset_trindices = []
        btlist = set()
        batches = os.listdir(self.rootpath+'trajp/')
        #print(len(cps))
        res = [ 0 for i in range(len(self.trainset)) ]
        cpv = [ [] for i in range(len(self.trainset)) ]
        repeat = [ 0 for i in range(len(self.trainset)) ]
        fea = {}
        testgr = {}
        dict_contrib = {}
        ind = 0


        #print(len(cps))
        wt=0
        for batchf in batches:
                net = torch.load(self.rootpath+'trajp/'+batchf)
                subset_pickindices =[]
                ep = int(batchf.split('_')[1].split('_')[0])
                bt = int(batchf.split('_')[3].split('.')[0])
                alpha = alphaval[cpind.index((ep,bt))]
                start = ((self.bsize-1)*bt)+bt
                end = start + (self.bsize-1)
                test_gradb = None
                for testi, data in enumerate(self.testloaderb,0):
                    inputs, targets = data
                    inputs, targets = inputs.float().to(self.device), targets.to(self.device)

                    predn = net(inputs)
                    lossn = self.helperobj.calc_loss_test(predn, targets) #loss function
                    lossy0z0_c = self.helperobj.calc_loss_test(net(inputs[:15].float().to(self.device)), targets[:15].to(self.device))
                    lossy0z1_c = self.helperobj.calc_loss_test(net(inputs[15:30].float().to(self.device)), targets[15:30].to(self.device))
                    lossy1z0_c = self.helperobj.calc_loss_test(net(inputs[30:45].float().to(self.device)), targets[30:45].to(self.device))
                    lossy1z1_c = self.helperobj.calc_loss_test(net(inputs[45:60].float().to(self.device)), targets[45:60].to(self.device))

                    fairloss_past = max(abs(lossy0z0_c-lossy0z1_c),abs(lossy1z0_c-lossy1z1_c)) #eo disparity
                    cgradloss = wt*lossn + (1-wt)*fairloss_past #combined loss-disparity ; wt = \lambda

                    if test_gradb is None:
                        test_gradb = self.helperobj.get_grad_test(cgradloss,net)
                        test_gradb = test_gradb.unsqueeze(0)

                    else:
                        test_gradb = torch.cat((test_gradb, self.helperobj.get_grad_test(cgradloss, net).unsqueeze(0)), axis = 0)

                for s in range(start,end+1):
                    if s>=len(self.trainset):
                         break
                    subset_trindices.append(s)
                    subset_pickindices.append(s)
                    cpv[s].append((ep,bt))


                subsetcp = torch.utils.data.Subset(self.trainset, subset_pickindices)
                trainsubloader = torch.utils.data.DataLoader(subsetcp, batch_size=1, shuffle=False, num_workers=2) #instances of the selected batch

                
                for batch_idx, (inputs, targets) in enumerate(trainsubloader):
                    inputs, targets = inputs.float().to(self.device), targets.to(self.device)
                    ind = ind + 1
                    train_grad_single = self.helperobj.get_grad(inputs, targets, net).unsqueeze(0)
                    tempf = test_gradb*train_grad_single
                    temp = (tempf + 0.5*(tempf*tempf))/len(trainsubloader)  #Feature value for each datapoint
                    repeat[subset_pickindices[batch_idx]]+=1
                    res[subset_pickindices[batch_idx]] += (alpha*(torch.sum(temp).item()))/repeat[subset_pickindices[batch_idx]] #score

        with open('./influence_scores.npy', 'wb') as f:
            np.save(f, np.array(res))

        return res


    ############ If computed data values are not to be used, the existing ones are removed and calculated again using selected batches ##############

    def scorevalue(self):

        ########### If one has resumed, scores have to be recomputed. In order to avoid any eventuality, making retain_scores false explicitly############# 
        if self.resume:
            self.keeprem = False #If training is resumed from some point, data values/scores have to be recomputed
 
        if not self.keeprem:
            if os.path.exists('./influence_scores.npy'):
                os.remove('./influence_scores.npy')

            #print("Score compute")
            scores = self.score_trajbatch()

        else:
            if os.path.exists('influence_scores.npy'):
                scores = np.load('./influence_scores.npy')
            else:
                scores = None #retain_scores True even if no scores are kept recomputed
                
        return scores
