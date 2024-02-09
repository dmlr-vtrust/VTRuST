import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
from torch.autograd import grad
from subsetpack.dataset import RandomBatchSampler
from subsetpack.scoreval_datasel import DataValue
from subsetpack.model import Model
from subsetpack.helper import HelperFunc
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random
import os
import cv2
import argparse
import time
import copy
from numpy import linalg as LA
import scipy.stats
from subsetpack.helper import HelperFunc
#from subsetpack.noise import gaussian_noise,shot_noise,gaussian_blur,impulse_noise,speckle_noise


class BatchSel(object):

	def __init__(self,trainset,train,testset,subtest,test,noiseloader,noise2loader,model,helpobj,confdata):

		self.trainset = trainset
		self.train_loader=train
		self.testset = testset
		self.subset_test = subtest
		self.test_loader=test
		self.noise_loader = noiseloader
		self.noise2_loader = noise2loader
		self.model = model
		self.batchl = [i for i in range(0,len(self.train_loader))]
		self.rootpath = confdata['root_dir']
		self.resume = confdata['resume']
		self.epochs = confdata['epochs']
		self.numtp = confdata['num_trajpoint']
		self.step_in_epoch = 0
		self.csel = confdata['csel']
                #self.csel_epoch_interval = confdata['num_freqep']
		self.helpobj = helpobj
		self.confdata = confdata
		self.count = 0
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.criterion = nn.CrossEntropyLoss()
		if self.resume:
			checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
			self.model.load_state_dict(checkpointpath['model'])
			self.start_epoch = checkpointpath['epoch']+1
		else:
			self.start_epoch = 0
		self.lr = 0.1
		self.optimizer = optim.SGD(self.model.parameters(), self.lr)
		self.net1 = self.model
		self.initmodel = self.model
		
		self.model.to(self.device)
		self.net1.to(self.device)
		if not os.path.exists(self.rootpath):
			os.mkdir(self.rootpath)
		if not os.path.exists(self.rootpath+'checkpoint/'):
			os.mkdir(self.rootpath+'checkpoint/')
		if not(self.resume) and os.path.exists(self.rootpath+'misc/lastretain.pth'):
			os.remove(self.rootpath+'misc/lastretain.pth')



	def initialize(self):

		netv = {}
		for ix in self.batchl:
			netv[ix] = {}		

		lossval = [0 for i in range(2*len(self.test_loader))]
		cz = [[] for i in range(2*len(self.test_loader))]
		czs = [[] for i in range(2*len(self.test_loader))]

		return netv, lossval,cz,czs

	def savemodel(self,netv=None,batchid=None,epoch=None,unif=False):

		if netv!=None:
			netv[batchid] = self.model.state_dict()
			self.count = self.count + 1

		if unif==True:
			torch.save(self.model.state_dict(), self.rootpath+'checkpoint/epoch_'+str(epoch)+'.pth')


	########## Trains the model on a dataset; runs VTruST at an epoch; stores the miscellaneous results and also the BATCH indices with their weights ###########
	def fit(self):


		self.helpobj = HelperFunc(self.train_loader,self.test_loader,self.testset,self.noise_loader,self.noise2_loader,self.model,self.confdata)

		for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
		#for epoch in range(1):

			eptime = time.time()
			self.count = 0
			trloss = 0
			netv, lossval, cz, czs = self.initialize()
			self.model.train()

			print("Epoch")
			print(epoch)
			
			for batch_idx, (inputs, targets) in enumerate(self.train_loader):

				start = time.time()
				self.model.train()
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				print(batch_idx)
				inputs = inputs.squeeze(1)		
				if self.csel is True and epoch==self.epochs-1:	
				#if self.csel is True and epoch%csel_epoch_interval==0
					lossval,cz,czs = self.helpobj.valuefunc_cb(epoch,batch_idx,inputs.to(self.device),targets,lossval,cz,czs,self.initmodel,self.model)	

				self.model.train()
				self.optimizer.zero_grad()
				outputs = self.model(inputs.to(self.device))
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizer.step()

				if self.csel is True and epoch==self.epochs-1:
					self.savemodel(netv=netv,batchid=batch_idx,epoch=epoch) #saving required model parameters

				self.savemodel(epoch=epoch)
				self.step_in_epoch+=1

				trloss = trloss + loss.item()

			print("Epoch time")
			print(time.time()-eptime)
			print('Epoch '+str(epoch)+', Loss '+str(trloss/len(self.train_loader)))			
			#Callback function to run Algorithm 1: VTruST
			if epoch==self.epochs-1:
			#if self.csel is True and epoch%csel_epoch_interval==0
				self.helpobj.vtrust_cb(lossval,czs,self.model,epoch,netv)
		dvalueobj = DataValue(self.trainset,self.testset,self.test_loader,self.noise_loader,self.noise2_loader,self.model,self.helpobj,self.confdata)
		scores = dvalueobj.scorevalue()
