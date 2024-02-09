import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
#from subsetpack.noise import gaussian_noise,shot_noise,gaussian_blur,impulse_noise,speckle_noise
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import pickle
import random
import os
import argparse
import time
import copy
from numpy import linalg as LA

class HelperFunc(object):

	def __init__(self,train,test,testset,noiseloader,noise2loader,model,confdata):

		self.num_entry = 0
		self.epochs = confdata['epochs']
		self.train_loader=train
		self.test_loader=test
		self.test_set = testset
		self.noise_loader = noiseloader
		self.noise2_loader = noise2loader
		self.model = model
		self.dv2o = np.zeros((2*len(self.test_loader),1))	
		self.cv_val = np.zeros((2*len(self.test_loader),len(self.train_loader)))		
		self.rootpath = confdata['root_dir']
		self.resume = confdata['resume']
		self.csel = confdata['csel']
		#self.csel_epoch_interval = confdata['num_freqep']
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.selected = [[] for i in range(confdata['epochs'])]
		self.numtp = confdata['num_trajpoint']
		self.confdata = confdata
		self.criterion = nn.CrossEntropyLoss(reduction='mean')
		self.criterion_indiv = nn.CrossEntropyLoss(reduction='none')
		self.lastbatch = len(self.train_loader)-1
		self.softprob = nn.Softmax(dim=1)
		self.net1 = self.model
		self.model.to(self.device)
		self.net1.to(self.device)
		if not os.path.exists(self.rootpath):
			os.mkdir(self.rootpath)
		if not os.path.exists(self.rootpath+'trajp/'):
			os.mkdir(self.rootpath+'trajp/')
		if not os.path.exists(self.rootpath+'misc/'):
			os.mkdir(self.rootpath+'misc/')

		self.lr = 0.1
		self.So = []
		self.Soe = []
		self.reslisto = []
		self.Io = []
		self.Eo = []
		self.dvno = []
		self.esno = []

	def calc_loss(self,y, t):
		loss = self.criterion(y,t)
		return loss

	def calc_loss_test(self,y,y_a,t):
		loss = 0*self.criterion(y,t)+1.0*self.criterion(y_a,t)
		return loss

	def get_grad(self,input, target, model):

		model.eval()
		z, t = input.to(self.device), target.to(self.device)
		y = model(z)
		loss = self.calc_loss(y, t)
		# Compute sum of gradients from model parameters to loss
		params = model.linear.weight
		result = list(grad(loss, params))[0].detach()

		return result

	def get_grad_test(self,loss, model):

		model.eval()
		# Compute sum of gradients from model parameters to loss
		params = model.linear.weight
		result = list(grad(loss, params))[0].detach()
		del loss
		return result


	# Value function callback that computes value function delta and aggregate estimate C
	def valuefunc_cb(self,epoch,batch_idx,inputs,targets,lossval,cz,czs,initmodel,model):

		param = copy.deepcopy(model)	
		if batch_idx==0: #first time computing value function
			if os.path.exists(self.rootpath+'misc/lastretain.pth'):
				self.net1 = torch.load(self.rootpath+'misc/lastretain.pth')
				if self.resume and self.num_entry==0:
					self.batchgrad = self.get_grad(inputs, targets, self.net1)
			else:
				self.net1 = copy.deepcopy(initmodel)
				self.batchgrad = self.get_grad(inputs, targets, initmodel)
		#else:
		model.eval()
		self.net1.eval()
		for testi,(data,datasub) in enumerate(zip(self.noise_loader,self.test_loader),0):
			delv = []
			delvc = []
			adv_images = []
			inputs_test, targets_test = data
			inputs_sub_test, targets_sub_test = datasub

			inputs_test, targets_test = inputs_test.to(self.device), targets_test.to(self.device)
			inputs_sub_test, targets_sub_test = inputs_sub_test.to(self.device), targets_sub_test.to(self.device)

			predn = model(inputs_test)
			predn_a = model(inputs_sub_test)

			predn1 = self.net1(inputs_test)
			predn1_a = self.net1(inputs_sub_test)

			self.batchgrad = self.get_grad(inputs, targets, self.net1)
			lossn = self.calc_loss_test(predn_a, predn, targets_test) #combined loss value function using current model parameters
			lossn1 = self.calc_loss_test(predn1_a,predn1,targets_test) #combined loss value function using past model parameters
			testgrad = self.get_grad_test(lossn1,self.net1)
			
			lossval[testi] = lossval[testi]+(lossn.item()-lossn1.item())
			cmat = torch.sum(testgrad*self.batchgrad).item()
			#Second order approximation
			czs[testi].append(cmat+0.5*(cmat*cmat))
			cz[testi].append(cmat)


		for testi,(data,datasub) in enumerate(zip(self.noise2_loader,self.test_loader),0):
			delv = []
			delvc = []
			adv_images = []
			inputs_test, targets_test = data
			inputs_sub_test, targets_sub_test = datasub

			inputs_test, targets_test = inputs_test.to(self.device), targets_test.to(self.device)
			inputs_sub_test, targets_sub_test = inputs_sub_test.to(self.device), targets_sub_test.to(self.device)

			predn = model(inputs_test)
			predn_a = model(inputs_sub_test)

			predn1 = self.net1(inputs_test)
			predn1_a = self.net1(inputs_sub_test)

			self.batchgrad = self.get_grad(inputs, targets, self.net1)
			lossn = self.calc_loss_test(predn_a, predn, targets_test)
			lossn1 = self.calc_loss_test(predn1_a,predn1,targets_test)
			testgrad = self.get_grad_test(lossn1,self.net1)
			lossval[testi+len(self.test_loader)] = lossval[testi+len(self.test_loader)]+(lossn.item()-lossn1.item()) 

			cmat = torch.sum(testgrad*self.batchgrad).item()
			#Second order approximation
			czs[testi+len(self.test_loader)].append(cmat+0.5*(cmat*cmat))
			cz[testi+len(self.test_loader)].append(cmat)

		model.train()              
		self.net1 = copy.deepcopy(model)
			
		if batch_idx==self.lastbatch: #last batch when checksel is run
			param.load_state_dict(self.net1.state_dict())
			torch.save(param,self.rootpath+'misc/lastretain.pth')

		return lossval,cz,czs


	#ALgorithm 1 for saving selected batches along with importance weights/values
	#Calls Algorithm 2: DataReplace for substituting the already selected batches with better ones giving better approximation
	def vtrust_cb(self,dv2w,cvs2w,model,epoch,netv):
		# Number of to-be selected checkpoints

		if self.csel is True and epoch==self.epochs-1:
                #if self.csel is True and epoch%csel_epoch_interval==0:
			ko = self.numtp
		
			paramc = copy.deepcopy(model)
			dv2w = np.array(dv2w)
			cvs2w = np.array(cvs2w)

			if self.resume and self.num_entry==0:
				alphao = np.load(self.rootpath+'misc/alpha_val.npy')
				esto = np.load(self.rootpath+'misc/estimate.npy')
				self.dv2o = np.load(self.rootpath+'misc/valuefunc.npy')
				self.cv_val = np.load(self.rootpath+'misc/est_valuefunc.npy')
				Soval = np.load(self.rootpath+'misc/cpind_val.npy')
				for elem in range(Soval.shape[0]):
					self.So.append((Soval[elem][0],Soval[elem][1])) 
				self.Sco = np.load(self.rootpath+'misc/estimate_grad.npy')
  

			self.dv2o = self.dv2o + dv2w.reshape(-1,1) #value function
			self.cv_val = self.cv_val + cvs2w

			cv2o = normalize(self.cv_val, axis = 0) #Normalise
			
			for batch in range(cv2o.shape[1]):
				if len(self.So) < ko:
					self.So.append((epoch,batch)) 
					self.Soe.append((epoch,batch))
					if batch == 0:
						self.Sco = -self.lr*(cv2o[:,batch]) # Adding to S
					else:
						self.Sco = np.vstack([self.Sco,-self.lr*(cv2o[:,batch])]) # Adding to S
				else:
					####### DataReplace module ######
					alphao,esto,self.Sco,reso,self.So,self.Soe= self.datareplace(self.dv2o,esto,cv2o,alphao,epoch,batch)


				if len(self.So)>1:
					alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(self.dv2o))[0] # Update alpha
					esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) #Update eta

			self.dvno.append(LA.norm(self.dv2o))
			self.esno.append(LA.norm(esto))
			self.reslisto.append(LA.norm(reso)/LA.norm(self.dv2o))
			self.Io.append(self.dv2o)
			self.Eo.append(esto)

			for sl in range(len(self.So)):
				ep,bt = self.So[sl]
				ep1,bt1 = self.Soe[sl]
				if ep == epoch:
					self.selected[ep].append(bt)
				if ep1==epoch:
					paramc.load_state_dict(netv[bt1])
					#saving selected batches
					torch.save(paramc,self.rootpath+'trajp/epoch_'+str(ep1)+'_batch_'+str(bt1)+'.pth')


			if epoch > 0 or len(os.listdir(self.rootpath+'trajp/'))>0:
				files = os.listdir(self.rootpath+'trajp/')
				for f in files:
					ep,bt = int(f.split('_')[1]),int(f.split('_')[3].split('.')[0])
					if (ep,bt) not in self.Soe:
						#remove existing unneccesary batches
						os.remove(self.rootpath+'trajp/epoch_'+str(ep)+'_batch_'+str(bt)+'.pth')

			np.save(self.rootpath+'misc/alpha_val.npy',alphao)
			np.save(self.rootpath+'misc/valuefunc.npy',np.asarray(self.dv2o))
			np.save(self.rootpath+'misc/est_valuefunc.npy',np.asarray(self.cv_val))
			np.save(self.rootpath+'misc/estimate.npy',np.asarray(esto))
			np.save(self.rootpath+'misc/estimate_grad.npy',np.asarray(self.Sco))
			np.save(self.rootpath+'misc/cpind_val.npy',self.Soe)
			np.save(self.rootpath+'misc/cumulloss_val.npy',np.asarray(self.Io))
			np.save(self.rootpath+'misc/estimate_val.npy',np.asarray(self.Eo))
			self.num_entry = self.num_entry + 1
			cpv = np.load(self.rootpath+'misc/cpind_val.npy')
			val_ind = []
			for elem in range(cpv.shape[0]):
				val_ind.append((alphao[elem][0],cpv[elem][0],cpv[elem][1]))
			np.save(self.rootpath+'trajp_value_indices.npy',np.asarray(val_ind))
			print(epoch)
			print(self.selected[epoch])

		else:
			pass


	#Algorithm 2 replaces existing batches with new ones that better approximate the value function
	def datareplace(self,dv2o,esto,cv2o,alphao,epoch,batch):

		reso = dv2o - esto #residual vector
		betav = -np.inf
		ind = None
		exbatch = []

		for s in range(self.Sco.shape[0]):
			self.Sco[s]=-self.lr*cv2o[:,self.So[s][1]]
			self.So[s]= (epoch,self.So[s][1])
			exbatch.append(self.So[s][1])

		alphao = lstsq(self.Sco.dot(self.Sco.T),self.Sco.dot(dv2o))[0] # Update alpha
		esto = np.dot(self.Sco.T,alphao.reshape(-1,1)) # Update eta
		nprojo = np.dot(-self.lr*cv2o[:,batch].reshape(-1,1).T,reso) #proj of X_i

		if batch not in exbatch:
			for s in range(self.Sco.shape[0]):
				eprojo = np.dot(self.Sco[s].reshape(-1,1).T,reso) #proj of X_j
				if abs(nprojo) > abs(eprojo) and alphao[s]<0 and alphao[s]+abs(eprojo)>betav:
					betav = alphao[s]+abs(eprojo)
					ind = s

			if ind!=None:	
				self.Sco[ind] = -self.lr*cv2o[:,batch]
				self.So[ind] = (epoch,batch)
				self.Soe[ind] = (epoch,batch)

		return alphao,esto,self.Sco,reso,self.So,self.Soe
