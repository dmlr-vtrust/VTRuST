#Defining dataset class

import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

################# Creating training dataset class #################

class CompasData(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

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

        compas_data = pd.read_csv('propublica_data_for_fairml.csv', na_values='?')
        scaler = StandardScaler()
        compas_data[['Number_of_Priors']] = scaler.fit_transform(compas_data[['Number_of_Priors']])
        X = compas_data.drop(['Two_yr_Recidivism'],axis=1).values
        y = compas_data['Two_yr_Recidivism'].values
        z = compas_data['Female'].values

        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size = 0.2, random_state = 0) #4937, 1235

        self.trainset = CompasData(X_train, y_train)
        self.testset = CompasData(X_test, y_test)
        self.confdata = confdata


    def load_data(self):

        sampler = RandomBatchSampler(torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SequentialSampler(self.trainset),batch_size=self.confdata['trainbatch'],drop_last=False))
        trainloader = torch.utils.data.DataLoader(self.trainset,batch_sampler=sampler)
        subset_indices = list(np.load('compas_idx.npy'))  #validation set
        subset_test = torch.utils.data.Subset(self.testset, subset_indices)
        testloader = torch.utils.data.DataLoader(subset_test, batch_size=self.confdata['testbatch'], shuffle=False, num_workers=2)
        testloader_s = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=2)

        return trainloader, testloader, self.trainset, subset_test, testloader_s
