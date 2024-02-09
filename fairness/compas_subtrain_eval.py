
import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torchvision
from operator import itemgetter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import os
################# Creating training dataset class #################

class CompasData(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i], i



class Dataset(object):

    def __init__(self):

        compas_data = pd.read_csv('propublica_data_for_fairml.csv', na_values='?')
        scaler = StandardScaler()
        compas_data[['Number_of_Priors']] = scaler.fit_transform(compas_data[['Number_of_Priors']])
        X = compas_data.drop(['Two_yr_Recidivism'],axis=1).values
        y = compas_data['Two_yr_Recidivism'].values
        z = compas_data['Female'].values

        self.X_train, self.X_test, self.y_train, self.y_test, self.z_train, self.z_test = train_test_split(X, y,z, test_size = 0.2, random_state = 42)

        self.z_train = torch.FloatTensor(self.z_train).squeeze()
        self.z_test = torch.FloatTensor(self.z_test).squeeze()
        self.trainset = CompasData(self.X_train, self.y_train)
        self.testset = CompasData(self.X_test, self.y_test)


    def load_data(self):

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = 10, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=10, shuffle=False, num_workers=2)

        return trainloader, testloader, self.trainset, self.testset, self.X_train, self.y_train, self.X_test, self.y_test, self.z_train, self.z_test

import torch.nn as nn
import torch.nn.functional as F

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

import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
net = Network()
net = net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.25, weight_decay=0.025)

ds = Dataset()
trainloader, testloader, trainset, testset, X_train,y_train, X_test,y_test, z_train, z_test= ds.load_data()

print(len(trainloader))
print(len(testloader))

temp = np.load('influence_scores.npy')
values = []
for i in range(temp.shape[0]):
    if temp[i]!=0:
       values.append((temp[i],i))

sval = sorted(values,key=itemgetter(0),reverse=True)
subset_trindices = []
for v in sval:
    subset_trindices.append(v[1])

subset_trindices = subset_trindices[:2970] #60%

#Creating susbet using scores

X_train_use = X_train[subset_trindices]
y_train_use = y_train[subset_trindices]


def test(X_test, y_test, net):

	net.eval()

	correct = 0
	total   = 0
	end = len(X_test)

	class_correct = list(0. for i in range(2))
	class_total   = list(0. for i in range(2))
	with torch.no_grad():
    	
		for index in range(0,end,50):
			
			inputs, labels = X_test[index:min(index+50,end)], y_test[index:min(index+50,end)]
			inputs, labels = torch.from_numpy(inputs).float() , torch.from_numpy(labels)
			inputs, labels = inputs.to(device), labels.to(device)
        
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			_, pred = torch.max(outputs, 1)
			c = (pred == labels).squeeze()

			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	print('Accuracy : %d %%' % (100 * correct / total))

	return correct, total


#Training
end = len(X_train_use)
best_acc = 0
for epoch in range(20):
    running_loss = 0
    total_loss = 0
    for index in range(0,end,10):
        inputs, labels = X_train_use[index:min(index+10,end)], y_train_use[index:min(index+10,end)]
        inputs, labels = torch.from_numpy(inputs).float() , torch.from_numpy(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
    print('Epoch'+str(epoch))
    print(total_loss/(len(X_train)/10))
    correct, total = test(X_test, y_test, net)
    running_loss = 0
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
    if not os.path.isdir('checkpoint_sub'):
      os.mkdir('checkpoint_sub')
    torch.save(state, './checkpoint_sub/compas_f60.pth')
    print("Best accuracy")
    print(best_acc)


PATH = './checkpoint_sub/compas_f60.pth'


net = Network()
net.load_state_dict(torch.load(PATH)['net'])



def test_fairness(model_, X, y, s1):
    
    model_.eval()
    
    y_hat = model_(X)
    prediction = (torch.max(y_hat.data, 1)[1] > torch.tensor(0)).int()
    y = (y > 0.0).int()
    z_0_mask = (s1 == 0.0)

    z_1_mask = (s1 == 1.0)

    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))

    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)

    y_0 = int(torch.sum(y_0_mask))
    y_1 = int(torch.sum(y_1_mask))


    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)
 
    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1
        
    
    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))
    
    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1
    
    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1
    
    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1

    recall = Pr_y_hat_1_y_1
    precision = float(torch.sum((prediction == 1)[y_1_mask])) / (int(torch.sum(prediction == 1)) + 0.00001)
    
    y_hat_neq_y = float(torch.sum((prediction == y.int())))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    test_f1 = 2 * recall * precision / (recall+precision+0.00001)
    
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    min_eo_0 = min(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    max_eo_0 = max(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    min_eo_1 = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    max_eo_1 = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    
    DP = max(abs(Pr_y_hat_1_z_0 - Pr_y_hat_1), abs(Pr_y_hat_1_z_1 - Pr_y_hat_1))
    
    EO_Y_0 = max(abs(Pr_y_hat_1_y_0_z_0 - Pr_y_hat_1_y_0), abs(Pr_y_hat_1_y_0_z_1 - Pr_y_hat_1_y_0))
    EO_Y_1 = max(abs(Pr_y_hat_1_y_1_z_0 - Pr_y_hat_1_y_1), abs(Pr_y_hat_1_y_1_z_1 - Pr_y_hat_1_y_1))

    return {'DP_diff': DP, 'EqOdds_diff': max(EO_Y_0, EO_Y_1)}



metrics_test = test_fairness(net, torch.from_numpy(X_test).float(), torch.from_numpy(y_test), z_test)
print('Accuracy or 1-ER')
print(best_acc)
print('Fairness Metrics')
print(metrics_test)
