from subsetpack.dataset import Dataset
from subsetpack.model import Model
from subsetpack.run import BatchSel
from subsetpack.helper import HelperFunc
import config_create
import os
import json

def main():

	############ Run the config file to create a dictionary ##########
	#os.system('python config_create.py')
	
	with open("config.json", "r") as fp:
		confdata = json.load(fp) #holds the various configurable parameters needed ahead

	########### Defining dataset class for loading the required data #############
	dataobj = Dataset(confdata)
	trainloader, testsubloader,trainset, testset, subset_test, noise_loader, testloader_s,noise2_loader, testloader_s2 = dataobj.load_data()

	########### Defining model class for loading the model architecture ###########
	modelobj = Model()
	model = modelobj.ResNet18()

	########### Trains the model on the dataset and selects batches along with their importance weights using VTruST
	helpobj = HelperFunc(trainloader,testsubloader,testset,noise_loader,noise2_loader,model,confdata)
	trajobj = BatchSel(trainset,trainloader,testset,subset_test,testsubloader,noise_loader,noise2_loader,model,helpobj,confdata)
	trajobj.fit()

if __name__=='__main__':
	main()
