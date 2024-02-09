from subsetpack.dataset import Dataset
from subsetpack.model import Model
from subsetpack.run import BatchSel
from subsetpack.helper import HelperFunc
import config_create
import os
import json

def main():

	############ Run the config file to create a dictionary ##########	
	with open("config.json", "r") as fp:
		confdata = json.load(fp) #holds the various configurable parameters needed ahead

	########### Defining dataset class for loading the required data #############
	dataobj = Dataset(confdata)
	trainloader, testloader, trainset, testset, testloader_s = dataobj.load_data()

	########### Defining model class for loading the model architecture ###########
	modelobj = Model()
	model = modelobj.netmodel()

	########### Trains the model on the dataset and valuates datapoints using VTruST
	helpobj = HelperFunc(trainloader,testloader,model,confdata)
	batchobj = BatchSel(trainset,trainloader,testset,testloader,model,helpobj,confdata)
	batchobj.fit()

if __name__=='__main__':
	main()
