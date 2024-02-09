#import yaml
import json

params = {}

#Flags for Trajectory Selection
params['csel'] = True #True for executing VTruST during training
params['resume'] = False #True for resuming training where it was left off

#Flags for Data Valuation
params['retain_scores'] = False #False for new/replaced computation of data values, True for using existing ones

#Hyperparameters during Trajectory Selection
params['trainbatch'] = 100 #Training data batch size
params['testbatch'] = 60 #Test data batch size
params['epochs'] = 20 #Training epochs
params['num_trajpoint'] = 32 #Number of to-be selected batches
params['num_freqep'] = 40 #Optional ; Frequency of epochs at which VTruST will be executed

#Path
params['root_dir'] = "./main/"

with open("config.json", "w") as outfile:
    json.dump(params, outfile)
