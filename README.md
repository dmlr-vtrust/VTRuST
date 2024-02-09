# VTRuST

This is the code repository for our paper 'VTruST : Controllable value function based subset selection for Data-Centric Trustworthy AI'.

# Getting Started

Run the following commands to setup the environment:

```
git clone https://github.com/SoumiDas/VTruST.git

cd VTruST

conda env create -f requirements.txt
```

After the environment gets installed,

```
conda activate vtrust
```

We provide the following instructions for both Fairness and Robustness on COMPAS and CIFAR-10 respectively.

# Current Setup for Fairness

<i>Dataset:</i> COMPAS

<i>Model:</i> 2 layer network

The default parameters are provided in ```config.json```. One can vary the parameters by running ```python config_create.py```

In order to obtain selected datapoints and their scores, from VTruST, run

```python experiment.py```

Train the selected set of datapoints by running

```python compas_subtrain_eval.py```

# Current Setup for Robustness

<i>Dataset:</i> CIFAR-10

<i>Model:</i> ResNet-18

Please download 'CIFAR-10-C' from https://zenodo.org/record/2535967 and store it under robustness/

The default parameters are provided in ```config.json```. One can vary the parameters by running ```python config_create.py```

In order to obtain selected datapoints and their scores, from VTruST, run

```python experiment.py```

Train the selected set of datapoints by running

```python train_eval_SA.py```

Test the model for robustness by running

```python test_robust.py```

# Code Structure

The different modules in which this code repository has been organised is:

1. Dataset (dataset.py) - One can set up their own dataset as needed.

2. Model (model.py) - Different models can be added here.

3. Datapoint selection during training (run.py) - Selects datapoints and provides scores.

4. Value function definition and VTruST algorithm (helper.py) - Different other value functions can be incorporated in helper.py/valuefunc_cb().

6. Subset training (compas_subtrain_eval.py or train_eval_SA.py) - Returns the trained model using the subset and some evaluated metrics. 


# Using VTruST for other datasets/models

<b> Change in dataset </b>

For changing datasets, one needs to modify the definitions of the classes ```Dataset_train``` and ```Dataset_test``` in ```dataset.py``` as per requirement.

<b> Change in model </b>

1. Current ResNet implementation is a modified form of the original ResNet suited for images of lower dimension like that of CIFAR10. If we want to use the different architectures of this version of ResNet, we can follow the comments in the code and simply change the self.block and self.layers in the _init_ of ```class Model``` in ```model.py```.

2. If we want to use the original ResNet18 architecture or any other existing architectures from torchvision.models, include the line 
```model = models.resnet18(pretrained=True) #use torch models``` in model.py.

3. ```get_grad()``` function in ```helper.py``` will have the last layer name changed as per the architecture. For the current ResNet18 model, the last layer name is ```self.linear``` ; hence for extracting the last layer output, line number 70 in ```helper.py``` is ```params = model.linear.weight``` and similarly for fairness, line number x in ```helper.py``` is ```params = model.layer3.weight```.

For the inbuilt torch model, the last layer name for ResNet implementations, is ```self.fc``` ; hence for extracting the last layer output, the line will turn to ```params = model.fc.weight```.

Overall, while using any other torch models,  one needs to definitely know the last layer name for computing gradients.

<b> Change in training parameters </b>

1. For learning rate, optimizer, one needs to change them from ```def _init_ ```of ```class BatchSel()``` in ```run.py```; 

2. For type of loss like CrossEntropy, one needs to specify them in ```def _init_ ```of ```class BatchSel()``` in ```run.py``` and ```def _init_ ```of ```class HelperFunc()``` in ```helper.py```.
