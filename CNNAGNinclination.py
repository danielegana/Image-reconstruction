#%%

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset,random_split
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
import torch.nn.init as init
import re
import os
from sklearn.metrics import mean_squared_error


pythondir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/"
filedir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/agndisk/agndisk/files/"
binarydir="/Users/danielegana/Dropbox (PI)/ML/code/AGN/agndisk/agndisk/"

os.chdir(pythondir)
import subprocess
from helpers import plot



#%%
lendata=100
inclination = np.random.rand(lendata)*np.pi/2
os.chdir(binarydir)

os.makedirs(filedir+"inputs", exist_ok=True)
os.makedirs(filedir+"outputvis", exist_ok=True)
os.makedirs(filedir+"outputint", exist_ok=True)

#%%
for x,y in enumerate(inclination):
    with open(filedir+"inputs/input" + str(x), 'w') as file:
    # Write each number to the file
           file.write(str(inclination[x]) + '\n')
           bashcommand="./ssdisk"+" -run "+str(x)+" -i "+str(inclination[x])
           os.system(bashcommand)

#%%

def imagenumber(imagefile):
  return int(re.findall(r'\d+', imagefile)[0])

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = sorted([img for img in os.listdir(root_dir+"/outputint") if not img.startswith('.')],key=imagenumber)
        self.params = sorted([params for params in os.listdir(root_dir+"/inputs") if not params.startswith('.')],key=imagenumber)
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,"outputint" ,self.images[idx])
        image = np.genfromtxt(image_name,delimiter=" ")
        param_name=os.path.join(self.root_dir,"inputs" , self.params[idx])
        params = np.genfromtxt(param_name,delimiter=" ")
        # Convert image to tensor and normalize
        feature = torch.tensor(image, dtype=torch.float) 
        label = torch.tensor(params, dtype=torch.float)
        return feature,label
    
class visDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = sorted([img for img in os.listdir(root_dir+"/outputvis") if not img.startswith('.')],key=imagenumber)
        self.params = sorted([params for params in os.listdir(root_dir+"/inputs") if not params.startswith('.')],key=imagenumber)
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,"outputvis" ,self.images[idx])
        image = np.genfromtxt(image_name,delimiter=" ")
        param_name=os.path.join(self.root_dir,"inputs" , self.params[idx])
        params = np.genfromtxt(param_name,delimiter=" ")
        # Convert image to tensor and normalize
        feature = torch.tensor(image, dtype=torch.float) 
        label = torch.tensor(params, dtype=torch.float)
        return feature,label
#%%
agndatavis=visDataset(filedir)
agndataint=ImageDataset(filedir)
visflag=1

if visflag==1:
    agndata=agndatavis
else:
    agndata=agndataint


train_size = int(0.8 * len(agndata))
val_size = len(agndata) - train_size

#train_dataset, test_dataset = random_split(agndata, [train_size, val_size])
train_dataset = Subset(agndata, range(0,train_size))
test_dataset = Subset(agndata, range(train_size,len(agndata)))


batch_size=20
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)


#%%
train_images,train_labels=[torch.squeeze(x.view(x.size(0)*x.size(1), -1),1).numpy() for x, y in train_dataset], [y for x, y in train_dataset]
train_images=np.stack(train_images)
train_labels=np.stack(train_labels)

#%%

test_images,test_labels=[torch.squeeze(x.view(x.size(0)*x.size(1), -1),1).numpy() for x, y in test_dataset], [y for x, y in test_dataset]
test_images=np.stack(test_images)
test_labels=np.stack(test_labels)

### A DT
# %%
model=tree.DecisionTreeRegressor()
model.fit(train_images,train_labels)

#%% Benchmark is the average inclination

y_pred = np.ones(len(test_dataset))*np.pi/4
mse = mean_squared_error(test_labels, y_pred)
print("MSE Benchmark: ",mse)


# %% Now compute the mse for the test images
y_pred = model.predict(test_images)
frac_error=model.predict(test_images)/test_labels
mse = mean_squared_error(test_labels, y_pred)
print("MSE DT: ",mse)
print("Target inclination (deg): ",test_labels/(np.pi/180))
print("Fractional Error: ",frac_error)

# %%

### PLOTS
# %%
plt.figure(figsize=(20, 15))
plt.imshow(agndatavis[0][0][0:10,0:10])
plt.figure(figsize=(20, 15))
plt.imshow(agndataint[0][0])

# %%
plt.imshow(agndataint[0][0])
# %%
plt.figure(figsize=(20, 15))
plt.imshow(agndata[0][0][0:10,0:10])

# %%
