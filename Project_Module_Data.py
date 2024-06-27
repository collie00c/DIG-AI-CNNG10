########################################################################################################################
# Aufgabe:
#
# Bereitstellung aller notwendigen Daten für die Ausführung des Projektes
# wie Konstante, Klassen für Dataset und Modell, Hyperparameter 
#
# Anmerkungen:
#    NICHT AUSFÜHRBAR
#
#    Festlegung der Modell-Architektur durch die Hyperparameter: 
#           Die Dimensionen der Tupel, welche die Schichten definieren, müssen zu NUM_CONVLAYER und NUM_LINLAYER passen!
#           NUM_CONVLAYER und NUM_LINLAYER müssen >= 2 sein
#    
#########################################################################################################################



import h5py
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda

from torch.utils.data import Dataset

#########################################################
# Constants (DO NOT CHANGE):

NUM_CLASSES=10
FRAC = {
        'TRAIN': 0.7,
        'VAL':   0.2,  
        'TEST':  0.1 
       }

#########################################################
# Parameter:

SIZE_IMG=64
NUM_EPOCHS=16

#####################################################################################################################
# Hyperparameter und Definition der CNN Architektur
#   Wichtige Anmerkungen :
#   Die Dimensionen der Tupel, welche die Schichten definieren, müssen zu NUM_CONVLAYER und NUM_LINLAYER passen!
#   NUM_CONVLAYER und NUM_LINLAYER müssen >= 2 sein

NUM_FILTER=int(SIZE_IMG/2)   
HYP = {
       'SIZE_BATCH'    : 64,
       'SIZE_IMG'      : SIZE_IMG,  
       'LEARN_RATE'    : 0.001,             
       'NUM_FILTER'    : int(SIZE_IMG),         
       'SIZE_FILTER'   : (3,2),                       # Filtergrösse 
       'OUT_CHANNELS'  : (NUM_FILTER, NUM_FILTER*2),  # Ausgangs-Kanäle (Filter) 
       'POOLING'       : (1,1),
       'RATE_DROPOUT'  : 0,                         
       'NEURONS'       : (128,NUM_CLASSES),      # Neuronen (Ausgang)

       'NUM_CONVLAYER' : 2,                      # Anzahl der Conv layer, die zu berücksichtigen sind (Minimum= 2 !)
       'NUM_LINLAYER'  : 2                       # Anzahl der Lin layer, die zu berücksichtigen sind  (Minimum = 2 !)
       }

# Hyperparameter als string für die Plot-Ausgaben:
HYP_STRING=f"batchsz={HYP['SIZE_BATCH']}, imgsz={HYP['SIZE_IMG']}, lr={HYP['LEARN_RATE']}, filter_out={HYP['OUT_CHANNELS']}, filtersz={HYP['SIZE_FILTER']}, pooling={HYP['POOLING']}, dropout={HYP['RATE_DROPOUT']}, neurons={HYP['NEURONS']}"
#####################################################################################################################

def one_hot_encoding(y):
    return nn.functional.one_hot(y.to(torch.int64), num_classes=NUM_CLASSES).float()


transform_G10 = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))     
])


''' # OK funktioniert, kann aber nicht gespeichert werden, daher ausgelagert in Funktion one_hot_encoding !
# One-hot encoding   
target_transform_G10 = transforms.Compose([ 
    Lambda(lambda y: (nn.functional.one_hot(y.to(torch.int64), num_classes=NUM_CLASSES)).float())         
])
'''

target_transform_G10 = transforms.Compose([ 
   one_hot_encoding             
])

transform_G10_R = transforms.Compose([ 
    transforms.Resize((HYP['SIZE_IMG'], HYP['SIZE_IMG'])),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))     
])


###########################################################################
# DataSet-Klasse für den Galaxy-Datensatz:

class GalaxyDataset(Dataset):
    def __init__(self, h5_file, transform=None, target_transform=None):
        self.h5_file = h5_file
        self.transform = transform
        self.target_transform = target_transform

        with h5py.File(h5_file, 'r') as F:
            self.images = np.array(F['images'][()])     # torch.tensor(F['images'][()])
            self.labels = torch.tensor(F['ans'][()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        # Konvertiere das NumPy-Array in ein PIL-Bild
        image_pil = Image.fromarray(image)
        
        if self.transform:
            image_pil = self.transform(image_pil)
        if self.target_transform:
            label = self.target_transform(label)

        return image_pil, label
    
    def __getlabels(self):
        return self.labels

#####################################################################################

def Get_mapsize (mapsize,sizefilter,padding,stride):
    mapsize_new=((mapsize-sizefilter+2*padding)/stride)+1
    return mapsize_new


#####################################################################################
# CNN Modell  

class CNN_G10(nn.Module):
    #def __init__(self, num_classes=HYP['NUM_CLASSES'], imgsize=HYP['SIZE_IMG']):
    def __init__(self):    
        
        super(CNN_G10, self).__init__()
        self.flatten = nn.Flatten()

        self.mapsize=SIZE_IMG                 # Initiale Map-Size

        # Convolutional-Layer
        self.conv = nn.ModuleList()
        for i in range(HYP['NUM_CONVLAYER']):
           if i==0:
               in_filters=3   # 3 Farbkanäle
           else:
               in_filters=HYP['OUT_CHANNELS'][i-1]    

           self.conv.append (nn.Conv2d(in_channels=in_filters, out_channels=HYP['OUT_CHANNELS'][i], kernel_size=HYP['SIZE_FILTER'][i], stride=1, padding=1))
                      
           self.mapsize=int(Get_mapsize(self.mapsize, HYP['SIZE_FILTER'][i], 1, 1))   # Neue Feature Map size nach der i-ten Convolution Schicht
           if HYP['POOLING'][i] == 1:
              self.mapsize=int(self.mapsize/2)     # wird halbiert, falls Pooling-Layer aktiv                      
                                       
        # Pooling-Layer        
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)  # 

        # Dropout-Layer
        self.dropout = nn.Dropout(p=HYP['RATE_DROPOUT'])  # (z.B. 0.5 für 50% Dropout)
             
        # Linear-Layer     
        self.lin = nn.ModuleList()
        for i in range(HYP['NUM_LINLAYER']):    # i startet bei 0 !!!
           if i==0:  # 1. lineare Schicht
               in_neurons = HYP['OUT_CHANNELS'][HYP['NUM_CONVLAYER']-1] * self.mapsize * self.mapsize   # letzter Output aus den conv-Schichten
           else:
               in_neurons = HYP['NEURONS'][i-1]     

           self.lin.append (nn.Linear(in_neurons, HYP['NEURONS'][i]))
        

    def forward(self, x):
        
        for i in range(HYP['NUM_CONVLAYER']):
            if HYP['POOLING'][i] == 1:
                x = self.pool(torch.relu(self.conv[i](x)))
            else:
                x = torch.relu(self.conv[i](x))
            
        x = self.flatten(x)

        if HYP['RATE_DROPOUT'] > 0:
           x=self.dropout (x)

        for i in range(HYP['NUM_LINLAYER']-1):
            x = torch.relu(self.lin[i](x))
        x = self.lin[HYP['NUM_LINLAYER']-1](x)

        # x = torch.softmax(x, dim=1)   # konvergiert nicht, relevant für die prediction !!!
        return x

################################################################################################################

