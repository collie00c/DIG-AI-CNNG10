##########################################################################
# Aufgabe:
#
# Plot der Klassenverteilungs-Histogramme auf Trainings-, Validierungs- und Testdatensatz
# Plot von den herunterskalierten Eingangsbildern für das CNN 
#
# Anmerkungen:
#    Benötigt Project_Module_Data, Project_Module_Plot
#
#
##########################################################################

import numpy as np

import torch
from torch.utils.data import random_split

from Project_Module_Data import GalaxyDataset,FRAC
from Project_Module_Plot import Plot_Histogramm, show_img_class
from Project_Module_Data import transform_G10_R


#========================================================================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_Galaxy10_R =  GalaxyDataset ('Galaxy10_DECals.h5', transform=transform_G10_R, target_transform=None)   # transform_G10


# Teile den Datensatz in Trainings- und Testdatensätze auf (reproduzierbar)
train_size = int(FRAC['TRAIN'] * len(dataset_Galaxy10_R))    
val_size = int(FRAC['VAL'] * len(dataset_Galaxy10_R)) 
test_size = len(dataset_Galaxy10_R) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset_Galaxy10_R ,[train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))
print('Test set has {} instances'.format(len(test_dataset)))



train_classes=np.array([])

# Erstelle einen DataLoader
train_dataloader  = torch.utils.data.DataLoader(train_dataset, batch_size=int(len(train_dataset)/10), shuffle=False)

# Iteriere durch alle Batches
for batch_idx, (train_features, train_labels) in enumerate(train_dataloader):
    train_classes=np.append(train_classes, train_labels)

    

# Erstelle ein Histogramm des Trainings-Datensatzes
Plot_Histogramm (train_classes,[x for x in list(range(0,2010,200))], num_classes=10, title=f'Trainings-Datensatz ({train_size} Instanzen)', x_label='Klasse', y_label='Häufigkeit')


val_classes=np.array([])
# Erstelle einen DataLoader
val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=int(len(val_dataset)/10), shuffle=False)
# Iteriere durch alle Batches
for batch_idx, (val_features, val_labels) in enumerate(val_dataloader):
   val_classes=np.append(val_classes, val_labels)
   
   if batch_idx==3:
       show_img_class(val_features, val_labels)    
       break
   
# Erstelle ein Histogramm des Validierungs-Datensatzes
Plot_Histogramm (val_classes,[x for x in list(range(0,1010,100))], num_classes=10, title=f'Validierungs-Datensatz ({val_size} Instanzen)', x_label='Klasse', y_label='Häufigkeit')


test_classes=np.array([])
# Erstelle einen DataLoader
test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=int(len(test_dataset)/10), shuffle=False)
# Iteriere durch alle Batches
for batch_idx, (test_features, test_labels) in enumerate(test_dataloader):
   test_classes=np.append(test_classes, test_labels)
      
# Erstelle ein Histogramm des Test-Datensatzes
Plot_Histogramm (test_classes,[x for x in list(range(0,1010,100))], num_classes=10, title=f'Test-Datensatz ({test_size} Instanzen)', x_label='Klasse', y_label='Häufigkeit')
