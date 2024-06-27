##########################################################################
# Aufgaben:
#
# Durchführung des Trainings in Verbindung mit der Validierung
# Prädiktion auf dem Testdatensatz
# Plot von den Metriken und Losses
# Speicherung von relevanten Daten für den separaten Plot von den Metriken durch Project_PlotMetrics.py
#
# Anmerkungen:
#    Benötigt Project_Module_Data, Project_Module_Plot
#    GPU wird noch nicht unterstützt
##########################################################################


from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

#from torchvision.transforms import ToTensor, Lambda

from torch.utils.data import random_split
from torchmetrics.classification import MulticlassROC, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from Project_Module_Data import GalaxyDataset, CNN_G10, FRAC, HYP, HYP_STRING, NUM_EPOCHS, NUM_CLASSES, SIZE_IMG, NUM_FILTER
from Project_Module_Data import transform_G10_R, target_transform_G10
from Project_Module_Plot import Plot_Metrics

#################################################################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#################################################################################################################

dataset_Galaxy10_R =  GalaxyDataset ('Galaxy10_DECals.h5', transform=transform_G10_R, target_transform=target_transform_G10) 

# Teile den Datensatz in Trainings- und Testdatensätze auf (reproduzierbar)
train_size = int(FRAC['TRAIN'] * len(dataset_Galaxy10_R))    
val_size = int(FRAC['VAL'] * len(dataset_Galaxy10_R)) 
test_size = len(dataset_Galaxy10_R) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset_Galaxy10_R ,[train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))
print('Test set has {} instances'.format(len(test_dataset)))

#################################################################################################################
################################ TRAINING #######################################################################
#################################################################################################################

# Erstelle DataLoader für Trainings- und Validierungsdatensatz
train_dataloader  = torch.utils.data.DataLoader(train_dataset, batch_size=HYP['SIZE_BATCH'], shuffle=True)   # shuffle=True: Datensätze mischen während des Trainings
val_dataloader    = torch.utils.data.DataLoader(val_dataset, batch_size=HYP['SIZE_BATCH'], shuffle=False)     
test_dataloader   = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  

# Modell Instanz 
model = CNN_G10()  
print (model)

model.train()

# Weitere Hyperparameter:
# Geeigneter Loss für mehrere Kategorien und Optimizer:
criterion =   nn.CrossEntropyLoss()   
optimizer =   optim.Adam (params=model.parameters(), lr=HYP['LEARN_RATE']) 
#optimizer =  optim.Rprop (params=model.parameters(), lr=HYP['LEARN_RATE'])       Schlecht!

# Für Visualisierung
epoch_train_loss = []
epoch_train_accuracy=[]
epoch_train_mcc=[]
epoch_train_f1=[]

epoch_val_loss = []
epoch_val_accuracy=[]
epoch_val_f1=[]

for epoch in range(NUM_EPOCHS):
    model.train()
     
    # Für die Mittelwertbildung über alle Batches:
    batch_train_loss = []
    batch_train_accuracy=[]
    batch_train_mcc=[]
    batch_train_f1=[]

    batch_val_loss=[]
    batch_val_accuracy=[]
    batch_val_f1=[]
    
    # Iteriere durch alle Batches:
    for batch_idx, (train_features, train_labels) in enumerate(train_dataloader):

        if epoch==0 and batch_idx==0:
            print(f":{batch_idx}: Feature batch shape: {train_features.size()}")
            print(f":{batch_idx}: Labels batch shape:  {train_labels.size()}")
  
        # Gradienten zurücksetzen            
        optimizer.zero_grad()  

        outputs=model(train_features)
        loss = criterion(outputs, train_labels.squeeze())     # dim1 entfernen

        # Wähle für die Prädiktion die Klasse mit der höchsten Wahrscheinlichkeit
        _ , predicted = torch.max(outputs, dim=1)       
        # Rücktransformation der One-Hot Kodierung der train_labels
        labels = torch.argmax(train_labels, dim=1)   
                                     
        # Gradienten berechnen              
        loss.backward()   
        # Gewichte aktualisieren via Backpropagation
        optimizer.step() 
               
        # Metriken für das Trainings-Dataset berechnen (pro Epoche):

        # Metriken pro Batch berechnen
        batch_accuracy = accuracy_score(labels, predicted) * 100
        batch_mcc = matthews_corrcoef (labels, predicted)    
        batch_f1 = f1_score(labels, predicted, average='weighted')

        # Für die Mittelwertbildung über alle Batches
        batch_train_loss.append(loss.item())
        batch_train_accuracy.append(batch_accuracy)
        batch_train_mcc.append(batch_mcc)
        batch_train_f1.append(batch_f1)

    
    # Mittelwerte über die Batch-Werte berechnen:
    train_loss_mean=np.mean(batch_train_loss) 
    # Akkuranz berechnen
    train_accuracy_mean =np.mean(batch_train_accuracy)  
    # MCC berechnen:
    train_mcc_mean = np.mean(batch_train_mcc)
    #F1 Score berechnen
    train_f1_mean = np.mean(batch_train_f1)

    # Für die spätere Visualisierung:
    epoch_train_loss.append(train_loss_mean)
    epoch_train_accuracy.append(train_accuracy_mean)
    epoch_train_mcc.append(train_mcc_mean)
    epoch_train_f1.append(train_f1_mean)
   
    # Metriken für das Validierungs-Dataset berechnen (pro Epoche):
    model.eval()
    for b_idx, (val_features, val_labels) in enumerate(val_dataloader):
        with torch.no_grad():
            outputs=model(val_features)
            loss = criterion(outputs, val_labels.squeeze())    # dim1 entfernen

            # Wähle für die Prädiktion die Klasse mit der höchsten Wahrscheinlichkeit
            _ , predicted = torch.max(outputs, dim=1)       
            # Rücktransformation der One-Hot Kodierung der train_labels
            labels = torch.argmax(val_labels, dim=1)   

            batch_accuracy = accuracy_score(labels, predicted) * 100
            batch_f1 = f1_score(labels, predicted, average='weighted')

            # Für die Mittelwertbildung über alle Batches
            batch_val_loss.append(loss.item())
            batch_val_accuracy.append(batch_accuracy)
            batch_val_f1.append(batch_f1)
            

    # Mittelwerte über die Batch-Werte berechnen:
    val_loss_mean=np.mean(batch_val_loss) 
    # Akkuranz berechnen
    val_accuracy_mean =np.mean(batch_val_accuracy)
    #F1 Score berechnen
    val_f1_mean = np.mean(batch_val_f1)

    # Für die spätere Visualisierung:
    epoch_val_loss.append(val_loss_mean)
    epoch_val_accuracy.append(val_accuracy_mean)
    epoch_val_f1.append(val_f1_mean)
        
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}] ==> (Train-Loss, Val-Loss): ({train_loss_mean},{val_loss_mean}) ==>
           (Train-Acc, Val-Acc): ({round(train_accuracy_mean, 2)}, {round(val_accuracy_mean, 2)}) ==>
           (Train-F1, Val-F1): ({round(train_f1_mean, 2)}, {round(val_f1_mean, 2)})")  


########################################################################################
########### Prädiktion #################################################################
########################################################################################

model.eval()
    
batch_test_accuracy=[]
batch_test_f1=[]
batch_test_mcc=[]

batch_test_probabilities=[]
batch_test_labels=[]
batch_test_predictions=[]

for b_idx, (test_features, test_labels) in enumerate(test_dataloader):
    with torch.no_grad():
        outputs=model(test_features)
        batch_probabilities = nn.functional.softmax(outputs, dim=1)
        
        # Wähle für die Prädiktion die Klasse mit der höchsten Wahrscheinlichkeit
        _ , predicted = torch.max(outputs, dim=1)       
        # Rücktransformation der One-Hot Kodierung der test_labels
        labels = torch.argmax(test_labels, dim=1)   

        batch_accuracy = accuracy_score(labels, predicted) * 100
        batch_f1 = f1_score(labels, predicted, average='weighted')
        batch_mcc = matthews_corrcoef (labels, predicted)  

        # Für die Mittelwertbildung über alle Batches:
        batch_test_accuracy.append(batch_accuracy)
        batch_test_f1.append(batch_f1)
        batch_test_mcc.append (batch_mcc)

        batch_test_probabilities.append(batch_probabilities)                                    
        batch_test_labels.append(labels)
        batch_test_predictions.append(predicted)


# Akkuranz berechnen
test_accuracy_mean =np.mean(batch_test_accuracy)
# F1 Score berechnen
test_f1_mean = np.mean(batch_test_f1)
# Matthews Score berechnen
test_mcc_mean = np.mean(batch_test_mcc)

print(f'Accuracy, F1, Matthews = {test_accuracy_mean}, {test_f1_mean}, {test_mcc_mean}')


y_test = torch.cat([tensor for tensor in batch_test_labels])
y_pred = torch.cat([tensor for tensor in batch_test_predictions])
y_prob = torch.cat([tensor for tensor in batch_test_probabilities])


########################################################################################################################
# Metriken und Modell speichern ########################################################################################
########################################################################################################################

formatted_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
np.savez("./Plot_Metrics_"+formatted_time+".npz", hyp=HYP_STRING, num_classes=NUM_CLASSES,
          acc=test_accuracy_mean, f1=test_f1_mean, matthews=test_mcc_mean,
          etl=np.array(epoch_train_loss),
          evl=np.array(epoch_val_loss),
          eta=np.array(epoch_train_accuracy),
          eva=np.array(epoch_val_accuracy), 
          etf=np.array(epoch_train_f1),
          evf=np.array(epoch_val_f1),
          y_test=y_test.numpy(), y_pred=y_pred.numpy(), y_prob=y_prob.numpy() )

###################################################################################################################

# Speichern des Modells, internen Modell-Zustands und der Hyperparameter:
file_model="./cnn_G10_"+formatted_time+".pth"
torch.save({
    'model': model,
    'model_state_dict': model.state_dict(),
    'hyperparameters': HYP,
    'NUM_CLASSES' : NUM_CLASSES,
    'SIZE_IMG' : SIZE_IMG,
    'NUM_FILTER': NUM_FILTER
}, file_model)


###################################################################################################################
# Metriken plotten:
###################################################################################################################
plt.figure(figsize=(10, 6)) 

Plot_Metrics (epoch_train_loss, 'Cross Entropy Loss Entwicklung über Epochen', 'Epochen', 'Loss', y_lim=(0,0), legend='Training: '+HYP_STRING)
Plot_Metrics (epoch_val_loss,   'Cross Entropy Loss Entwicklung über Epochen', 'Epochen', 'Loss', y_lim=(0,0), legend='Validation')
plt.show()

Plot_Metrics (epoch_train_accuracy, 'Akkuranz Entwicklung über Epochen', 'Epochen', 'Accuracy [%]', y_lim=(0,0), legend='Training: '+HYP_STRING)
Plot_Metrics (epoch_val_accuracy,   'Akkuranz Entwicklung über Epochen', 'Epochen', 'Accuracy [%]', y_lim=(0,0), legend='Validation')
plt.show()

Plot_Metrics (epoch_train_f1, 'F1-Score Entwicklung über Epochen', 'Epochen', 'F1-Score', y_lim=(0,0), legend='Training: '+HYP_STRING)
Plot_Metrics (epoch_val_f1, 'F1-Score Validation Entwicklung über Epochen', 'Epochen', 'F1-Score', y_lim=(0,0), legend='Validation')
plt.show()

metric = MulticlassROC(num_classes=NUM_CLASSES)
metric.update(y_prob, y_test)
fig_, ax_ = metric.plot(score=True)
plt.show()

metric = MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
metric.update(y_prob, y_test)
fig_, ax_ = metric.plot()
plt.show()

metric = MulticlassPrecisionRecallCurve(num_classes=NUM_CLASSES)
metric.update(y_prob, y_test)
fig_, ax_ = metric.plot(score=True)
plt.show()

