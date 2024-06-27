##########################################################################
# Aufgabe:
#
# Plot der von Project.py gespeicherten Metriken und Losses der Trainingsläufe
# Plot der Metriken auf den Testdaten basierend auf gespeicherten Labels und Prädiktionen von Project.py
#
# Anmerkungen:
#    Benötigt Project_Module_Plot
#    Vor der Ausführung muss Project.py ausgeführt worden sein
#    Ggf. ist die Variable path noch anzupassen

##########################################################################

import numpy as np
import glob
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torchmetrics.classification import MulticlassROC, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve

from Project_Module_Plot import Plot_Metrics

# Laden der gespeicherten Werte aus der .npz Datei
# Pfad zum Unterverzeichnis der Metric Daten (ggf. anpassen)
####################################################
path = './full-stack-machine-learning/data/*.npz'
####################################################

# Liste aller .npz Dateien im Verzeichnis
npz_files = glob.glob(path)

# Laden jeder npz-Datei im Verzeichnis
for file in npz_files:
    data = np.load(file, allow_pickle=True)
    
    print(f'{file}: {data.files}') 
   
    if 'metrics' in data and 'labels' in data:
        print('Not supported anymore')
    else: 
        
        acc= round(float(data['acc']), 2)
        f1= round(float(data['f1']), 2)
        matthews= round(float(data['matthews']), 2)
        metrics=f"Accuracy={acc}, F1-Score={acc}, Matthews-Score={matthews}"           
        
        print(data['hyp'])
        print(metrics)
            
        Plot_Metrics (data['etl'], 'Cross Entropy Loss Entwicklung über Epochen', 'Epochen', 'Loss', y_lim=(0,0), legend='Training: '+str(data['hyp']))
        Plot_Metrics (data['evl'], 'Cross Entropy Loss Entwicklung über Epochen', 'Epochen', 'Loss', y_lim=(0,0), legend='Validation')
        plt.show()

        Plot_Metrics (data['eta'], 'Akkuranz Entwicklung über Epochen', 'Epochen', 'Accuracy [%]', y_lim=(0,0), legend='Training: '+str(data['hyp']))
        Plot_Metrics (data['eva'], 'Akkuranz Entwicklung über Epochen', 'Epochen', 'Accuracy [%]', y_lim=(0,0), legend='Validation')
        plt.show()

        Plot_Metrics (data['etf'], 'F1-Score Entwicklung über Epochen', 'Epochen', 'F1-Score', y_lim=(0,0), legend='Training: '+str(data['hyp']))
        Plot_Metrics (data['evf'], 'F1-Score Validation Entwicklung über Epochen', 'Epochen', 'F1-Score', y_lim=(0,0), legend='Validation')
        plt.show()
                          

        metric = MulticlassROC(num_classes=int(data['num_classes']))
        metric.update(torch.tensor(data['y_prob']), torch.tensor(data['y_test']))
        fig_, ax_ = metric.plot(score=True)
        plt.show()

        metric = MulticlassConfusionMatrix(num_classes=int(data['num_classes']))
        metric.update(torch.tensor(data['y_prob']), torch.tensor(data['y_test']))
        fig_, ax_ = metric.plot()
        plt.show()

        metric = MulticlassPrecisionRecallCurve(num_classes=int(data['num_classes']))
        metric.update(torch.tensor(data['y_prob']), torch.tensor(data['y_test']))
        fig_, ax_ = metric.plot(score=True)
        plt.show()
   
