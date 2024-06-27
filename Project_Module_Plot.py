########################################################################################################################
# Aufgabe:
#
# Bereitstellung von nützlichen Routinen für die Plot-Aufgaben
# 
# Anmerkungen:
#    NICHT AUSFÜHRBAR
#    Benötigt Project_Module_Data
#
#    Festlegung der Modell-Architektur durch die Hyperparameter: 
#           Die Dimensionen der Tupel, welche die Schichten definieren, müssen zu NUM_CONVLAYER und NUM_LINLAYER passen!
#           NUM_CONVLAYER und NUM_LINLAYER müssen >= 2 sein
#    
#########################################################################################################################


# from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import textwrap

from Project_Module_Data import SIZE_IMG, NUM_CLASSES


LABELS_MAP = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Round Smooth",
    4: "Cigar Shaped Smooth",
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on without Bulge",
    9: "Edge-on with Bulge"
}


def show_img (img,label):
 # Display last image and label.
    img_transformed = img.permute(1, 2, 0)
    plt.imshow(img_transformed)  
    plt.show()
    print(f"PIL:   {img}")
    print(f"Label: {label}")
    return None



def show_img_class(images, labels):
    # Display image and label.
    
    labels=labels.numpy()
    
    print(images.shape)
    print(labels.shape)
 
    # Beispielbilder plotten:
    fig = plt.figure(figsize=(10,10))
    for i in range(10):
        fig.suptitle(f'Example Galaxy Images (size {SIZE_IMG} x {SIZE_IMG})')
        ax = fig.add_subplot(2,5, i+1)
        ax.axis('off')

        try:
            idx = np.where(labels == i)[0][1]
            img_transformed = images[idx].permute(1, 2, 0)
            plt.imshow(img_transformed)
            plt.grid(None)
            # plt.tight_layout()
            #plt.title(f'Class {idx}: {LABELS_MAP[idx]}')
            #plt.title(str(labels[idx])+': '+LABELS_MAP[labels[idx]])
            plt.title(LABELS_MAP[labels[idx]])
            #plt.title(f'Label = {labels[idx]}')
        except:
            pass    

    plt.show()



def Plot_Histogramm (y_values, y_ticks=[], num_classes=NUM_CLASSES, title='Histogramm', x_label='Klasse', y_label='Häufigkeit'):
        
    plt.hist(y_values, bins=range(num_classes+1), edgecolor='black', align='left')
    plt.yticks(y_ticks)
    plt.ylabel(y_label)
    
    plt.xticks(range(num_classes),fontsize=18)      
    plt.xlabel(x_label)

    plt.title(title)
    plt.show()



def Plot_Metrics(listvalues, title, x_label, y_label, y_lim=(0,0), legend='', first_eq_second=False):
        
    plt.rc('font', size=12)  # Setze die Schriftgröße für alle Elemente    
    if first_eq_second:         
       listvalues[0]=listvalues[1]  # Korrektur zwecks Dimensionierung des Plots
    
    if legend != '':
       wrapped_legend = "\n".join(textwrap.wrap(text=legend, width=80))            
       plt.plot(listvalues, label=wrapped_legend)
    else:
       plt.plot(listvalues)       
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim != (0,0):    
       plt.ylim(y_lim[0], y_lim[1])
    plt.legend(loc='best')
    



