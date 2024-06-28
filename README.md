DIG-AI-CNNG10
AI Project Repository

Dieses Projekt wurde ausgeführt in der Version Python 3.9.13

Die Quellcodedateien Project_*.* können in den SOURCE Pfad des  von DIGETHIC zur Verfügung
gestellten ML Repositories (Pfad ...\full-stack-machine-learning\SOURCE) kopiert werden
und stehen dann zur Ausführung bereit.

Ausführbar sind die Dateien:
Project.py:
Project_PlotMetrics.py
Project_PlotExamples.py

Alle benötigten Python Bibliotheken finden sich in requirements.txt
Es wurden den Requirements des ML Repositories ein paar zusätzliche Bibliotheken hinzugefügt 

Der Datensatz befindet sich in der Datei "Galaxy10_DECals.h5"
Aufgrund seiner Größe von 2.5 GB kann er mit den Basis GitHub Fähigkeiten nicht direkt hochgeladen werden und
steht gesondert über den Link <https://astronn.readthedocs.io/en/latest/galaxy10.html>
zum Download via Browser zur Verfügung
Er muss im aktuellen Verzeichnis der Terminalsession bereitgestellt sein


Das Hauptprogramm ist Project.py:
##########################################################################
# Aufgaben:
#
# Durchführung des Trainings in Verbindung mit der Validierung
# Prädiktion auf dem Testdatensatz
# Plot von den Metriken und Losses
# Speicherung von relevanten Daten für den separaten Plot von den Metriken durch Project_PlotMetrics.py
#       in das aktuelle Verzeichnis
# Anmerkungen:
#    Benötigt Project_Module_Data, Project_Module_Plot
#    GPU wird noch nicht unterstützt
#
##########################################################################



Programm Project_PlotMetrics.py
##########################################################################
# Aufgabe:
#
# Plot der von Project.py gespeicherten Metriken und Losses der Trainingsläufe
# Plot der Metriken auf den Testdaten basierend auf gespeicherten Labels und Prädiktionen durch Project.py
#
# Anmerkungen:
#    Benötigt Project_Module_Plot
#    Vor der Ausführung muss Project.py ausgeführt worden sein
#    Ggf. ist die Variable path noch anzupassen

##########################################################################


Programm Project_PlotExamples.py
##########################################################################
# Aufgabe:
#
# Plot der Klassenverteilungs-Histogramme auf Trainings-, Validierungs- und Testdatensatz
# Plot von den herunterskalierten Eingangsbildern für das CNN 
#
# Anmerkungen:
#    Benötigt Project_Module_Data, Project_Module_Plot
#
##########################################################################





