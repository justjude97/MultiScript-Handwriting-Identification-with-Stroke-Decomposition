# Stroke Fragmentation and Training/Evaluation Code
this directory contains the implementation of "Multi-Script Writer Identification through Fragmenting Strokes". The stroke fragmentation method is performed through *prepare_datasets.py* which is run on it's own. The model training is implmented as a python script accepting arguments from the command line. Each experiment training run is then automated using the *run_cerug.sh* bash script.

## Evaluation
The evaluation of all three CERUG Experiments is carried out with the *CERUG_Experiment_{number}.ipynb* notebooks.

## Prototyping
The prototyping folder contains examples and rough-draft implementations of the stroke fragmentation and training processes. This folder can be ignored when running the experiments.