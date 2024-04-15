#!/usr/bin/env bash

echo "starting CERUG experiments!"

rootdir=experiments/CERUG/$(date +%Y_%m_%d)
echo "root directory:" $rootdir

#TEMP (probably) for development
rm -rf $rootdir
mkdir -p $rootdir

dataset_fp="./prepared datasets/CERUG/"
echo "dataset location:" $(realpath -s "$dataset_fp")

echo "running experiment 1 (train on chinese and test on english)"
experiment_fp=$rootdir/experiment_1
mkdir $experiment_fp
#2&>1 means to redirect stderr to stdout. to print both errors and standard output
python training.py "$experiment_fp" "$dataset_fp" CN | tee $experiment_fp/log.txt

echo "running experiment 2 (train on english and test on chinese)"
experiment_fp=$rootdir/experiment_2
mkdir $experiment_fp
python training.py "$experiment_fp" "$dataset_fp" EN | tee $experiment_fp/log.txt

echo "running experiment 3 (train on chinese and english (manually merged in MIXED))"
experiment_fp=$rootdir/experiment_3
mkdir $experiment_fp
python training.py "$experiment_fp" "$dataset_fp" MERGED | tee $experiment_fp/log.txt
