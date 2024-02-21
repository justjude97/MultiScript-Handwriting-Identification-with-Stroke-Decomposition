#!/usr/bin/env bash

echo "starting CERUG experiments!"

rootdir=experiments/CERUG/$(date +%Y_%m_%d)
echo "root directory:" $rootdir
mkdir -p $rootdir

dataset_fp="./prepared datasets/CERUG"
echo "dataset location:" $(realpath -s "$dataset_fp")

echo "running experiment 1 (train on chinese and test on english)"
experiment_fp=$rootdir/experiment_1
mkdir $experiment_fp
echo "test program. delete later" | tee $experiment_fp/log.txt

echo "running experiment 2 (train on english and test on chinese)"
experiment_fp=$rootdir/experiment_2
mkdir $experiment_fp
echo "experiment 2 log" | tee $experiment_fp/log.txt

echo "running experiment 3 (train on chinese and english and train on MIXED)"
experiment_fp=$rootdir/experiment_3
mkdir $experiment_fp
echo "TODO..." | tee $experiment_fp/log.txt
