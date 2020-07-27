#!/bin/bash

output=$(cat list_best_parameters.txt | sed -n 1p)
folder=$(cat list_best_parameters.txt | sed -n 2p)
mkdir -p best_results/
cp -r results/$folder best_results/
echo $folder
cp "results/${output}.txt" best_results/
echo "done!"