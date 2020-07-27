#!/bin/bash

output=$(cat list_best_parameters_0.txt | sed -n 1p)
folder=$(cat list_best_parameters_0.txt | sed -n 2p)
mkdir -p best_results/ best_results/0/
cp -r results/0/$folder best_results/0/
echo $folder
cp results/0/"output_${output}.txt" best_results/0/
echo "done!"