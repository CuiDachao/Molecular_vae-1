#!/bin/bash

for file in `find results/ -name output*`
do
    cat ${file} | grep "Training ::" >> loss_results.txt #tr -dc "0-9"
    cat ${file} | grep "Validation ::" >> loss_results.txt #tr -dc "0-9"
    cat ${file} | grep "Testing ::" >> loss_results.txt #tr -dc "0-9"
    cat ${file} | grep "Valid molecules:" >> loss_results.txt #tr -dc "0-9"
    echo ${file} >> loss_results.txt
    echo "$file"
done


python "py_scripts/parse_results.py"
