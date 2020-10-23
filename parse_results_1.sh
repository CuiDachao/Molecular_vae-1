#!/bin/bash

rm loss_results.txt check_cases.txt summary_results.csv

for file in `find new_results/ -name output*`
do
    echo ${file} >> loss_results.txt
done

python "py_scripts/parse_results.py"
