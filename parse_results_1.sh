#!/bin/bash

rm loss_results.txt check_cases.txt summary_results.csv check_cases.txt ready_check_cases.txt

for file in `find new_results/ -name output*`
do
    echo ${file} >> loss_results.txt
done

python "py_scripts/parse_results.py"
