#!/bin/bash

rm e_check.log o_check.log
rm -r check_valid/
mkdir -p check_valid/
bsub  -P gpu -gpu - -M 50G -e e_check.log -o o_check.log -J check_50 "python py_scripts/check_valid_smiles.py"
