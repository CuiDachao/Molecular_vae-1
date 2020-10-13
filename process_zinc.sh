#!/bin/bash

mkdir -p data/
mkdir -p data/zinc/

rm e_zinc.log o_zinc.log

bsub -M 40G -e e_zinc.log -o o_zinc.log -J proc_zinc "python /hps/research1/icortes/acunha/python_scripts/Molecular_vae/py_scripts/process_zinc.py"
