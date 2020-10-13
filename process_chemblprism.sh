#!/bin/bash

mkdir -p data/
mkdir -p "data/750Kchembl_prism/"

rm e_chemblprism.log o_chemblprism.log

bsub -P gpu -gpu - -M 40G -e e_chemblprism.log -o o_chemblprism.log -J proc_chemblprism "python /hps/research1/icortes/acunha/python_scripts/Molecular_vae/py_scripts/process_chemblprism.py"