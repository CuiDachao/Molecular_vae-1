#!/bin/bash

mkdir -p '/hps/research1/icortes/acunha/data/ZINC_PRISM_SMILES/'

bsub -q research-rh74 -P gpu -gpu - -M 10G -e e_create.log -o o_create.log -J zinc_prism "python /hps/research1/icortes/acunha/python_scripts/vae_smiles_prism_zinc/py_scripts/create_dataset.py"