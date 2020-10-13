#!/bin/bash

rm e_prism_chembl_zinc.log o_prism_chembl_zinc.log
#rm -r data/PRISM_ChEMBL_ZINC/
#mkdir -p data/PRISM_ChEMBL_ZINC/
bsub -P gpu -gpu - -M 20G -e e_prism_chembl_zinc.log -o o_prism_chembl_zinc.log -J data_smiles "python py_scripts/process_dataset.py"
