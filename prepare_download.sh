#!/bin/bash

cp -v -r results/ to_delete/
cd to_delete/
find /hps/research1/icortes/acunha/python_scripts/Molecular_vae/to_delete/ -name "*pkl" > delete.txt
for file in `cat delete.txt`
do
    echo $file
    rm -r $file
done
pwd
