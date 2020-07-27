#!/bin/bash

#cyclical lr
for alpha in "1.0" "0.00005" ; do
    for lr in  "0.05" "0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00005"; do #"0.00001" 
        for n_epoch in "2000" ; do 
            for epoch_reset in "500" ; do
                for gam in "0.9" ; do
                    max_l="120"
                    size_batch="64"
                    dropout="0"
                    step="100"
                    seed="42"
                    perc_train="0.7"
                    perc_val="0.15"
                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/results/"
                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/results/${alpha}/"
                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/results/${alpha}/${max_l}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${epoch_reset}" && cd $_
                    mkdir -p model_values plots pickle
                
                    bsub -P gpu -gpu - -M 80G -e e_vae.log -o o_vae.log -J vae_smiles "python /hps/research1/icortes/acunha/python_scripts/Molecular_vae/py_scripts/molecular.py $alpha $max_l $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset"
                    echo "output_${max_l}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${epoch_reset}" >> "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/list_original_parameters_${alpha}.txt"
                done
            done
        done
    done
done
