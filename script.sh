#!/bin/bash
job_group="smiles_VAE"
bgmod -L 5 /$job_group

run_type="start"
for alpha in "1.0" ; do # "0.00005" ; do 
    for data_from in "prism_chembl_zinc" ; do # "zinc" ; do 
        for lr in  "0.0005" ; do # "0.0001" "0.00005" "0.00001" ; do # 
            for n_epoch in "5" ; do # "500" ; do 
                for epoch_reset in "250" ; do
                    for gam in "0.6" ; do
                        for dropout in "0.0" ; do # "0.1" "0.3" "0.5" ; do
                            for type_lr in "cyclical" ; do # "non_cyclical" ; do  
                                max_l="120"
                                size_batch="64"
                                step="50"
                                seed="42"
                                perc_train="0.7"
                                perc_val="0.15"
                                FILE="/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/alpha_${alpha}/${data_from}/output_${max_l}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_lr}.txt"
                                if [ -f "$FILE" ]; then
                                    echo "$FILE exists."
                                else 
                                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/"
                                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/alpha_${alpha}/"
                                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/alpha_${alpha}/${data_from}/"
                                    mkdir -p "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/alpha_${alpha}/${data_from}/${max_l}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_lr}" && cd $_
                                    mkdir -p model_values plots pickle
                                
                                    bsub -g /$job_group -P gpu -gpu - -M 70G -e e_vae.log -o o_vae.log -J vae_smiles "python /hps/research1/icortes/acunha/python_scripts/Molecular_vae/py_scripts/molecular.py $alpha $data_from $max_l $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_lr $run_type"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done
