#!/bin/bash

for file in `cat check_cases.txt` ; do
    rm $file
done

job_group="check_smiles_VAE"
bgmod -L 20 /$job_group

run_type="start*"
for file in `cat /hps/research1/icortes/acunha/python_scripts/Molecular_vae/ready_check_cases.txt` ; do
    echo "$file"
    alpha=$(echo $file | cut -d'-' -f 1)
    data_from=$(echo $file | cut -d'-' -f 2)
    lr=$(echo $file | cut -d'-' -f 4)
    n_epoch=$(echo $file | cut -d'-' -f 6)
    epoch_reset=$(echo $file | cut -d'-' -f 13)
    gam=$(echo $file | cut -d'-' -f 10)
    dropout=$(echo $file | cut -d'-' -f 9)
    type_lr=$(echo $file | cut -d'-' -f 14)
    run_percentage=$(echo $file | cut -d'-' -f 15)
    max_l=$(echo $file | cut -d'-' -f 3)
    size_batch=$(echo $file | cut -d'-' -f 5)
    step=$(echo $file | cut -d'-' -f 11)
    seed=$(echo $file | cut -d'-' -f 12)
    perc_train=$(echo $file | cut -d'-' -f 7)
    perc_val=$(echo $file | cut -d'-' -f 8)
    cd "/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/alpha_${alpha}/${data_from}/${max_l}_${lr}_${size_batch}_${n_epoch}_${perc_train}_${perc_val}_${dropout}_${gam}_${step}_${seed}_${epoch_reset}_${type_lr}_${run_percentage}"
    bsub -g /$job_group -P gpu -gpu - -M 10G -e e_vae.log -o o_vae.log -J vae_smiles "python /hps/research1/icortes/acunha/python_scripts/Molecular_vae/py_scripts/molecular.py $alpha $data_from $max_l $lr $size_batch $n_epoch $perc_train $perc_val $dropout $gam $step $seed $epoch_reset $type_lr $run_percentage $run_type"
done