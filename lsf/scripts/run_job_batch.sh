#! /usr/bin/bash

module purge
module load Singularity

declare -a nevents_arr=(48 48 48 48 48)
declare -a ds_arr=(20230928 20230928 20230928 20230928 20230928)
declare -a seed_arr=(43 44 45 46 47)
declare -a jitter_arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)

python eb_train.py --run_name ebtMNISTD1_supervised_ --config_file config/config_eb_PTr.yaml  --eb_ds_path ../datasets/ebtMNISTD1 --model_args_model_head cls_avgpool --ds_args_n_samples 48 --seed 42 --save_path ./saved_runs/

# get length of an array
length=${#nevents_arr[@]}
 
# use C style for loop syntax to read all values and indexes
for (( i=0; i<length; i++ ));
do
  #printf "Current index %d with value %s\n" $i "${nevents_arr[$i]}"
  #bsub -q gpu-short -gpu num=1:j_exclusive=no:gmem=20G -R "affinity[thread*4] select[mem>8000] rusage[mem=8000]"  -o cosyne_results/out_train_eb_transformer.%J.log -e cosyne_results/err_train_eb_transformer.%J.log  singularity exec --nv --cleanenv --env LSB_JOBID=%J ../drc_object_detection/singularity/tfod.sif python train_eb_transformer.py --n_samples=${nevents_arr[$i]} --ts_jitter=0.1 --np_seed=44 --ds_id=20230711
  bsub -q gpu-short -gpu num=1:j_exclusive=yes:gmem=20G -R "affinity[thread*4] select[mem>8000] rusage[mem=8000]"  -o cosyne_results/out_train_eb_transformer.%J.log -e cosyne_results/err_train_eb_transformer.%J.log  singularity exec --nv ../drc_object_detection/singularity/tfod.sif python train_eb_transformer.py --n_samples=${nevents_arr[$i]} --ts_jitter=${jitter_arr[$i]} --np_seed=${seed_arr[$i]} --ds_id=${ds_arr[$i]}
# --ts_shuffle
done


