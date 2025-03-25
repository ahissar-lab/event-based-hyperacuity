# README

This README file contains instructions on how to train the models presented in the 'Temporal coding enables hyperacuity in event based vision' paper using the accompanying supplied code.
See license.md for code usage licence.

## License

This project is licensed under the [Custom License](LICENSE.md), which allows for non-commercial use only. You may not use this code for commercial purposes without obtaining explicit permission from the author.

For more details, please refer to the full license in the [LICENSE.md](LICENSE.md) file.

## Setting up the environment

```sh
conda env create -f environment.yml -n [new_env_name]
```
The trainig code was tested on single GPU, in case of running on multi-GPU server please set the 'CUDA_VISIBLE_DEVICES' env vriable to a single gpu to allow proper execution. For example:

```sh
export CUDA_VISIBLE_DEVICES=0
```

## Downloading the required datasets
The tiny-event-based and tiny-frame-based datasets used to produce the results depicted in the paper can be downloaded from the following link: [datasets](https://drive.google.com/file/d/1CMu9Q_caL5lvrl86f6cVqcb_VM6gWnp2/view?usp=sharing) 

## Evaluate the results from the paper

### Train a supervised model

The general training command is:

```sh
python eb_train.py --run_name ${NAME}  --config_file {$CONFIG_FILE} --eb_ds_path ${DATASET} --model_args_model_head ${MODEL_HEAD} --ds_args_n_samples ${EVENTS} --batch_size ${BATCH} --epochs ${EPOCHS} --seed ${SEED} --save_path ${S_PATH}
```
where:
NAME: training instance identifying name
CONFIG_FILE: path to a file containing all required/optional parameters. parameters included at the cmd-line override values from the config file 
DATASET: path to directory containing dataset
MODEL_HEAD: model head to use, see code for avialable options
EVENTS: the number of events the model gets as an input for each training sample
BATCH: batch size
EPOCHS: number of training epochs
SEED: random number generator seed to use during training
S_PATH: path to save the training output files (make sure it exists)

The 'config' directory contains config files with the set of prarmeters used for the different configurations reported in the paper. see examples below.

Using the below mentioned config files would result with evaluating the performance of the trained model over the test-set data at the end of the training session.

Examples:
Supervised training of a PointTransformer model with the ebtMNISTD1 dataset using 48 events (remember to first activate the conda env):
```sh
python eb_train.py --run_name ebtMNISTD1_supervised --config_file config/config_eb_PTr.yaml  --eb_ds_path ../datasets/ebtMNISTD1 --model_args_model_head cls_avgpool --ds_args_n_samples 48 --seed 42 --save_path ./saved_runs/
```

Supervised training of a ResNet18 model with the fbtMNISTD1 dataset using 48 events (remember to first activate the conda env):
```sh
python eb_train.py --run_name fbtMNISTD1_supervised_resnet18_  --config_file config/config_fb_res18.yaml --eb_ds_path ../datasets/fbtMNISTD1 --model GrayscaleResNet18 --lr 0.001 --no_warmup --seed 42 --save_path ./saved_runs/
```

### NMNIST

### Train contrastive model 

### Train a supervised head over the contrastive pre-trained represetation

### Train a constrative large model 

### Train a GRU model

### Train Vernier clssifier

