'''
this is a hard copy of the original file, eb_train.py
that is modified to be used for contrastive learning
todo: refactor the main file eb_train.py to be used for contrastive learning to avoid code duplication
'''
import argparse
import os
import time

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR, ConstantLR, SequentialLR

from eb_models import TimeSeriesTransformer, TimeSeriesTransformerWithSubseq
# from util import cal_loss, Logger
from eb_dataset import EB_DS, EB_DS_batched, load_dataset
from utils import mkdir_if_needed, assign_run_name, load_config, save_config, write_to_json, parse_args_by_nested_prefix
from utils import  VanillaLogger
import warnings
import json

from parser import parser
import random

from losses import NTXentLoss

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.parallel.scatter_gather')
#
models = {'timeseriestransformer': TimeSeriesTransformer,
          'timeseriestransformerwithsubseq': TimeSeriesTransformerWithSubseq}

def _init_(args):
    #in a case of contradiction, the
    # arguments that are specified in the config file will be overwritten by the command line arguments
    if args.config_file:
        config = load_config(args.config_file)
        #load config to args. args will be used to overwrite the config
        #before that remove args that are set to None
        args.__dict__ = {k: v for k, v in args.__dict__.items() if v is not None}
        for key, value in config.items():
            if (key not in args.__dict__): # todo: or args.__dict__[key] is None:
                setattr(args, key, value)
            else:
                print('WARNING: argument {} is specified in both the config file and the command line. Using the command line value.'.format(key))

    # create the run folder
    mkdir_if_needed(args.save_path)
    args.run_name = assign_run_name(args.save_path, args.run_name)
    mkdir_if_needed(args.save_path + args.run_name)
    # create the log file
    args.json_log = args.save_path + args.run_name + '/results/epoch_data.json'
    io = VanillaLogger(args.save_path + args.run_name + '/run.log')
    #load config file if specified

    # save the config file
    save_config(args.__dict__, args.save_path + args.run_name + '/config.yaml')
    # create the checkpoints folder
    mkdir_if_needed('checkpoints')
    mkdir_if_needed('checkpoints/' + args.run_name)
    mkdir_if_needed('checkpoints/' + args.run_name + '/' + 'models')
    # copy py files to a folder in run_name
    # create a folder for the py files
    mkdir_if_needed(args.save_path  + args.run_name + '/' + 'pyfiles')
    # copy all py files to the folder
    os.system('cp *.py {}'.format( args.save_path  + args.run_name + '/' + 'pyfiles/'))
    # create a results folder
    mkdir_if_needed(args.save_path  + args.run_name + '/' + 'results')

    args.ds_args = parse_args_by_nested_prefix(args, 'ds_args', ['train', 'val'])
    print('debug ds_args: ', args.ds_args)
    args.model_args = parse_args_by_nested_prefix(args, 'model_args', [])

    return io


def train(args, io):
    if args.dataset == 'modelnet40':
        train_dataset = ModelNet40(partition='train', num_points=args.ds_args_n_samples)
        val_dataset = ModelNet40(partition='test', num_points=args.ds_args_n_samples)

    elif args.dataset == 'eb_ds':
        for modality in args.ds_args:
            args.ds_args[modality]['extra_dim_for_subsamples'] = True
        ds, labels, _ = load_dataset(base_path=args.eb_ds_path, n_samples=args.ds_args_n_samples)
        train_dataset = EB_DS(ds[:-args.n_validation], labels[:-args.n_validation],  **args.ds_args['train'])
        val_dataset = EB_DS(ds[-args.n_validation:], labels[-args.n_validation:],   **args.ds_args['val'])

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)


    device = torch.device("cuda" if args.cuda else "cpu")


    if args.model == 'timeseriestransformer' or args.model == 'timeseriestransformerwithsubseq':
        # here we only assign model parameters that are derived from the dataset parameters
        local_model_args = {}
        local_model_args['n_timesteps'] = args.ds_args_n_samples
        args.model_args.update(local_model_args)


    if args.model == 'timeseriestransformerwithsubseq':
        args.model_args['subseq_len'] = args.ds_args['train']['n_subsamples']

    io.print_and_log(f'Model args: {str(args.model_args)}')

    model = models[args.model]( **args.model_args).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr) #todo, revisit weight decay, weight_decay=1e-4)

    scheduler0 = LinearLR(opt, start_factor=0.01, total_iters=5)
    scheduler1 = ConstantLR(opt, factor=1.0, total_iters=40)
    scheduler2 = ExponentialLR(opt, gamma=0.9)
    scheduler = SequentialLR(opt, schedulers=[scheduler0,scheduler1, scheduler2], milestones=[5, int(0.4*args.epochs)])

    # scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = NTXentLoss(device=device)
    best_val_acc = 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0.0  # numbers of data
        model.train()
        train_pred = []
        train_true = []
        idx = 0  # iterations
        total_time = 0.0
        loop_start_time = time.time()
        time_monitor = 0
        for data_, label_ in (train_loader):
            times = []
            times.append(time.time())
            # data, label = data_.to(device), label_.to(device).squeeze()
            data  = data_.to(device)
            batch_size = data.size()[0]
            #in contrasitive learning the data is of a shape (batch_size, 2, n_timesteps, n_channels)
            # for passing the data to the model we need to reshape it to (batch_size*2, n_timesteps, n_channels)
            if args.flatten_contrastive_input:
                data = data.reshape(batch_size*2, *data.shape[2:])
            opt.zero_grad()
            start_time = time.time()
            projections = model(data)
            # split the projections back to the original shape
            if args.flatten_contrastive_input:
                projections = projections.view(batch_size, 2, *projections.shape[1:])
            z_i, z_j = projections.unbind(dim=1)
            loss = criterion(z_i, z_j)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            # preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            # train_true.append(label.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())
            times.append(time.time())
            these_times = np.diff(np.array(times))
            time_monitor += these_times
            idx += 1
        # loop_end_time = time.time()
        # train_true = np.concatenate(train_true)
        # train_pred = np.concatenate(train_pred)

        # collect epoch data into a dictionary
        epoch_data = {}
        epoch_data['epoch'] = epoch
        epoch_data['train_loss'] = train_loss * 1.0 / count
        # epoch_data['train_acc'] = metrics.accuracy_score(train_true, train_pred)
        # epoch_data['train_avg_acc'] = metrics.balanced_accuracy_score(train_true, train_pred)
        epoch_data['train_time'] = total_time
        epoch_data['lr'] = scheduler.get_last_lr()

        # ####################
        # # Validation
        # ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        total_time = 0.0
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            if args.flatten_contrastive_input:
                data = data.reshape(batch_size*2, *data.shape[2:])
            start_time = time.time()
            projections = model(data)
            # split the projections back to the original shape
            if args.flatten_contrastive_input:
                projections = projections.view(batch_size, 2, *projections.shape[1:])
            z_i, z_j = projections.unbind(dim=1)
            loss = criterion(z_i, z_j)
            count += batch_size
            val_loss += loss.item() * batch_size
            end_time = time.time()
            total_time += (end_time - start_time)
            # val_true.append(label.cpu().numpy())
            # val_pred.append(preds.detach().cpu().numpy())
        # # print ('validation total time is', total_time)
        # # val_true = np.concatenate(val_true)
        # # val_pred = np.concatenate(val_pred)
        # # val_acc = metrics.accuracy_score(val_true, val_pred)
        # # avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
        #
        epoch_data['val_loss'] = val_loss * 1.0 / count
        # # epoch_data['val_acc'] = val_acc
        # # epoch_data['val_avg_acc'] = avg_per_class_acc
        epoch_data['val_time'] = total_time

        outstr =  ("Epoch {}, train loss: {:.6f},  "
                    "val loss: {:.6f},  learning rate {:.3e}, elapsed time {:.3f}"
                ).format(
                    epoch,
                    epoch_data['train_loss'],
                    # epoch_data['train_acc'],
                    # epoch_data['train_avg_acc'],
                    epoch_data['val_loss'],
                    # epoch_data['val_acc'],
                    # epoch_data['val_avg_acc'],
                    epoch_data['lr'][0],
                    epoch_data['train_time'] + epoch_data['val_time']
)
        #append to the epoch data to the log file
        write_to_json(args.json_log, epoch_data)

        io.print_and_log(outstr)
        # if val_acc >= best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.run_name)
        #
        scheduler.step()
        torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.run_name)
if __name__ == "__main__":

    args = parser.parse_args()

    io = _init_(args)

    io.print_and_log(str(args))
    io.print_and_log(f'Dataset args (training): {str(args.ds_args["train"])}')
    io.print_and_log(f'Dataset args (validation): {str(args.ds_args["val"])}')

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.cuda:
        io.print_and_log(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.print_and_log('Using CPU')

    train(args, io)

