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

from eb_models import TimeSeriesTransformer, TimeSeriesTransformerWithSubseq, VanillaCNN, GrayscaleResNet18, TimeSeriesGRU
# from util import cal_loss, Logger
from eb_dataset import EB_DS, EB_DS_batched, load_dataset
from utils import mkdir_if_needed, assign_run_name, load_config, save_config, write_to_json, parse_args_by_nested_prefix
from utils import load_and_preprocess_headless_checkpoint, weigh_freezer, relabler, remove_checkpoint_prefix
from utils import  VanillaLogger
from utils import parse_strings_to_lists
from utils import compute_mean_offsets
import warnings
import json

from parser import parser
import random
from losses import NTXentLoss
import time

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.parallel.scatter_gather')
#
models = {'timeseriestransformer': TimeSeriesTransformer,
          'timeseriestransformerwithsubseq': TimeSeriesTransformerWithSubseq,
          'vanillaCNN':VanillaCNN,
          'GrayscaleResNet18':GrayscaleResNet18,
          'TimeSeriesGRU':TimeSeriesGRU}


def _init_(args):
    # Load config file if specified
    if args.config_file:
        config = load_config(args.config_file)
        args.__dict__ = {k: v for k, v in args.__dict__.items() if v is not None}
        for key, value in config.items():
            if key not in args.__dict__:
                setattr(args, key, value)
            else:
                print(
                    f'WARNING: argument {key} is specified in both the config file and the command line. Using the command line value.')

    parse_strings_to_lists(args)

    # check if recovery is needed
    args.resume_run_name = None
    if args.check_job_recovery:
        # Add LSF job id to the args
        lsb_jobid = os.getenv('LSB_JOBID')
        args.job_id = lsb_jobid + (
            ('.' + args.job_id_suffix) if args.job_id_suffix is not None and len(args.job_id_suffix) > 0 else '')
        if lsb_jobid is None:
            raise Exception('CLUSTER mode specified, but LSF job id is not found')

        # Check if a file 'lsf_manager/lsf_top.<job_id>' exists.
        lsf_top_file = 'lsf_manager/lsf_top.' + args.job_id
        args.run_name_history = []
        if os.path.exists(lsf_top_file):
            with open(lsf_top_file, 'r') as file:
                for line in file:
                    args.run_name_history.append(line.strip())

            # Check if the last entry is '>>> DONE'
            if args.run_name_history and args.run_name_history[-1] == '>>> DONE':
                print('Run already finished. Exiting...')
                exit(0)

            # Check for run names in the history for 'last_model_saved'
            for run_name in reversed(args.run_name_history):
                if os.path.exists('checkpoints/' + run_name + '/models/last_model_saved'):
                    args.resume_run_name = run_name
                    break
        else:
            with open(lsf_top_file, 'w') as file:
                file.write('')

        # Create a new entry for the current run
        print('debug args.save_path:', args.save_path)
        args.run_name = assign_run_name(args.save_path, args.run_name + '_' + lsb_jobid if lsb_jobid else '')
        with open(lsf_top_file, 'a') as file:
            file.write(args.run_name + '\n')
    else:
        # Generate the run name without recovery
        args.run_name = assign_run_name(args.save_path, args.run_name)



    # Create the necessary directories after recovery check
    mkdir_if_needed(args.save_path + args.run_name)
    args.json_log = args.save_path + args.run_name + '/results/epoch_data.json'
    args.json_test_data = args.save_path + args.run_name + '/results/test_data.json'
    io = VanillaLogger(args.save_path + args.run_name + '/run.log')

    # Save the config file
    save_config(args.__dict__, args.save_path + args.run_name + '/config.yaml')
    mkdir_if_needed('checkpoints')
    mkdir_if_needed('checkpoints/' + args.run_name)
    mkdir_if_needed('checkpoints/' + args.run_name + '/models')
    mkdir_if_needed('lsf_manager')
    mkdir_if_needed(args.save_path + args.run_name + '/pyfiles')
    os.system('cp *.py {}'.format(args.save_path + args.run_name + '/pyfiles/'))
    mkdir_if_needed(args.save_path + args.run_name + '/results')

    args.ds_args = parse_args_by_nested_prefix(args, 'ds_args', ['train', 'val'])
    print('debug ds_args: ', args.ds_args)
    args.model_args = parse_args_by_nested_prefix(args, 'model_args', [])
    args.loss_args = parse_args_by_nested_prefix(args, 'loss_args', [])
    args.opt_args = parse_args_by_nested_prefix(args, 'opt_args', [])

    args.start_epoch = 0

    if args.resume_run_name is not None:
        os.system(
            'cp ' + args.save_path + args.resume_run_name + '/results/epoch_data.json ' + args.save_path + args.run_name + '/results/epoch_data.json')
        os.system(
            'cp ' + args.save_path + args.resume_run_name + '/run.log ' + args.save_path + args.run_name + '/run.log')
        io = VanillaLogger(args.save_path + args.run_name + '/run.log', append=True)
        io.print_and_log(f'Resuming from run {args.resume_run_name}')

    return io

def create_dataloaders(args, shuffle_order=None, do_subsplits=True, eval=False):
    '''
    we refer to splits in the dataset that have dedicated folders as supersplits
    we refer to splits in the dataset that are created by splitting the data in the same folder as subsplits
    '''

    if args.dataset == 'modelnet40':
        train_dataset = ModelNet40(partition='train', num_points=args.ds_args_n_samples)
        val_dataset = ModelNet40(partition='test', num_points=args.ds_args_n_samples)
    elif args.dataset == 'eb_ds':
        if args.supervision_mode == 'contrastive':
            for modality in args.ds_args:
                args.ds_args[modality]['extra_dim_for_subsamples'] = True

        #if both eb_ds_path and eb_ds_name are specified, throw an error, same if none of them are specified
        ds_by_path = hasattr(args, 'eb_ds_path') and args.eb_ds_path is not None
        frame_based = hasattr(args, 'frame_based') and args.frame_based is not None

        ds_by_tonic_name = hasattr(args, 'eb_ds_tonic_name') and args.eb_ds_tonic_name is not None
        if ds_by_path + ds_by_tonic_name != 1:
            raise Exception('Exactly one of eb_ds_path and eb_ds_tonic_name should be specified')

        if ds_by_path and not frame_based:
            ds, labels, shuffle_order = load_dataset(base_path=os.path.join(args.eb_ds_path,args.ds_test_subdir if eval else args.ds_train_subdir),
                                                                           n_samples=args.ds_args_n_samples,
                                                                           shuffle_order=shuffle_order, jitter_ts=args.eb_ds_ts_jitter,
                                                                            imu_channels=args.eb_ds_imu_channels if hasattr(args, 'eb_ds_imu_channels') else None,
                                                     time_align_by_imu_edge=args.eb_ds_time_align_by_imu_edge if hasattr(args, 'eb_ds_time_align_by_imu_edge') else False)
        elif frame_based:
            ds, labels, shuffle_order = load_dataset(base_path=os.path.join(args.eb_ds_path,args.ds_test_subdir if eval else args.ds_train_subdir),
                                                                           shuffle_order=shuffle_order,
                                                                           frame_based=frame_based)
            #todo: organize more elegant handling of None
        else: #ds_by_name, otherwise the exception would have been raised before
            ds, labels, shuffle_order = load_dataset(tonic_name=args.eb_ds_tonic_name, tonic_train_split=not eval, tonic_path=args.eb_ds_tonic_path,
                                                     n_samples=args.ds_args_n_samples,
                                                      shuffle_order=shuffle_order, jitter_ts=args.eb_ds_ts_jitter)

        if hasattr(args, 'relabel_opt') and args.relabel_opt is not None:
            labels = relabler(labels, args.relabel_opt)
        if do_subsplits:
            train_dataset = EB_DS(ds[:-args.n_validation], labels[:-args.n_validation],  **args.ds_args['train']) if args.n_validation > 0 else (
                EB_DS(ds, labels,  **args.ds_args['train']))
            val_dataset = EB_DS(ds[-args.n_validation:], labels[-args.n_validation:],   **args.ds_args['val']) if args.n_validation > 0 else None
        else:
            val_dataset = EB_DS(ds, labels, **args.ds_args['val'])

    else:
        raise Exception('Unknown dataset')

    use_padding_mask = False if not hasattr(val_dataset, 'return_padding_mask') else val_dataset.return_padding_mask
    record_spike_count = use_padding_mask

    if do_subsplits:
        drop_last_in_train = args.n_validation > 0 #todo: this is a temporary solution that ensures backward compatibility with the previous version
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=True, drop_last=drop_last_in_train, pin_memory=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True) if args.n_validation > 0 else None

    if do_subsplits:
        return train_loader, val_loader, use_padding_mask, record_spike_count, shuffle_order
    else:
        return val_loader, use_padding_mask, record_spike_count

def create_model(args, io):
    # if args.model == 'timeseriestransformer' or args.model == 'timeseriestransformerwithsubseq':
    #     'vanillaCNN': VanillaCNN,
    #     'GrayscaleResNet18': GrayscaleResNet18,
    if not(args.model in ['vanillaCNN', 'GrayscaleResNet18']):
        # here we only assign model parameters that are derived from the dataset parameters
        local_model_args = {}
        local_model_args['n_timesteps'] = args.ds_args_n_samples
        args.model_args.update(local_model_args)

    if args.model == 'timeseriestransformerwithsubseq':
        args.model_args['subseq_len'] = args.ds_args['train']['n_subsamples']

    io.print_and_log(f'Model args: {str(args.model_args)}')

    model = models[args.model]( **args.model_args)

    if args.pretrained_model_path is not None and args.pretrained_model_path != '':
        if args.pretrained_model_no_head:
            o = model.load_state_dict(load_and_preprocess_headless_checkpoint(args.pretrained_model_path),strict=False)
            print('load pretrained model from {} WITHOUT HEAD'.format(args.pretrained_model_path))
            print('missing keys', o.missing_keys)
            print('unexpected keys', o.unexpected_keys)

        else:
            cp_dict = torch.load(args.pretrained_model_path)
            if 'model_state_dict' in cp_dict:
                cp_dict = cp_dict['model_state_dict']
            model.load_state_dict(remove_checkpoint_prefix(cp_dict))
            print('load pretrained model from {}'.format(args.pretrained_model_path))

    if args.train_head_only:
        weigh_freezer(model, dont_freeze_prefix='cls_head.')

    return model


def create_optimizer(model, args):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4,**args.opt_args)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, **args.opt_args)  # todo, revisit weight decay, weight_decay=1e-4)

    scheduler0 = LinearLR(opt, start_factor=0.01, total_iters=5)
    scheduler1 = ConstantLR(opt, factor=1.0, total_iters=40)
    scheduler2 = ExponentialLR(opt, gamma=0.9)
    scheduler = SequentialLR(opt, schedulers=[scheduler0 if args.do_warmup else ConstantLR(opt, factor=1.0),
                                              scheduler1, scheduler2],
                             milestones=[5, int(0.4 * args.epochs)])
    # scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    return opt, scheduler

def create_output_string(epoch_data, args, record_spike_count=False):
    if args.supervision_mode == 'supervised':
        outstr = ("Epoch {}, train loss: {:.6f}, train acc: {:.6f}, train avg acc: {:.6f}, "
                  "val loss: {:.6f}, val acc: {:.6f}, val avg acc: {:.6f}, learning rate {:.3e}, elapsed time {:.3f}"
                  ).format(
            epoch_data['epoch'],
            epoch_data['train_loss'],
            epoch_data['train_acc'],
            epoch_data['train_avg_acc'],
            epoch_data['val_loss'],
            epoch_data['val_acc'],
            epoch_data['val_avg_acc'],
            epoch_data['lr'][0],
            epoch_data['train_time'] + epoch_data['val_time']
        )
    else:
        outstr = ("Epoch {}, train loss: {:.6f}, val loss: {:.6f}, learning rate {:.3e}, elapsed time {:.3f}"
                  ).format(
            epoch_data['epoch'],
            epoch_data['train_loss'],
            epoch_data['val_loss'],
            epoch_data['lr'][0],
            epoch_data['train_time'] + epoch_data['val_time']
        )
    if record_spike_count:
        spike_count_str = f", train avg spike count: {epoch_data['train_avg_spike_count']:.2f}, val avg spike count: {epoch_data['val_avg_spike_count']:.2f}"
        outstr += spike_count_str
    return outstr


def run_epoch(model, loader, criterion, opt, device, epoch, mode='train', use_padding_mask=False, result_prefix=None,
              record_spike_count=False, scheduler=None, args=None):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    loss = 0.0
    count = 0.0
    pred = []
    true = []
    total_time = 0.0
    n_spikes = 0

    for data_ in loader:
        #printing and then deleting one charecter is a hack to prevent the output from being printed in the console
        # print('..', end='', flush=True)
        # print('\b', end='', flush=True)

        #wait for 0.01 seconds to prevent the output from being printed in the console
        #todo: this is a temporary solution, find a better way to prevent the freezes
        time.sleep(0.01)
        if use_padding_mask:
            if record_spike_count:
                n_spikes += np.sum(np.logical_not(data_['padding_mask'].numpy()))

            data = data_['data'].to(device)
            padding_mask = data_['padding_mask'].to(device)

            if args.supervision_mode == 'supervised':
                label = data_['label'].to(device).squeeze()
        else:
            # print('debug data_:', np.shape(data_[0]))
            data, label = data_[0].to(device), data_[1].to(device).squeeze()

        batch_size = data.size()[0]

        if use_padding_mask:
            padding_mask = data_['padding_mask'].to(device)

        if args.flatten_contrastive_input and args.supervision_mode == 'contrastive':
            data = data.reshape(batch_size*2, *data.shape[2:])

        if mode == 'train':
            opt.zero_grad()

        start_time = time.time()

        model_output = model(data, padding_mask=padding_mask) if use_padding_mask else model(data)

        if args.supervision_mode == 'supervised':
            logits = model_output
            loss_batch = criterion(logits, label)
        else:  # args.supervision_mode == 'contrastive'
            projections = model_output
            if args.flatten_contrastive_input:
                projections = projections.view(batch_size, 2, *projections.shape[1:])
            z_i, z_j = projections.unbind(dim=1)
            loss_batch = criterion(z_i, z_j)

        if mode == 'train':
            loss_batch.backward()
            opt.step()

        end_time = time.time()
        total_time += (end_time - start_time)

        count += batch_size
        loss += loss_batch.item() * batch_size

        if args.supervision_mode == 'supervised':
            preds = logits.max(dim=1)[1]
            true.append(label.cpu().numpy())
            pred.append(preds.detach().cpu().numpy())

    if args.supervision_mode == 'supervised':
        true = np.concatenate(true)
        pred = np.concatenate(pred)
        acc = metrics.accuracy_score(true, pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
    else:
        acc = None
        avg_per_class_acc = None
    result_prefix = result_prefix if result_prefix is not None else mode
    epoch_data = {
        'epoch': epoch,
        f'{result_prefix}_loss': loss * 1.0 / count,
        f'{result_prefix}_acc': acc,
        f'{result_prefix}_avg_acc': avg_per_class_acc,
        f'{result_prefix}_time': total_time
    }

    if mode == 'train':
        epoch_data['lr'] = scheduler.get_last_lr()

    if record_spike_count:
        epoch_data[f'{result_prefix}_avg_spike_count'] = n_spikes / count

    return epoch_data


def train(args, io, prepare_objects_and_return=False):

    shuffle_order = None
    if args.resume_run_name is not None:
        checkpoint = torch.load('checkpoints/%s/models/last_model.t7' % args.resume_run_name)
        if 'shuffle_order' in checkpoint:
            shuffle_order = checkpoint['shuffle_order']


    train_loader, val_loader, use_padding_mask, record_spike_count, shuffle_order = create_dataloaders(args, shuffle_order=shuffle_order)

    if args.en_autodetect_offsets:
        #pass through the dataset to compute offsets
        args.model_args['offsets'] = -compute_mean_offsets(train_loader)
        print('manually computed offsets', args.model_args['offsets'])

    model = create_model(args, io)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {total_params}")

    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    if not hasattr(args,'disable_data_parallel') or not args.disable_data_parallel:
        model = nn.DataParallel(model)

    opt, scheduler = create_optimizer(model, args)
    criterion = nn.CrossEntropyLoss() if args.supervision_mode == 'supervised' else NTXentLoss(device=device, **args.loss_args)

    best_val_acc = 0
    if args.resume_run_name is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc']
        args.start_epoch = checkpoint['epoch'] + 1
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state'])
        random.setstate(checkpoint['python_random_state'])

    if prepare_objects_and_return:
        return model, opt, scheduler, criterion, train_loader, val_loader, device, best_val_acc, args, io

    #todo: define a behavior for a case where no more epochs are left
    for epoch in range(args.start_epoch, args.epochs):
        train_data = run_epoch(model, train_loader, criterion, opt, device, epoch, mode='train',
                               use_padding_mask=use_padding_mask, record_spike_count=record_spike_count,
                               scheduler=scheduler,args=args)
        val_data = run_epoch(model, val_loader, criterion, opt, device, epoch, mode='val',
                             use_padding_mask=use_padding_mask, record_spike_count=record_spike_count,args=args)

        epoch_data = {**train_data, **val_data}

        outstr = create_output_string(epoch_data,args, record_spike_count=record_spike_count)

        write_to_json(args.json_log, epoch_data)
        io.print_and_log(outstr)
        scheduler.step()

        #checkpointing: save the best model and the last model
        #before saving the last model remove empty file 'last_model_saved' if it exists
        #after saving the last model create an empty file 'last_model_saved'
        if val_data['val_acc'] is not None and val_data['val_acc'] >= best_val_acc:
            best_val_acc = val_data['val_acc']
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % args.run_name)

        if os.path.exists('checkpoints/%s/models/last_model_saved' % args.run_name):
            os.remove('checkpoints/%s/models/last_model_saved' % args.run_name)

        full_state_cp = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            # Save the random generator states
            'np_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all(),
            'python_random_state': random.getstate(),
            'shuffle_order': shuffle_order if shuffle_order is not None else 'NOT_SAVED'
        }
        torch.save(full_state_cp, 'checkpoints/%s/models/last_model.t7' % args.run_name)
        open('checkpoints/%s/models/last_model_saved' % args.run_name, 'w').close()

    #in case of evaluation mode run one epoch and exit
    if args.eval:
        eval_loader, use_padding_mask, record_spike_count = create_dataloaders(args, do_subsplits=False,  eval=True)
        test_data = run_epoch(model, eval_loader, criterion, opt, device, args.start_epoch, mode='val', result_prefix='test',
                             use_padding_mask=use_padding_mask, record_spike_count=record_spike_count,args=args)
        # outstr = create_output_string(val_data, args, record_spike_count=record_spike_count)
        io.print_and_log('\n Test results  '.join([f'{k}: {v}, ' for k,v in test_data.items()]))
        write_to_json(args.json_test_data, test_data)

if __name__ == "__main__":

    args = parser.parse_args()
    # parse_strings_to_lists(args)

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

    # write a file that contains the last checkpoint path
    with open(args.save_path + args.run_name + '/last_checkpoint_path', 'w') as file:
        file.write('checkpoints/%s/models/last_model.t7' % args.run_name)
    #in the lsf mode add '>>> DONE' to the lsf_manager/lsf_top.<job_id> file
    if args.check_job_recovery:
        with open('lsf_manager/lsf_top.' + args.job_id, 'a') as file:
            file.write('>>> DONE\n')
    # write an empty file to indicate that the run has finished
    open(args.save_path + args.run_name + '/run_finished', 'w').close()
    io.print_and_log('Run finished')