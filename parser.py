"""
Parser module for command-line arguments and configuration parsing for the project.
"""

import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='event based classifier')

# configuration file
parser.add_argument('--config_file', type=str,
                    help='config file path')

# command line arguments (overrides config file, as implemented in eb_train.py)
# result saving params
parser.add_argument('--save_path', type=str,
                    help='where to save this run data')
parser.add_argument('--run_name', type=str, metavar='N',
                    help='Name of the experiment')
parser.add_argument('--job_id_suffix', type=str, metavar='N', default='',
                    help='job id suffix')
parser.add_argument('--tag', type=str, metavar='N',
                    help='tag for the run, does not have to be unique')
# model params
parser.add_argument('--model', type=str,
                    help='which model you want to use')
parser.add_argument('--model_args_model_head', type=str, choices=['cls_mlp', 'cls_avgpool','proj_avgpool'],
                    help='model head')
parser.add_argument('--model_args_dropout_rate', type=float,
                    help='dropout rate')
parser.add_argument('--model_args_num_classes', type=int,
                    help='number of categories in the dataset')
parser.add_argument('--model_args_n_layers', type=int,
                    help='number of transformer layers')
parser.add_argument('--model_args_num_heads', type=int,
                    help='number of heads in the transformer')
parser.add_argument('--model_args_d_k', type=int,
                    help='dimension of key')
parser.add_argument('--model_args_drop_subseq_prob', type=float,
                    help='probability of removing a subsequence in the transformer with subsequences')

parser.add_argument('--model_args_model_head_init_method', type=str,
                    help='model head initialization method')

# add boolean model_args_extra_dim_for_subsamples
parser.add_argument('--model_args_extra_dim_for_subsamples', type=bool,
                    help='add extra dim for subsamples')
parser.add_argument('--en_model_args_extra_dim_for_subsamples', action='store_true',
                    dest='model_args_extra_dim_for_subsamples')
parser.add_argument('--no_model_args_extra_dim_for_subsamples', action='store_false',
                    dest='model_args_extra_dim_for_subsamples')
# add boolean ds_args_extra_dim_for_subsamples
parser.add_argument('--ds_args_extra_dim_for_subsamples', type=bool,
                    help='add extra dim for subsamples')
parser.add_argument('--en_ds_args_extra_dim_for_subsamples', action='store_true',
                    dest='ds_args_extra_dim_for_subsamples')
parser.add_argument('--no_ds_args_extra_dim_for_subsamples', action='store_false',
                    dest='ds_args_extra_dim_for_subsamples')

# dataset params
parser.add_argument('--dataset', type=str, metavar='N', choices=['modelnet40', 'eb_ds'])
parser.add_argument('--eb_ds_path', type=str,
                    help='path to event base dataset')
parser.add_argument('--ds_train_subdir', type=str, default='train',
                    help='train subdir')
parser.add_argument('--ds_test_subdir', type=str, default='test',
                    help='test subdir')

parser.add_argument('--eb_ds_ts_jitter', type=float,
                    help='add jitter to event timestamp at ds loading')
parser.add_argument('--eb_ds_imu_channels', type=str,
                    help='imu channels')

parser.add_argument('--ds_args_n_samples', type=int,
                    help='num of points to use')
parser.add_argument('--ds_args_time_limit', type=float,
                    help='max time from the initial event to the last event')
parser.add_argument('--num_workers', type=int,
                    help='num of workers to use')
parser.add_argument('--relabel_opt', type=str,
                    help='relabel option')

#parameters for handling tonic dataset: eb_ds_tonic_name, eb_ds_tonic_path:
parser.add_argument('--eb_ds_tonic_name', type=str,
                    help='name of tonic dataset')
parser.add_argument('--eb_ds_tonic_path', type=str,
                    help='path to tonic dataset')

# parser.add_argument('--batched_ds', type=bool,
#                     help='use batched dataset')
#batch size:
parser.add_argument('--batch_size', type=int,
                    help='batch size')
parser.add_argument('--model_args_d_timeseries', type=int,
                    help='dimension of pointcloud. Use -1 for default')
parser.add_argument('--ds_args_shuffle_events', type=bool,
                    help='shuffle events within a data sample')
parser.add_argument('--n_validation', type=int,
                    help='number of validation samples')
# passing intervals via parser is done using strings that are later converted to lists
parser.add_argument('--ds_args_train_from_time_interval', type=str,
                    help='train from time interval')
parser.add_argument('--ds_args_val_from_time_interval', type=str,
                    help='validate from time interval')

# passing augmentation parameters via parser is done using strings that are later converted to lists
# add parameters, augment_whole_shifts, augment_per_event_shifts:
parser.add_argument('--ds_args_train_augment_whole_shifts', type=str,
                    help='augment whole shifts')
parser.add_argument('--ds_args_train_augment_per_event_shifts', type=str,
                    help='augment per event shifts')
parser.add_argument('--ds_args_test_augment_whole_shifts', type=str,
                    help='augment whole shifts')
parser.add_argument('--ds_args_test_augment_per_event_shifts', type=str,
                    help='augment per event shifts')
parser.add_argument('--ds_args_augment_whole_shifts', type=str,
                    help='augment whole shifts')
parser.add_argument('--ds_args_augment_per_event_shifts', type=str,
                    help='augment per event shifts')

# loss params
parser.add_argument('--loss_args_temperature', type=float,
                    help='temperature for contrastive loss')

parser.add_argument('--loss_args_p_drop_negative_samples', type=float,
                    help='probability of dropping negative samples in contrastive loss')


# training params
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool,
                    help='Use SGD')
parser.add_argument('--en_warmup', action='store_true',
                    dest='do_warmup')
parser.add_argument('--no_warmup', action='store_false',
                    dest='do_warmup')

parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--opt_args_eps', type=float, metavar='N',
                    help='Adam epsilon')


# evaluation params
parser.add_argument('--pretrained_model_path', type=str, metavar='N',
                    help='Pretrained model path')
#pretrained_model_no_head
parser.add_argument('--pretrained_model_no_head', type=bool,
                    help='Pretrained model path')
parser.add_argument('--en_pretrained_model_no_head', action='store_true',
                    dest='pretrained_model_no_head')
parser.add_argument('--no_pretrained_model_no_head', action='store_false',
                    dest='pretrained_model_no_head')

#train head only
parser.add_argument('--train_head_only', type=bool,
                    help='train head only')
parser.add_argument('--en_train_head_only', action='store_true',
                    dest='train_head_only')
parser.add_argument('--no_train_head_only', action='store_false',
                    dest='train_head_only')

#setting eval mode
parser.add_argument('--eval', type=bool,
                    help='evaluate the model')
parser.add_argument('--en_eval', action='store_true',
                    dest='eval')
parser.add_argument('--no_eval', action='store_false',
                    dest='eval')

#contrastive learning params
parser.add_argument('--flatten_contrastive_input', type=bool,
                    help='flatten contrastive input')
parser.add_argument('--en_flatten_contrastive_input', action='store_true',
                    dest='flatten_contrastive_input')
parser.add_argument('--no_flatten_contrastive_input', action='store_false',
                    dest='flatten_contrastive_input')

#check_job_recovery params
parser.add_argument('--check_job_recovery', type=bool,
                    help='check job recovery')
parser.add_argument('--en_check_job_recovery', action='store_true',
                    dest='check_job_recovery')
parser.add_argument('--no_check_job_recovery', action='store_false',
                    dest='check_job_recovery')

#sgd on/off
# parser.add_argument('--use_sgd', type=bool,
#                     help='Use SGD')
parser.add_argument('--en_use_sgd', action='store_true',
                    dest='use_sgd')
parser.add_argument('--no_use_sgd', action='store_false',
                    dest='use_sgd')

#disable data parallel
# parser.add_argument('--disable_data_parallel', type=bool,
#                     help='disable data parallel')
parser.add_argument('--disable_data_parallel', action='store_true',
                    dest='disable_data_parallel')
parser.add_argument('--enable_data_parallel', action='store_false',
                    dest='disable_data_parallel')

#ds_args_shuffle_events
parser.add_argument('--en_ds_args_shuffle_events', action='store_true',
                    dest='ds_args_shuffle_events')
parser.add_argument('--no_ds_args_shuffle_events', action='store_false',
                    dest='ds_args_shuffle_events')

#eb_ds_time_align_by_imu_edge
parser.add_argument('--en_eb_ds_time_align_by_imu_edge', action='store_true',
                    dest='eb_ds_time_align_by_imu_edge')
parser.add_argument('--no_eb_ds_time_align_by_imu_edge', action='store_false',
                    dest='eb_ds_time_align_by_imu_edge')

#eb_ds_time_align_by_imu_edge
parser.add_argument('--en_ds_args_one_hot_coordinates', action='store_true',
                    dest='ds_args_one_hot_coordinates')
parser.add_argument('--no_ds_args_one_hot_coordinates', action='store_false',
                    dest='ds_args_one_hot_coordinates')

#en_autodetect_offsets
parser.add_argument('--en_autodetect_offsets', action='store_true',
                    dest='en_autodetect_offsets')
parser.add_argument('--no_autodetect_offsets', action='store_false',
                    dest='en_autodetect_offsets')

#default values for boolean arguments
parser.set_defaults(pretrained_model_no_head=False)
parser.set_defaults(train_head_only=False)
parser.set_defaults(flatten_contrastive_input=False)
parser.set_defaults(use_sgd=False)
parser.set_defaults(no_cuda=False)
parser.set_defaults(check_job_recovery=False)
parser.set_defaults(eval=True)
parser.set_defaults(disable_data_parallel=False)
parser.set_defaults(do_warmup=True)
parser.set_defaults(eb_ds_time_align_by_imu_edge=False)
parser.set_defaults(eb_ds_ts_jitter=0.0)
parser.set_defaults(ds_args_one_hot_coordinates=False)
parser.set_defaults(en_autodetect_offsets = False)


