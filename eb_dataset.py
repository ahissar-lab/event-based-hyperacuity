import os
import glob
import pickle
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle


class EB_DS(Dataset):
    '''
    Data Generator for Event Based Data
    input:
        data: list of events matrices
        labels: list of labels
        n_samples: number of samples to take from each event vector
        start_index: index to start from in each event vector
        start_time: time to start from in each event vector
        from_time_interval: list or tuple representing the time interval to sample from
        n_subsamples: number of samples to take from the time interval
        shuffle_events: shuffle events within a data sample
        add_parity_channel1: add a channel with the parity of the 1st spatial coordinate
        add_parity_channel2: add a channel with the parity of the 2nd spatial coordinate
        periodic_time_encoding: list of periods to encode the time with. None for no encoding.
        extra_dim_for_subsamples: whether to add an extra dimension for the subsamples. If false all the subsamples are concatenated.
            If true the subsamples are stacked along the first dimension.
        augment_whole_shifts: list of std of random gaussian shifts (of length 4 + optional imu data) to apply to a sample as a whole
        augment_per_event_shifts: list of std of random gaussian shifts (of length 4 + optional imu data) to apply to each event
        frame_based: whether the data is frame based (this option is used with standard, frame based data)
    '''
    def __init__(self, data, labels,
                 n_samples=None,
                 start_index=None,
                 start_time=None,
                 time_limit=None,
                 from_time_interval=None,
                 n_subsamples=None,
                 shuffle_events=True,
                 shuffle_timestamps=False,
                 add_parity_channel1=False,
                 add_parity_channel2=False,
                 periodic_time_encoding=None,
                 extra_dim_for_subsamples=False,
                 augment_whole_shifts=None,
                 augment_per_event_shifts=None,
                 one_hot_coordinates=False,
                 frame_based=False,
                 offsets=None,
                 scalings=None,
                 ):
        if not frame_based:
            self.data = [zero_pad(d, n_samples=n_samples) for d in data]
        else:
            self.data = data
        self.labels = labels
        self.n_samples = n_samples
        self.n_subsamples = n_subsamples
        self.start_index = start_index
        self.start_time = start_time
        self.time_limit = time_limit
        self.from_time_interval = from_time_interval
        self.shuffle_events = shuffle_events
        self.add_parity_channel1 = add_parity_channel1
        self.add_parity_channel2 = add_parity_channel2
        self.periodic_time_encoding = periodic_time_encoding
        self.extra_dim_for_subsamples = extra_dim_for_subsamples
        self.augment_whole_shifts = augment_whole_shifts
        self.augment_per_event_shifts = augment_per_event_shifts
        self.one_hot_coordinates = one_hot_coordinates
        self.frame_based = frame_based
        self.offsets = offsets
        self.scalings = scalings

        if self.n_subsamples is not None and self.time_limit is not None:
            raise ValueError("only one of n_subsamples and time_limit can be specified")

        if self.from_time_interval is not None:
            # Select starting from a time interval
            #todo: treat intervals with insufficient data
            if self.n_subsamples is None:
                self.n_subsamples = [self.n_samples]
                self.from_time_interval = [self.from_time_interval]

        self.return_padding_mask = self.time_limit is not None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        # a hook for handling frame based data
        if self.frame_based:
            x = np.squeeze(x[-1]['frame'])/255.
            return x.astype(np.float32), y

        if self.start_index is not None:
            # Select starting from a fixed index
            start = min(self.start_index, len(x) - self.n_samples)
        elif self.start_time is not None:
            # Select starting from a time value
            start = np.searchsorted(np.array(x)[:, 0], self.start_time)
            start = min(start, len(x) - self.n_samples)
        elif self.from_time_interval is not None:
            x_selected = select_samples_from_intervals(x, self.n_subsamples, self.from_time_interval, extra_dim_for_subsamples=self.extra_dim_for_subsamples)
            # print('x_selected.shape: ', x_selected.shape)
        else:
            # Select starting from a random position
            start = np.random.randint(len(x) - self.n_samples) if len(x) > self.n_samples else 0

        if self.from_time_interval is None:
            x_selected = x[start:start + self.n_samples]
        else:
            pass #in case of time intervals the data is already selected

        if self.shuffle_events:
            # Shuffling along the t dimension for each sample
            x_selected = shuffle(x_selected)

        if self.augment_whole_shifts is not None:
            # if not None, the augment_whole_shifts is a list of std of random gaussian shifts (of length 4) to apply to a sample as a whole
            x_selected = x_selected + np.random.normal(loc=0.0, scale=self.augment_whole_shifts, size=(1,4))

        if self.augment_per_event_shifts is not None:
            # if not None, the augment_per_event_shifts is a list of std of random gaussian shifts (of length 4) to apply to each event
            x_selected = x_selected + np.random.normal(loc=0.0, scale=self.augment_per_event_shifts, size=(self.n_samples, 4))

        if self.add_parity_channel1:
            # Adding a channel with the parity of the x coordinate
            x_selected = np.concatenate([x_selected, np.expand_dims(np.mod(x_selected[:, 1], 2), axis=1)], axis=1)
        if self.add_parity_channel2:
            # Adding a channel with the parity of the y coordinate
            x_selected = np.concatenate([x_selected, np.expand_dims(np.mod(x_selected[:, 2], 2), axis=1)], axis=1)

        if self.periodic_time_encoding is not None:
            # Adding a channels with the periodic encoding of the time constants given by elements of periodic_time_encoding
            x_selected = np.concatenate([x_selected, np.stack([np.sin(2 * np.pi * x_selected[:, 0] / t) for t in self.periodic_time_encoding], axis=1)], axis=1)

        if self.time_limit is not None:
            # ensure that all the events are within the time limit with respect to the first event
            # create a mask for the events that are outside the time limit
            mask = (x_selected[:, 0] - x_selected[0, 0]) > self.time_limit
            return {'data': x_selected.astype(np.float32), 'padding_mask':mask, 'label': y}

        if self.one_hot_coordinates:
            #expand coordinates 1,2 into one hot encoding
            x_selected_spatial = xy_to_one_hot(x_selected[:,1:3],
                                       one_hot_grid_size=[25,10],
                                       one_hot_bin_size=[1,1],
                                       one_hot_offset=[10.5,20.5],
                                       overflow_handling="nullify")
            #collapse into a 1D array
            x_selected_spatial = np.reshape(x_selected_spatial, (x_selected_spatial.shape[0], -1))
            # print('x_selected_spatial.shape: ', x_selected_spatial.shape)
            x_selected = np.concatenate([x_selected[:,:1], x_selected_spatial, x_selected[:,3:]], axis=1)

        if self.offsets is not None:
            x_selected = x_selected + self.offsets

        if self.scalings is not None:
            x_selected = x_selected * self.scalings

        return x_selected.astype(np.float32), y



class EB_DS_batched(Dataset):
    # Data Generator for Event Based Data
    def __init__(self, data, labels,
                 batch_size,
                 shuffle_t=True,
                 n_samples=None,
                 start_index=None,
                 start_time=None,
                 shuffle_samples=True,
                 ):

        if (start_index is not None) and (start_time is not None):
            raise ValueError("only one of start_index and start_time can be specified")

        self.data = data
        self.labels = [int(uu) for uu in labels]
        self.batch_size = batch_size
        self.shuffle_t = shuffle_t
        self.indices = np.arange(len(data))
        self.n_samples = n_samples
        self.start_index = start_index
        self.start_time = start_time
        self.shuffle_samples = shuffle_samples
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x_ = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = []
        for x in batch_x_:
            if self.start_index is not None:
                ii = min(self.start_index, len(x) - self.n_samples)
            elif self.start_time is not None:
                ii = np.searchsorted(np.array(x)[:, 0], self.start_time)
                ii = min(ii, len(x) - self.n_samples)
            else:
                ii = np.random.randint(len(x) - self.n_samples) if len(x) > self.n_samples else 0
            batch_x.append(x[ii:ii + self.n_samples])

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(self.labels[index * self.batch_size:(index + 1) * self.batch_size])

        if self.shuffle_t:
            # Shuffling along the t dimension for each sample
            batch_x = np.array([shuffle(x) for x in batch_x], dtype=np.float32)

        x_tensor = torch.tensor(batch_x, dtype=torch.float32)
        y_tensor = torch.tensor(batch_y, dtype=torch.long)
        # return batch_x, batch_y
        return x_tensor, y_tensor

    def on_epoch_end(self):
        # Shuffle indices for the next epoch
        if self.shuffle_samples:
            shuffle_lists_in_unison(self.data, self.labels)


def shuffle_lists_in_unison(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    list1[:], list2[:] = zip(*combined)

def shuffle_lists_in_unison3(list1, list2, list3):
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    list1[:], list2[:], list3[:] = zip(*combined)

def shuffle_list_by_indices(list1, indices):
    list1[:] = [list1[i] for i in indices]

def rescale_ds(ds, scaling=[1e-3, 1, 1, 1]):
    if np.shape(ds[0])[1] > 4:
        #pad the scaling with ones
        scaling = scaling + [1] * (np.shape(ds[0])[1] - 4)
        #issue warning
        print('WARNING: scaling vector is too short, padding with ones...')
    return [d * [scaling] for d in ds]

def jitter_timestamp(ds, jitter_ts=0.0):
    for events in ds:
        # events[:,0] += np.random.normal(loc=0.0, scale=jitter_ts, size = events.shape[0]).astype(np.dtype('float64'))
        events[:,0] += np.random.normal(loc=0.0, scale=jitter_ts, size = events.shape[0])

def zero_pad(d,n_samples=None):
    if len(d) >= n_samples:
        return d
    else:
        d_ = np.zeros((n_samples,np.shape(d)[1]))
        d_[:len(d)] = d
        return d_
def extract_imu_record(imu_data, imu_channels, imu_name_prefix, target_timestamps):
    # inputs:
    # imu data organized as numpy array with typical columns being of following types
    # dtype=[('timestamp', '<i4'), ('gyroscope0', '<f4'), ('gyroscope1', '<f4'), ('gyroscope2', '<f4'), ('accelerometer0', '<f4'), ('accelerometer1', '<f4'), ('accelerometer2', '<f4')])
    # imu_channels being e.g. [0,1,2] for gyroscope
    # imu_name_prefix being e.g. 'gyroscope'
    # target_timestamps for which imu data is to be extracted
    # output:
    # imu_rec: list of len(imu_channels) numpy arrays, each array of the same length as target_timestamps
    # with the corresponding imu data extracted from the imu_data
    # imu data is assigned to the target timestamps by the closest imu timestamps

    imu_rec = []
    #for each target timestamp find the closest imu timestamps
    imu_timestamps = imu_data[:]['timestamp']
    #create distance matrix
    dist_mat = np.abs(imu_timestamps[:,None] - target_timestamps[None,:])
    #find the closest imu timestamps
    closest_imu_timestamps = imu_timestamps[np.argmin(dist_mat, axis=0)]
    #extract the imu data for the closest timestamps
    for cc in imu_channels:
        imu_rec.append(imu_data[:][f'{imu_name_prefix}{cc}'][np.searchsorted(imu_timestamps, closest_imu_timestamps)])
    return imu_rec


def load_dataset(base_path=None, preprocess=True, n_samples=None, shuffle_order=None, do_shuffle=True, jitter_ts=0.0, 
                 field_names=['timestamp', 'x', 'y', 'polarity'],
                 imu_channels=None, imu_name_prefix='gyroscope',
                 time_align_by_imu_edge=False,
                 time_align_by_imu_edge_threshold=-15,
                 time_align_by_imu_edge_dir='down',
                 time_align_by_imu_edge_channel=1,
                 time_align_by_imu_edge_centered = True,
                 tonic_name=None, tonic_train_split=None, tonic_path=None, tonic_field_names=['t','x', 'y', 'p'],
                 frame_based=False):
    events_lst = []
    labels_lst = []
    time_edge_lst = []

    #check that only one of base_path and tonic_name is specified
    if base_path is not None and tonic_name is not None:
        raise ValueError("only one of base_path and tonic_name can be specified")

    if tonic_name is not None and imu_channels is not None:
        raise ValueError("imu_channels are not supported for tonic datasets")

    if tonic_name is not None and time_align_by_imu_edge:
        raise ValueError("time_align_by_imu_edge is not supported for tonic datasets")

    #ensure frame_based is False for tonic datasets
    if tonic_name is not None and frame_based:
        raise ValueError("frame_based is not supported for tonic datasets")

    if frame_based:
        ds, labels = load_dataset_frames(base_path)

    else: #event based dataset
        if base_path is not None:
            base_path += '/' if base_path[-1] != '/' else ''

            events_file_path = os.path.join(base_path, "events.pkl")
            if not os.path.isfile(events_file_path):
                raise Exception(f'ERROR: couldn\'t find events file at {events_file_path}')
            print('loading ' + events_file_path + '...')
            with open(events_file_path, 'rb') as f:
                events = pickle.load(f)

            if (imu_channels is not None) or time_align_by_imu_edge:
                imu_file_path = os.path.join(base_path, "imu.pkl")
                if not os.path.isfile(imu_file_path):
                    raise Exception('ERROR: couldn\'t find imu file')
                print('loading ' + imu_file_path + '...')
                with open(imu_file_path, 'rb') as f:
                    imu = pickle.load(f)

                # check that the number of events and imu samples match
                if len(events) != len(imu):
                    raise ValueError("number of events and imu samples do not match")

            for ee, events_sym in enumerate(events):
                timestamps = events_sym[:][field_names[0]]
                imu_rec = []
                if imu_channels is not None:
                    imu_rec = extract_imu_record(imu[ee], imu_channels, imu_name_prefix, timestamps)
                    if ee % 5000 == 0:
                        print(f'extracted imu data for {ee} samples')

                if time_align_by_imu_edge:
                    timestamps_imu = imu[ee][:][field_names[0]]
                    imu_speed = imu[ee][f'{imu_name_prefix}{time_align_by_imu_edge_channel}']
                    imu_edge_idx = find_threshold_crossings(imu_speed, time_align_by_imu_edge_threshold, direction=time_align_by_imu_edge_dir)
                    imu_edge = timestamps_imu[imu_edge_idx[0]] if imu_edge_idx is not None else np.nan
                    # print(f'debug: imu_edge: {imu_edge}')
                    # if ee > 100:
                    #     break
                    time_edge_lst.append(imu_edge)

                events_mat = np.column_stack([timestamps,
                                              events_sym[:]['x'], events_sym[:]['y'], events_sym[:][field_names[3]]] + imu_rec)
                events_lst.append(events_mat)

            if time_align_by_imu_edge:
                #clean and center time_edge_lst
                time_edge_lst = np.array(time_edge_lst)
                mean_t = np.nanmean(time_edge_lst)
                print(f'mean imu edge timestamp: {mean_t}')
                #replace nans with the mean
                time_edge_lst[np.isnan(time_edge_lst)] = mean_t
                if time_align_by_imu_edge_centered:
                    time_edge_lst = time_edge_lst - mean_t
                for ee, events_mat in enumerate(events_lst):
                    events_mat[:,0] = events_mat[:,0] - time_edge_lst[ee]

            file_base_name = os.path.basename(events_file_path)
            dirname = os.path.dirname(events_file_path)

            lables_file_path = dirname + '/' + 'labels' + file_base_name[6:]
            print(lables_file_path)
            if os.path.isfile(lables_file_path):
                with open(lables_file_path, 'rb') as f:
                    labels_ = pickle.load(f)
                labels_lst.append(labels_)
            else:
                raise Exception('ERROR: couldn\'t find lables file')
        elif tonic_name is not None:
            import tonic #importing only if needed
            if tonic_name == 'NMNIST':
                tonic_dataset = tonic.datasets.NMNIST(save_to=tonic_path, train=tonic_train_split)
                for datum in tonic_dataset:
                    events_lst.append(np.column_stack([datum[0][:][ff] for ff in tonic_field_names]))
                    labels_lst.append(datum[1])
            else:
                raise Exception(f'ERROR: tonic dataset {tonic_name} is not supported yet')

        labels = np.concatenate(labels_lst, axis=0) if np.ndim(labels_lst[0]) > 0 else np.array(labels_lst) #todo: check if needed
        ds = events_lst

        if preprocess:
            ds = rescale_ds(ds)
            if jitter_ts != 0.0:
                jitter_timestamp(ds, jitter_ts)
            ds = [zero_pad(d, n_samples=n_samples) for d in ds]

    if shuffle_order is None:
        labels = labels.tolist()
        shuffle_order = [i for i in range(len(labels))]
        if do_shuffle:
            shuffle_lists_in_unison3(ds, labels, shuffle_order)
    else: #shuffle the dataset and labels according to the shuffle_order
        shuffle_list_by_indices(ds, shuffle_order)
        shuffle_list_by_indices(labels, shuffle_order)

    return ds, labels, shuffle_order


def load_dataset_frames(base_path='./'):
    base_path += '/' if base_path[-1] != '/' else ''
    print('loading dataset from ' + base_path + '...')
    files_path = glob.glob(base_path + "frames*.pkl", recursive=True)
    print(files_path)
    print(len(files_path), ' ds files found.')

    frames_lst_all = []
    labels_lst_all = []
    for frames_file_path in files_path:
        print('loading ' + frames_file_path + '...')

        frames = []
        with open(frames_file_path, 'rb') as f:
            frames = pickle.load(f)

        [frames_lst_all.append(frame) for frame in frames]

        file_base_name = os.path.basename(frames_file_path)
        dirname = os.path.dirname(frames_file_path)

        lables_file_path = dirname + '/' + 'labels' + file_base_name[6:]
        print(lables_file_path)
        if os.path.isfile(lables_file_path):
            with open(lables_file_path, 'rb') as f:
                labels_ = pickle.load(f)
            labels_lst_all.append(labels_)
        else:
            raise Exception('ERROR: couldn\'t find lables file')

    labels = np.concatenate(labels_lst_all, axis=0)

    return frames_lst_all, labels

import numpy as np

def xy_to_one_hot(xy, one_hot_grid_size, one_hot_bin_size, one_hot_offset, overflow_handling="nullify"):
    # inputs:
    # xy: numpy array of shape (n_samples, 2) with x and y coordinates
    # one_hot_grid_size: size of the one hot grid, i.e. how many bins in each direction
    # one_hot_bin_size: size of the one hot bin,
    # one_hot_offset: offset of the one hot grid
    # overflow_handling: how to handle out-of-bound elements ('discard', 'clip', 'raise', 'nullify')
    # output:
    # one_hot: numpy array of shape (n_samples, one_hot_grid_size, one_hot_grid_size) with one hot encoding of the xy coordinates

    # Calculate bin indices for x and y coordinates
    bins = ((xy - one_hot_offset) / one_hot_bin_size).astype(int)

    if overflow_handling == "raise":
        if np.any(bins < 0) or np.any(bins >= one_hot_grid_size):
            raise ValueError("Some coordinates exceed the bin boundaries.")

    elif overflow_handling == "clip":
        bins = np.clip(bins, 0, one_hot_grid_size - 1)

    elif overflow_handling == "discard":
        # Mask to keep valid bins only
        valid_mask = (bins >= 0) & (bins < one_hot_grid_size)
        valid_mask = valid_mask.all(axis=1)
        bins = bins[valid_mask]
        xy = xy[valid_mask]

    elif overflow_handling == "nullify":
        # Mask to zero invalid bins
        valid_mask = (bins >= 0) & (bins < one_hot_grid_size)
        valid_mask = valid_mask.all(axis=1)
        bins[~valid_mask] = 0

    else:
        raise ValueError("Invalid value for overflow_handling. Choose 'discard', 'clip', or 'raise'.")

    # Initialize the one_hot array
    one_hot = np.zeros((xy.shape[0], one_hot_grid_size[0], one_hot_grid_size[1]))

    # Use advanced indexing to set the appropriate positions to 1
    one_hot[np.arange(xy.shape[0]), bins[:, 0], bins[:, 1]] = 1

    return one_hot


def select_samples_from_intervals(x, n_samples_list, time_intervals, extra_dim_for_subsamples=False):
    """
    Selects random samples from specified intervals in the data array.

    Parameters:
    x (array-like): The data array to sample from.
    n_samples_list (list of int): List of numbers of samples to take from each interval.
    time_intervals (list of tuple): List of time intervals from which to sample.
    extra_dim_for_subsamples (bool): Whether to add an extra dimension for the subsamples. If false all the subsamples are concatenated.
    If true the subsamples are stacked along the first dimension.

    Returns:
    list: Concatenated list of selected samples from each interval.
    """
    x_selected = []
    # print('x.shape: ', np.shape(x))
    for n_samples, interval in zip(n_samples_list, time_intervals):
        start = np.searchsorted(np.array(x)[:, 0], interval[0])
        end = np.searchsorted(np.array(x)[:, 0], interval[1])
        start = min(start, len(x) - n_samples)
        start = np.random.randint(start, end - n_samples) if end - start > n_samples else start
        this_x_selected = x[start:start + n_samples]
        # print('this_x_selected.shape: ', np.shape(this_x_selected))
        if extra_dim_for_subsamples:
            x_selected.append(this_x_selected)
        else:
            x_selected.extend(this_x_selected)
        # print('internal x_selected.shape: ', np.shape(x_selected))
    # print('internal concat x_selected.shape: ', np.shape(np.concatenate(x_selected)))
    # return np.concatenate(x_selected)
    return np.array(x_selected)

#todo: put into utils
def find_threshold_crossings(vector, threshold, direction='up'):
    """
    Find the index of the first threshold crossing in the specified direction.

    Parameters:
    vector (numpy.ndarray): Input vector.
    threshold (float): The threshold value.
    direction (str): 'up' for crossing from below, 'down' for crossing from above.

    Returns:
    int: The index of the first crossing, or -1 if no crossing is found.
    """

    # Check for 'up' direction (crossing from below)
    if direction == 'up':
        # Find indices where vector crosses threshold from below
        crossings = np.where((vector[:-1] < threshold) & (vector[1:] >= threshold))[0]
    # Check for 'down' direction (crossing from above)
    elif direction == 'down':
        # Find indices where vector crosses threshold from above
        crossings = np.where((vector[:-1] > threshold) & (vector[1:] <= threshold))[0]

    else:
        raise ValueError("Direction must be 'up' or 'down'")

    # Return the first crossing index, or -1 if no crossing is found
    return crossings if len(crossings) > 0 else None


# Usage
# my_dataset = EB_DS(data, labels, n_samples, start_index=start_index, start_time=start_time)
# my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    import time
    n_iter = 100
    t1  = time.time()
    ds, labels, _ = load_dataset(base_path='/shareds/eb_datasets/20230711/', preprocess=True, n_samples=64)
    train_dataset = EB_DS(ds[:-6000], labels[:-6000], n_samples=64)
    train_loader = DataLoader(train_dataset, num_workers=0,
                              batch_size=32, shuffle=True, pin_memory=True)
    t2 = time.time()
    # for i in range(n_iter):
    #     print(train_dataset[i][0].shape, train_dataset[i][1],  train_dataset[i][1].shape)
    for idx, (x, y) in enumerate(train_loader):
        # print(x.shape, y, y.shape)
        if idx > n_iter:
            break
    t3 = time.time()
    print('load time: ', t2-t1)
    print('iter time: ', (t3-t2)/n_iter)