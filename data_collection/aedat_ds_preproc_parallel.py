#import tensorflow as tf
import numpy as np
import pickle
import os
from dv import AedatFile
import multiprocessing
#from multiprocessing.pool import Pool
from itertools import repeat

# returns aeadat file list in a specific directoty, sorted according to symbols indexes
# indicated in file names.
def get_aedat_file_list(f_path='./mnist_dvs/aedat_files/',fn_st='mnist',fn_en='.aedat4'):
    file_list=[]
    file_key_list=[]
    for filename in os.listdir(f_path):
        if filename.startswith(fn_st) and filename.endswith(fn_en):
            file_list.append(str(os.path.join(f_path, filename)))
    print('{} aedat files found.'.format(len(file_list)))
          
    for ind in range(len(file_list)):
        #file_key_list.append(int(file_list[ind].split('/')[-1].split('-')[0].split('_')[1])) 
        file_key_list.append(int(file_list[ind].split('/')[-1].split('-')[0].split('_')[-2])) 
    print(file_key_list)
    d=dict(zip(file_key_list, file_list))

    return d

# extract events, triggers and imu data from an aedat file
def read_aedat_file(fpath):
    events = []
    imus = []
    triggers = []
    
    with AedatFile(fpath) as f:
            # events-> named numpy array
#            events.append(np.hstack([packet for packet in f['events'].numpy()])) 
#            if len(f['events'].numpy())==0:#debug
            # count = 0
            # for packet in f['events']:
            #     count+=1
            # print(fpath, count)
            events = []
            if 'events' in f.names:
                events = np.hstack([packet for packet in f['events'].numpy()])
            
            # triggers-> named numpy array
            lst_trig_data = [(packet.timestamp, packet.type) for packet in f['triggers']]
            trig_dtype = np.dtype({'names': ['timestamp', 'type'], 'formats': ['i8', 'i1']})
            triggers = np.array(lst_trig_data, dtype=trig_dtype)
            
            imus = []
            if 'imu' in f.names:
                # imu-> named numpy array
                lst_imu_data = [(packet.timestamp, packet.gyroscope[0], packet.gyroscope[1], packet.gyroscope[2], 
                                packet.accelerometer[0], packet.accelerometer[1], packet.accelerometer[2], packet.temperature) for packet in f['imu']]
                imu_dtype = np.dtype({'names': ['timestamp', 'gyroscope0', 'gyroscope1', 'gyroscope2', 
                                      'accelerometer0', 'accelerometer1', 'accelerometer2','temperature'], 
                                    'formats': ['i8', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4']})
                imus = np.array(lst_imu_data, dtype=imu_dtype)

            frames = []
            if 'frames' in f.names:
                lst_frames_data = [(frame.timestamp, frame.image) for frame in f['frames']]
                frames_dtype = np.dtype([('timestamp','i8'),('frame',np.uint8, (10,10,1) )])
                frames = np.array(lst_frames_data, frames_dtype)
            
    return events, frames, imus, triggers

# filtering triggers is required when spurios triggers are presented in data (the case with DVS128)
# if trial length calculated by trigger time is shorter then trial_th trigger is deleted.
def filter_triggers(triggers, trial_th):
    trig_filtered = np.zeros_like(triggers)

    filtered_count = 1
    trig_filtered[0] = triggers[0]
    for i in range(1,triggers.shape[0]):
        if triggers['timestamp'][i]-trig_filtered['timestamp'][filtered_count-1] > trial_th*1e3:
            trig_filtered[filtered_count] = triggers[i]
            filtered_count += 1
#        else:
#            print(i)

    trig_filtered = trig_filtered[trig_filtered['timestamp']>0]
#    print(filtered_count)
#    print(trig_filtered.shape)
    assert(filtered_count == 1000)
    return trig_filtered

# Build events list of individual symbols, parsed using triggers timestamps
def parse_events_stream(events_stream, frames_stream, imu_stream, trig_stream, single_trig = False, trial_duration = 300, fileid=0):
    events=[]
    frames=[]
    IMUs=[]
    onset_offset=[]
    
    trigs_ts  = trig_stream['timestamp']    
    
    events_ts = []
    if len(events_stream)>0:
        events_ts = events_stream['timestamp']

    frames_ts = []
    if len(frames_stream)>0:
         frames_ts   = frames_stream['timestamp']    
    
    imus_ts = []
    if len(imu_stream)>0:
         imus_ts   = imu_stream['timestamp']
    
##
    if single_trig:
        sym_acq_onset_ts = trigs_ts[trig_stream['type']==1]
        sym_acq_offset_ts =  sym_acq_onset_ts + int(trial_duration*1e3)
    else:
        sym_acq_onset_ts = trigs_ts[trig_stream['type']==1]
        sym_acq_offset_ts =  trigs_ts[trig_stream['type']==2]
#    # triggers switched 2- onset, 1- offset
#    sym_acq_onset_ts = trigs_ts[trig_stream['type']==2]
#    sym_acq_offset_ts = trigs_ts[trig_stream['type']==1]
#    sym_acq_onset_ts = np.delete(sym_acq_onset_ts, sym_acq_onset_ts.shape[0]-1)
#    sym_acq_offset_ts = np.delete(sym_acq_offset_ts, 0)
##
#    assert(sym_acq_onset_ts.shape == sym_acq_offset_ts.shape)
    if sym_acq_onset_ts.shape != sym_acq_offset_ts.shape:
        print('Warning!, file id: ' + str(fileid) + '. Onset: ' + str(sym_acq_onset_ts.shape) + ', offset: ' + str(sym_acq_offset_ts.shape))

        if sym_acq_onset_ts.shape[0] + 1 == sym_acq_offset_ts.shape[0]:
            sym_acq_offset_ts = sym_acq_offset_ts[1:]
        else:
            sym_acq_onset_ts = sym_acq_onset_ts[:-1]

    sym_onset_offset = np.column_stack((sym_acq_onset_ts, sym_acq_offset_ts))
    
##
#    ev_parse_start_ind = 0
#    imu_parse_start_ind = 0
##
    for symonoff in sym_onset_offset:
##
        if len(events_stream)>0:
            ev_parse_start_ind=np.where(events_ts>=symonoff[0])[0][0]
            if (events_ts>=symonoff[1]).any():
                ev_parse_stop_ind=np.where(events_ts>=symonoff[1])[0][0]
        if len(imu_stream)>0:
            imu_parse_start_ind=np.where(imus_ts>=symonoff[0])[0][0]
            imu_parse_stop_ind=np.where(imus_ts>=symonoff[1])[0][0]
        if len(frames_stream)>0:
            frames_parse_start_ind=np.where(frames_ts>=symonoff[0])[0][0]
            if (frames_ts>=symonoff[1]).any(): #??
                frames_parse_stop_ind=np.where(frames_ts>=symonoff[1])[0][0]
            else: #??
                frames_parse_stop_ind = frames_parse_start_ind+1

##
        if len(events_stream)>0:
            sym_events = events_stream[ev_parse_start_ind:ev_parse_stop_ind]
            sym_events['timestamp'] -= symonoff[0]
            # modifying timestamp type i8->i4 and apending to list
            new_dt_dscr = sym_events.dtype.descr
            new_dt_dscr[0] = ('timestamp', 'i4')
            events.append(sym_events.astype(new_dt_dscr))
            #evn_dtype = np.dtype({'names': ['timestamp', 'x', 'y', 'polarity'], 'formats': ['i4', 'i2', 'i2', 'i1']})
            #events.append(np.array(sym_events[['timestamp','x','y','polarity']], evn_dtype))
        
        if len(imu_stream)>0:
            sym_imu = imu_stream[imu_parse_start_ind:imu_parse_stop_ind]
            sym_imu['timestamp'] -= symonoff[0]
            # modifying timestamp type i8->i4 and apending to list
            new_dt_dscr = sym_imu.dtype.descr
            new_dt_dscr[0] = ('timestamp', 'i4')
            IMUs.append(sym_imu.astype(new_dt_dscr))
            #imu_dtype = np.dtype({'names': ['timestamp', 'gyroscope0', 'gyroscope1', 'gyroscope2', 'accelerometer0', 'accelerometer1', 'accelerometer2'], 
            #                    'formats': ['i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4']})
            #IMUs.append(np.array(sym_imu, imu_dtype))

        if len(frames_stream)>0:
            sym_frames = frames_stream[frames_parse_start_ind:frames_parse_stop_ind]
            sym_frames['timestamp'] -= symonoff[0]
            # modifying timestamp type i8->i4 and apending to list
            new_dt_dscr = sym_frames.dtype.descr
            new_dt_dscr[0] = ('timestamp', 'i4')
            frames.append(sym_frames.astype(new_dt_dscr))
        
        onset_offset.append(symonoff)

##
#        ev_parse_start_ind=np.where(events_ts>=symonoff[1])[0][0]
#        imu_parse_start_ind=np.where(imus_ts>=symonoff[1])[0][0]
##
        
    return events, frames, IMUs, onset_offset

def parse_aedat_file(dict_item, dvs128):
    trial_n, fpath, = dict_item
    
    t_duration = 300 #trial duration in ms
    iti = t_duration*2.5 #inter trial interval in ms

    events_stream, frames_stream, imu_stream, trig_stream = read_aedat_file(fpath)
    if dvs128:
        trig_stream = filter_triggers(trig_stream, iti)
    
    #events, IMUs, onset_offset = parse_events_stream(events_stream, imu_stream, trig_stream)
    events, frames, IMUs, onset_offset = parse_events_stream(events_stream, frames_stream, imu_stream, trig_stream, single_trig = dvs128, trial_duration = t_duration, fileid=trial_n)
        
    print("File read & parsed: " + fpath + '. Events #: ' + str(len(events)))
    return events, frames, IMUs, onset_offset, trial_n

####################*******************####################################

if __name__ == '__main__':
    
    dvs128 = False #wheteher data was collected with dvs128 or with DAVIS240C
    filter_sym = True
    # filter_sym = False
    
    # test_set_first_index = 60000
    test_set_first_index = -1 #Vernier only dataset, no train/test split

    using_events = True #Whether aedat files include events (True) or frames (False)
    # using_events = False #Whether aedat files include events (True) or frames (False)
    
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/Vernier_randompos_D2_frames_20251015/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_frames_exp30_20251008/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_frames_exp30_20251005/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_frames_exp6_20251005/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_frames_exp20_20250930/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_kmnist_frames_exp6_20250911/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_kmnist_eb_20250909/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_fmnist_eb_20250810/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_fmnist_frames_exp6_20250810/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_eb_20250809/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_fmnist_frames_exp6_20250808/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D2_mnist_frames_exp6_20250807/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_fmnist_frames_exp4_20250806/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_fmnist_frames_exp8_20250806/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp10_20250805/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp4_20250805/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp6_20250804/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_fmnist_eb_20250803/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_kmnist_eb_20250730/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp8_20250730/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_kmnist_frames_exp6_20250729/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_fmnist_frames_exp6_20250728/'
    f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_eb_20250724/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp6_20250722/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp6_20250718/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_eb_20250716/'
    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/D1_mnist_frames_exp20_20250715/'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240820'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240807'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240805'
    #f_path = './dataset/aedat/D2_fmnist_azel_carton_bg_20240728'
    #f_path = './dataset/aedat/D2_kmnist_frames_carton_bg_20240711'
    # f_path = './dataset/aedat/D2_mnist_frames_carton_bg_20240711'
    # f_path = './dataset/aedat/D2_fmnist_frames_carton_bg_20240710'
    # f_path = './dataset/aedat/D2_fmnist_az_carton_bg_20240708'
    # f_path = './dataset/aedat/D2_mnist_az_carton_bg_20240707'
    # f_path = './dataset/aedat/D2_kmnist_az_carton_bg_20240705'
    # f_path = './dataset/aedat/D1_kmnist_az_carton_bg_20240703'
    # f_path = './dataset/aedat/D1_kmnist_frames_carton_bg_20240702'
    # f_path = './dataset/aedat/D1_fmnist_frames_carton_bg_20240630'
    # f_path = './dataset/aedat/D1_mnist_frames_carton_bg_20240628'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240625'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240624'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240618'
    # f_path = './dataset/aedat/D1_mnist_az_carton_bg_20240623'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240621'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240619_2'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_bg_20240619'
    # f_path = './dataset/aedat/D1_fmnist_az_carton_20240617'
    # f_path = './dataset/aedat/D1_kmnist_azel_20240615'
    # f_path = './dataset/aedat/D1_mnist_az_light_dark_20240617'
    # f_path='./dataset/aedat/D3_kmnist_azel_20240611'
    # f_path='./dataset/aedat/D3_mnist_frames_20240609'
    # f_path='./dataset/aedat/D3_mnist_azel_20240607'
    # f_path='./dataset/aedat/D3_fmnist_azel_20240604'
    # f_path='./dataset/aedat/D3_fmnist_frames_20240602'
    # f_path='./dataset/aedat/farther_fmnist_frames_20240515'
    # f_path='./dataset/aedat/farther_fmnist_azel_20240510'
    #f_path='./dataset/aedat/farther_mnist_azel_20240503'
    # f_path='./dataset/aedat/far_mnist_azel_20240424'
    # f_path='./dataset/aedat/fmnist_azel_20240413'
    # f_path='./dataset/aedat/mnist_azel_20240407'
    #f_path='./dataset/aedat/gobilda_mnist_vernier_az_20240320'
    #f_path='./dataset/aedat/gobilda_mnist_az_20240317'
    #f_path='./dataset/aedat/mnist_frames_20240123'
    #f_path='./dataset/aedat/mnist_stepm_az_20240119'
    #f_path='./dataset/aedat/mnist_stepm_az_20240118'
    #f_path='./dataset/aedat/mnist_stepm_az_20240117'
    #f_path='./dataset/aedat/fmnist_azel_nbiases_20240111'
    #f_path='./dataset/aedat/mnist_azel_nbiases_20240110'
    # f_path='./dataset/aedat/mnist_az_sbiases_20240107'
    #f_path='./dataset/aedat/mnist_az_20240103'
    #f_path='./dataset/aedat/mnist_azel_20231205'

    # f_path = '/home/eldad/Workspace/Data/event-based-hyperacuity/datasets/aedat/Vernier_randompos_D2_frames_20251104/'

    #aedat_files_dict = get_aedat_file_list(f_path='./dataset/aedat/fmnist_20230711',fn_st='mnist', fn_en='.aedat4')
    #aedat_files_dict = get_aedat_file_list(f_path='./dataset/aedat/mnist_dvs128_20230802',fn_st='events_dvs128', fn_en='.aedat4')
    #aedat_files_dict = get_aedat_file_list(f_path='./dataset/aedat/mnist_frames_20231122',fn_st='frames', fn_en='.aedat4')

    print('Using path: ' + f_path)
    
    if using_events:
        aedat_files_dict = get_aedat_file_list(f_path=f_path, fn_st='mnist', fn_en='.aedat4')
    else:
        aedat_files_dict = get_aedat_file_list(f_path=f_path, fn_st='frames', fn_en='.aedat4')

    with multiprocessing.Pool(processes=30) as pool:
        #data_ = pool.map(parse_aedat_file, list(aedat_files_dict.items()))
        data_ = pool.starmap(parse_aedat_file, zip(aedat_files_dict.items(), repeat(dvs128)))
     
    trial_index = []
    for single_file_data in data_:
#        [events_all.append(e) for e in single_file_data[0]]
#        [IMUs_all.append(i) for i in single_file_data[1]]
#        [triggers_all.append(t) for t in single_file_data[2]]
        trial_index.append(single_file_data[4])

    files_index = np.argsort(np.array(trial_index))

    events_all = []
    IMUs_all = []
    triggers_all = []
    frames_all = []
    for indx in files_index:
        [events_all.append(e) for e in data_[indx][0]]
        [frames_all.append(t) for t in data_[indx][1]]
        [IMUs_all.append(i) for i in data_[indx][2]]
        [triggers_all.append(t) for t in data_[indx][3]]
    
    # print('setting xy offset')
    # # x_roi_offset=90
    # # y_roi_offset=65
    # # x_roi_offset=90
    # # y_roi_offset=75
    # x_roi_offset=90
    # y_roi_offset=55
    
    ###
    x_roi_offset=70
    y_roi_offset=45
    # x_roi_offset=80
    # y_roi_offset=55
    ###
    for sym_events in events_all:
        sym_events[:]['x'] -= x_roi_offset
        sym_events[:]['y'] -= y_roi_offset

    _path=f_path + '/ds_labels.pkl'
    with open(_path, 'rb') as f:
        labels = pickle.load(f)

    print('Before filtering: ' + str(len(labels)) + ', ' + str(len(events_all)))
    if len(labels) == len(events_all)+1:
        print('Warning: A single missing symbol, assuming its the first one')
        labels = np.delete(labels, 0)
    # elif len(labels) > len(events_all)+1:
    #     print('Warning: missing symbols, some symbols were not acquired!')
    #     labels = labels[:len(events_all)]

    # if samples include non-native ds symbols (e.g. vernier samples), remove them before splitting to train/test sets
    if filter_sym:
        # items_to_omit = np.argwhere(labels>9).squeeze() #All but Vernier
        items_to_omit = np.argwhere(labels<=9).squeeze() #Vernier
        items_to_omit.sort()
        print('items_to_ommit:' + str(len(items_to_omit)))

        labels = np.delete(labels, items_to_omit)
        for omit_i in reversed(items_to_omit):
            if using_events:
                del events_all[omit_i]
            else:
                del frames_all[omit_i]
            del IMUs_all[omit_i]
            del triggers_all[omit_i]
            
    print('After filtering: ' + str(len(labels)) + ', ' + str(len(events_all)))
        
    # if samples include test-set, seperating test and training sets to different files.
    if test_set_first_index != -1:
        _path=f_path + '/ds_samples_indexes.pkl'
        with open(_path, 'rb') as f:
            samples_indx = pickle.load(f)
               
        events_train = []
        IMUs_train = []
        frames_train = []
        triggers_train = []
        labels_train = []
        events_test = []
        IMUs_test = []
        frames_test = []
        triggers_test = []
        labels_test = []

        # In case not all samples were acquired omiting not relevant samples indexes from array
        # if samples_indx.shape[0]>len(events_all):
        #     print('Warning!, samples indexes array is larger than events_all')
        #     samples_indx = samples_indx[:len(events_all)]

        print('len(frames_all): ' + str(len(frames_all)))
        for indx, sample_indx in enumerate(samples_indx):
            if sample_indx >= test_set_first_index:
                if using_events:
                    events_test.append(events_all[indx])
                else:
                    frames_test.append(frames_all[indx])
                IMUs_test.append(IMUs_all[indx])
                triggers_test.append(triggers_all[indx])
                labels_test.append(labels[indx])
            else:
                if using_events:
                    events_train.append(events_all[indx])
                else:
                    frames_train.append(frames_all[indx])
                IMUs_train.append(IMUs_all[indx])
                triggers_train.append(triggers_all[indx])
                labels_train.append(labels[indx])
        
        print('saving to files...')
        full_path='./train/events.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(events_train, f)
        full_path='./train/imu.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(IMUs_train, f)
        full_path='./train/frames.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(frames_train, f)
        full_path='./train/trig.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(triggers_train, f)
        full_path='./train/labels.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(labels_train, f)

        full_path='./test/events.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(events_test, f)
        full_path='./test/imu.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(IMUs_test, f)
        full_path='./test/frames.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(frames_test, f)
        full_path='./test/trig.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(triggers_test, f)
        full_path='./test/labels.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(labels_test, f)
                
    else:
        print('saving to files...')
        full_path='./events.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(events_all, f)
        full_path='./imu.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(IMUs_all, f)
        full_path='./frames.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(frames_all, f)
        full_path='./triggers.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(triggers_all, f)
        full_path='./labels.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(labels, f)


    print(len(events_all))
    print(len(IMUs_all))
    print(len(frames_all))
    print(len(triggers_all))
        
    print(trial_index)
    print(files_index)
    
#    print(type(data_))
#    print(len(data_))
#    print(type(data_[0]))
#    print(len(data_[0]))
#    print(type(data_[0][0]))
#    print(len(data_[0][0]))
#    print(type(data_[0][0][0]))
#    print(data_[0][0][0].shape)
#    print(type(data_[0][1]))
#    print(len(data_[0][1]))
#    print(type(data_[0][1][0]))
#    print(data_[0][1][0].shape)
#    print(type(data_[0][2][0]))
#    print(len(data_[0][2]))
#    print(data_[0][2][:10])
                                  
