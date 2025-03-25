import yaml
import json
import os

import torch

class TerminalCapturingLogger:
    """
    This class is used to capture the output of the terminal and save it to a log file.
    """
    def __init__(self,log_name):
        self.log_name = log_name
        self.terminal = sys.stdout
        self.log = open(log_name, "w")
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.log_name, "a")
        self.log.write(message)
        self.log.close() #to make log readable at all times...

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

class VanillaLogger:
    '''
    This class saves includes a method for printing to the terminal and saving to a log file.
    '''
    def __init__(self, log_name, append=False):
        self.log_name = log_name
        if not append:
            self.log = open(log_name, "w")
            self.log.close()

    def print_and_log(self, message):
        print(message)
        self.log = open(self.log_name, "a")
        self.log.write(message + "\n")
        self.log.close()

# Load and save YAML configuration files
def load_config(file_path):
    """
    Load a YAML configuration file.

    Parameters:
    file_path (str): The path to the YAML file to be loaded.

    Returns:
    dict: A dictionary containing the configuration.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, file_path):
    """
    Save a configuration dictionary to a YAML file.

    Parameters:
    config (dict): The configuration dictionary to save.
    file_path (str): The path where the YAML file will be saved.
    """
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# logging and saving utils

def mkdir_if_needed(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def assign_run_name(path, candidate_run_name='noname'):
    if not os.path.exists(path):
        raise Exception('ERROR: path does not exist')
    i = 0
    run_name = candidate_run_name
    while os.path.exists(path + run_name):
        i += 1
        run_name = candidate_run_name + str(i)
    return run_name

#logging training data

def write_to_json(file_path, this_dict):
    """
    Appends a dictionary of metrics to a JSON file.

    :param file_path: Path to the JSON file.
    :param metrics: A dictionary containing metric names and their values, including epoch.
    """
    with open(file_path, 'a') as file:
        json.dump(this_dict, file)
        file.write("\n")  # Add a newline to separate entries


def load_from_json(file_path):
    """
    Loads data from a JSON file and returns a flat dictionary of metrics.

    :param file_path: Path to the JSON file.
    :return: A dictionary with metric names as keys and lists of their values as values.
    """
    metrics_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            for metric, value in entry.items():
                if metric not in metrics_dict:
                    metrics_dict[metric] = []
                metrics_dict[metric].append(value)
    return metrics_dict

def parse_args_by_nested_prefix(args, prefix, subprefixes):
    """
    Parses attributes of an object based on a specified prefix and subprefixes,
    organizing them into a nested dictionary structure. Shared attributes (without subprefix)
    are included in each sub-dictionary.
    For example, for args with attributes args.foo=z, args.a_boo=w, args.a_b_c=x, args.a_d_c=y,
    prefix='a', subprefixes=['b','d'], it returns {'b': {'boo': w, 'c': x}, 'd': {'boo': w, 'c': y}}.
    :param args: An object with attributes.
    :param prefix: The main prefix of the attributes.
    :param subprefixes: A list of subprefixes.
    :return: A nested dictionary of attributes organized by subprefixes.
    """
    if len(subprefixes) == 0:
        subprefixes = None

    if subprefixes is None:
        # If no subprefixes are specified, return a dictionary that includes all attributes with the prefix
        res_dict = {}
        for key, value in vars(args).items():
            if key.startswith(prefix + '_'):
                attribute_name = key[len(prefix) + 1:]
                res_dict[attribute_name] = value
        return res_dict


    args_dict = vars(args)
    res_dict = {subprefix: {} for subprefix in subprefixes}

    # Identify and add shared attributes
    shared_attrs = {}
    for k, v in args_dict.items():
        if k.startswith(prefix + '_'):
            is_shared = True
            for sp in subprefixes:
                if k.startswith(prefix + '_' + sp + '_'):
                    is_shared = False
                    break
            if is_shared:
                attribute_name = k[len(prefix) + 1:]
                shared_attrs[attribute_name] = v

    # Update each subprefix dictionary with the shared attributes
    for subprefix in subprefixes:
        res_dict[subprefix].update(shared_attrs)

    # Add subprefix-specific attributes
    for key, value in args_dict.items():
        for subprefix in subprefixes:
            if key.startswith(prefix + '_' + subprefix + '_'):
                attribute_name = key[len(prefix + '_' + subprefix) + 1:]
                res_dict[subprefix][attribute_name] = value

    return res_dict

#checkpoint utils
def remove_checkpoint_prefix(checkpoint,prefix='module.'):
    return {k.partition(prefix)[-1]: v for k,v in checkpoint.items()}

def discard_keys_by_prefix(my_dict,prefix='cls_head.'):
    return {k: v for k,v in my_dict.items() if not k.startswith(prefix)}

def checkpoint_preprocess(cp):
    return discard_keys_by_prefix(remove_checkpoint_prefix(cp))

def load_and_preprocess_headless_checkpoint(cp_file):
    cp_dict = torch.load(cp_file)
    if 'model_state_dict' in cp_dict:
        return checkpoint_preprocess(cp_dict['model_state_dict'])
    else:
        #here we assume that the checkpoint contains only the model state dict
        return checkpoint_preprocess(cp_dict)

def weigh_freezer(model,dont_freeze_prefix='cls_head.',verbose=True):
    for ii, (name, param) in enumerate(model.named_parameters()):
        if not name.startswith(dont_freeze_prefix):
            if param.requires_grad:
                param.requires_grad = False
                if verbose:
                    print('freezing layer ',ii,' :',name, ' requires_grad:',param.requires_grad)
        else:
            if verbose:
                print('not freezing layer ',ii,' :',name, ' requires_grad:',param.requires_grad)

def relabler(labels,relabel_opt):
    if relabel_opt == 'vernier13_27':
        #all labels smaller that 20 are relabeled to 0, all labels greater or equal to 20 are relabeled to 1
        # take care of list of labels or tensor of labels
        if isinstance(labels,list):
            return [0 if x < 20 else 1 for x in labels]
        elif isinstance(labels,torch.Tensor):
            return torch.where(labels<20,torch.zeros_like(labels),torch.ones_like(labels))
        elif isinstance(labels,np.ndarray):
            return np.where(labels<20,0,1)
        else:
            raise ValueError(f'unsupported type {type(labels)}')
    else:
        raise NotImplementedError(f'relabel option "{relabel_opt}" is not implemented')

if __name__ == '__main__':
    #test parse_args_by_nested_prefix
    class Args:
        def __init__(self):
            self.a_u_boo = 1
            self.a_u_b_c = 2
            self.a_u_d_c = 3
            self.foo = 4
            self.bar = 5
            self.baz = 6
    args = Args()
    prefix = 'a_u'
    subprefixes = ['b', 'd']
    res_dict = parse_args_by_nested_prefix(args, prefix, subprefixes)
    print('args',args.__dict__)
    print('subprefixes: ', subprefixes)
    print('result',res_dict)

    #now retest with empty subprefixes
    subprefixes = []
    res_dict = parse_args_by_nested_prefix(args, prefix, subprefixes)
    print('args',args.__dict__)
    print('subprefixes: ', subprefixes)
    print('result',res_dict)

def parse_strings_to_lists(args):
    '''
    Goes over all the attributes of args and if an argument is a string that can be converted to a list, it converts it to a list.
    :param args: a class of arguments
    :return: None, modifies args in-place
    '''
    for key, value in vars(args).items():
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                setattr(args, key, eval(value))
            except SyntaxError:
                # If the string is not a valid list, do not change it
                pass


def compute_mean_offsets(dataset):
    offsets = 0
    for ii, (d,l) in enumerate(dataset):
        offsets += d.mean(axis=(0,1))
    # if offsets are torch tensors, convert them to numpy
    if isinstance(offsets,torch.Tensor):
        offsets = offsets.cpu().numpy()
    return offsets/(ii+1)
