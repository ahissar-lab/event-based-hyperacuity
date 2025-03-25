import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 n_timesteps = 64,
                d_timeseries = 4,
                d_k = 256,
                d_mlp = 512,
                d_mlp2 = 256,
                n_layers = 8,
                num_heads = 4,
                num_classes = 10,
                dropout_rate = 0.1,
                offsets  = None,
                scalings = None,
                dropout_ffn_from_layer = 4,
                subseq_len = None,
                model_head = 'cls_mlp',
                model_head_init_method = None,
                input_dense_init_method = None):

        '''
        Transformer for time series data.
        :param n_timesteps: number of timesteps in the input
        :param d_timeseries: dimensionality of the timeseries
        :param d_k: dimensionality of the key and query vectors
        :param d_mlp: dimensionality of the MLP in the feedforward layer
        :param d_mlp2: dimensionality of the MLP in the classification head
        :param n_layers: number of layers in the Transformer
        :param num_heads: number of heads in the Multi-Head Attention
        :param num_classes: number of classes in the classification head
        :param dropout_rate: dropout rate
        :param offsets: offsets to be added to the input
        :param scalings: scalings to be multiplied to the input
        :param dropout_ffn_from_layer: layer from which to apply dropout to the feedforward network
        :param subseq_len: list or tuple. when not None, the input is split into subsequences according to this list, each subsequence is processed by the transformer, and the outputs are concatenated
        :param model_head: head architecture. 'cls_mlp' or 'cls_avgpool'
        Receives input of shape (batch_size, n_timesteps, d_timeseries), returns logits of shape (batch_size, num_classes)
        '''
        super(TimeSeriesTransformer, self).__init__()
        if offsets is None:
            offsets = torch.zeros(1, 1, d_timeseries)
        if scalings is None:
            scalings = torch.ones(1, 1, d_timeseries)
        self.offsets = offsets
        self.scalings = scalings
        self.n_layers = n_layers
        self.dropout_ffn_from_layer = dropout_ffn_from_layer
        self.subseq_len = subseq_len
        self.model_head_init_method = model_head_init_method

        # Initial dense layer to expand dimensions
        #todo: add option to use dedicated input dense layer per subsequence in a case of multiple subsequences
        self.input_dense = nn.Linear(d_timeseries, d_k)
        if input_dense_init_method is not None:
            apply_init_method([self.input_dense], input_dense_init_method)

        # Defining n_layers of Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_k, num_heads, d_mlp, dropout_rate) for _ in range(n_layers)
        ])

        if model_head == 'cls_mlp':
            self.cls_head = ClassifierHead_MLP(d_k * n_timesteps,
                                               d_mlp2, num_classes, dropout_rate,do_flatten=True)
        elif model_head == 'cls_avgpool':
            self.cls_head = ClassifierHead_AvgPool(d_k, d_mlp2, num_classes, dropout_rate, init_method=model_head_init_method)
        elif model_head == 'proj_avgpool':
            self.cls_head = ProjectionHead_AvgPool(d_k, d_mlp2)
        elif model_head == 'none':
            self.cls_head = nn.Identity()
        else:
            raise ValueError('Unknown cls_head_arch')

    def forward(self, x, attn_mask=None, padding_mask=None):

        # Add offsets and scaling
        #convert to torch tensor locally to avoid problems with multiple GPUs
        torch_offsets = torch.tensor(self.offsets, device=x.device, dtype=x.dtype)
        torch_scalings = torch.tensor(self.scalings, device=x.device, dtype=x.dtype)

        if self.subseq_len is not None and (attn_mask is not None or padding_mask is not None):
            raise ValueError('subseq_len cannot be used with attn_mask or key_padding_mask')

        if self.subseq_len is not None:
            #in a case of multiple subsequences, we create an attention mask to prevent attention between subsequences
            #this mask is a matrix of shape (n_timesteps, n_timesteps) it is false between all pairs of timesteps that belong to different subsequences
            #otherwise, the mask is None
            attn_mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
            this_position = 0
            for subseq_len in self.subseq_len:
                attn_mask[this_position:this_position+subseq_len, this_position:this_position+subseq_len] = False
                this_position += subseq_len

        x = (x + torch_offsets) * torch_scalings

        x = self.input_dense(x)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, apply_dropout=(i >= self.dropout_ffn_from_layer), attn_mask=attn_mask, key_padding_mask=padding_mask)

        x = self.cls_head(x, mask=padding_mask)

        return x


class TimeSeriesTransformerWithSubseq(nn.Module):
    def __init__(self,
                 n_timesteps=64,
                 d_timeseries=4,
                 d_k=256,
                 d_mlp=512,
                 d_mlp2=256,
                 n_layers=8,
                 num_heads=4,
                 num_classes=10,
                 dropout_rate=0.1,
                 offsets=None,
                 scalings=None,
                 dropout_ffn_from_layer=4,
                 subseq_len=None,
                 model_head='cls_mlp',
                 shared_input_dense=False,
                 drop_subseq_prob=0.5,
                 extra_dim_for_subsamples=False):
        '''
        Transformer for time series data. With option to split the input into multiple subsequences.
        :param n_timesteps: number of timesteps in the input
        :param d_timeseries: dimensionality of the timeseries
        :param d_k: dimensionality of the key and query vectors
        :param d_mlp: dimensionality of the MLP in the feedforward layer
        :param d_mlp2: dimensionality of the MLP in the classification head
        :param n_layers: number of layers in the Transformer
        :param num_heads: number of heads in the Multi-Head Attention
        :param num_classes: number of classes in the classification head
        :param dropout_rate: dropout rate
        :param offsets: offsets to be added to the input
        :param scalings: scalings to be multiplied to the input
        :param dropout_ffn_from_layer: layer from which to apply dropout to the feedforward network
        :param subseq_len: list or tuple. when not None, the input is split into subsequences according to this list, each subsequence is processed by the transformer, and the outputs are concatenated
        :param model_head: head architecture. 'cls_mlp' or 'cls_avgpool'
        :param shared_input_dense: if True, the input is processed by a single dense layer before being split into subsequences. If False, each subsequence is processed by a dedicated dense layer.
        :param drop_subseq_prob: probability of dropping a subsequence during training
        Receives input of shape (batch_size, n_timesteps, d_timeseries), returns logits of shape (batch_size, num_classes)
        '''
        super(TimeSeriesTransformerWithSubseq, self).__init__()
        if offsets is None:
            offsets = torch.zeros(1, 1, d_timeseries)
        if scalings is None:
            scalings = torch.ones(1, 1, d_timeseries)
        self.offsets = offsets
        self.scalings = scalings
        self.n_layers = n_layers
        self.dropout_ffn_from_layer = dropout_ffn_from_layer
        self.subseq_len = subseq_len
        self.shared_input_dense = shared_input_dense
        self.drop_subseq_prob = drop_subseq_prob
        self.extra_dim_for_subsamples = extra_dim_for_subsamples

        # Initial dense layer to expand dimensions
        if shared_input_dense:
            self.input_dense = nn.Linear(d_timeseries, d_k)
        else:
            self.input_dense = nn.ModuleList([nn.Linear(d_timeseries, d_k) for _ in range(len(subseq_len))])

        if subseq_len is None:
            raise NotImplementedError
            # Defining n_layers of Transformer Layers
            # self.transformer_layers = nn.ModuleList([
            #     TransformerLayer(d_k, num_heads, d_mlp, dropout_rate) for _ in range(n_layers)
            # ])
        else:
            # in case of multiple subsequences, we need to split the input into subsequences
            # each subsequence is processed by a dedicated transformer, and the outputs are concatenated
            for i in range(len(subseq_len)):
                setattr(self, 'transformer_stack_{}'.format(i), TransformerCore(d_k=d_k,
                                                                                num_heads=num_heads,
                                                                                d_mlp=d_mlp,
                                                                                n_layers=n_layers,
                                                                                dropout_rate=dropout_rate,
                                                                                dropout_ffn_from_layer=dropout_ffn_from_layer))

        if model_head == 'cls_mlp':
            self.cls_head = ClassifierHead_MLP(d_k * (len(subseq_len) if subseq_len is not None else 1),
                                               d_mlp2, num_classes, dropout_rate,do_flatten=extra_dim_for_subsamples)
        elif model_head == 'cls_avgpool':
            self.cls_head = ClassifierHead_AvgPool(d_k, d_mlp2, num_classes, dropout_rate)
        elif model_head == 'proj_avgpool':
            self.cls_head = ProjectionHead_AvgPool(d_k, d_mlp2, skip_mean=True)
        elif model_head == 'none':
            self.cls_head = nn.Identity()
        else:
            raise ValueError('Unknown cls_head_arch')

    def forward(self, x):
        # Add offsets and scaling
        #convert to torch tensor locally to avoid problems with multiple GPUs
        torch_offsets = torch.tensor(self.offsets, device=x.device, dtype=x.dtype)
        torch_scalings = torch.tensor(self.scalings, device=x.device, dtype=x.dtype)

        x = (x + torch_offsets) * torch_scalings

        if self.shared_input_dense:
            x = self.input_dense(x)


        if self.subseq_len is None:
            raise NotImplementedError
            # Pass through each Transformer Layer
            # for i, layer in enumerate(self.transformer_layers):
            #     x = layer(x, apply_dropout=(i >= self.dropout_ffn_from_layer))
        else:
            # in case of multiple subsequences, we need to split the input into subsequences
            # each subsequence is processed by a dedicated transformer, and the outputs are passed through average
            # pooling and concatenated
            x_list = []
            this_position = 0
            for i, subseq_len in enumerate(self.subseq_len):
                if self.extra_dim_for_subsamples:
                    x_subseq = x[:, i, ...]
                else:
                    x_subseq = x[:, this_position:this_position+subseq_len, :]
                this_position += subseq_len
                if not self.shared_input_dense:
                    x_subseq = self.input_dense[i](x_subseq)
                x_subseq = getattr(self, 'transformer_stack_{}'.format(i))(x_subseq)
                x_subseq = torch.mean(x_subseq, dim=1)
                x_list.append(x_subseq)

            # if in training mode with probability self.drop_subseq_prob pick at most one subsequence and set it to zero
            # this is done by creating a subsequence of zeros and replacing one random subsequence
            # out of the original subsequences with zeros.
            if self.training:
                if torch.rand(1) < self.drop_subseq_prob:
                    x_list[torch.randint(len(x_list),(1,))] = torch.zeros_like(x_list[0])
                    #scale all other subsequences to compensate for the zero subsequence
                    x_list = [x * (len(x_list) / (len(x_list) - 1)) for x in x_list]

            if self.extra_dim_for_subsamples:
                x = torch.stack(x_list, dim=1)
            else:
                x = torch.cat(x_list, dim=1)

        # Classification head
        x = self.cls_head(x)

        return x

class TransformerCore(nn.Module):
    def __init__(self,
                 d_k=256,
                 d_mlp=512,
                 n_layers=8,
                 num_heads=4,
                 dropout_rate=0.1,
                 dropout_ffn_from_layer=4):
        '''
        Core of the Transformer. No classification head or input preprocessing.
        :param d_k: dimensionality of the key and query vectors
        :param d_mlp: dimensionality of the MLP in the feedforward layer
        :param n_layers: number of layers in the Transformer
        :param num_heads: number of heads in the Multi-Head Attention
        :param dropout_rate: dropout rate
        :param dropout_ffn_from_layer: layer from which to apply dropout to the feedforward network
        Receives input of shape (batch_size, n_timesteps, d_timeseries), returns logits of shape (batch_size, num_classes)
        '''
        super(TransformerCore, self).__init__()

        self.n_layers = n_layers
        self.dropout_ffn_from_layer = dropout_ffn_from_layer

        # Defining n_layers of Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_k, num_heads, d_mlp, dropout_rate) for _ in range(n_layers)
        ])


    def forward(self, x):
       # Pass through each Transformer Layer
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, apply_dropout=(i >= self.dropout_ffn_from_layer))
        return x


class TimeSeriesGRU(nn.Module):
    def __init__(self,
                 n_timesteps=64,
                 d_timeseries=4,
                 d_hidden=256,
                 d_mlp=512,
                 d_mlp2=256,
                 n_layers=8,
                 num_classes=10,
                 dropout_rate=0.1,
                 offsets=None,
                 scalings=None,
                 dropout_ffn_from_layer=4,
                 subseq_len=None,
                 model_head='cls_mlp',
                 model_head_init_method=None,
                 input_dense_init_method=None):
        '''
        GRU-based model for time series data.
        :param n_timesteps: number of timesteps in the input
        :param d_timeseries: dimensionality of the timeseries
        :param d_hidden: dimensionality of the hidden state in the GRU
        :param d_mlp: dimensionality of the MLP in the feedforward layer
        :param d_mlp2: dimensionality of the MLP in the classification head
        :param n_layers: number of GRU layers
        :param num_classes: number of classes in the classification head
        :param dropout_rate: dropout rate
        :param offsets: offsets to be added to the input
        :param scalings: scalings to be multiplied to the input
        :param dropout_ffn_from_layer: layer from which to apply dropout to the feedforward network
        :param subseq_len: list or tuple. when not None, the input is split into subsequences according to this list, each subsequence is processed by the GRU, and the outputs are concatenated
        :param model_head: head architecture. 'cls_mlp' or 'cls_avgpool'
        Receives input of shape (batch_size, n_timesteps, d_timeseries), returns logits of shape (batch_size, num_classes)
        '''
        super(TimeSeriesGRU, self).__init__()
        if offsets is None:
            offsets = torch.zeros(1, 1, d_timeseries)
        if scalings is None:
            scalings = torch.ones(1, 1, d_timeseries)
        self.offsets = offsets
        self.scalings = scalings
        self.n_layers = n_layers
        self.dropout_ffn_from_layer = dropout_ffn_from_layer
        self.subseq_len = subseq_len
        self.model_head_init_method = model_head_init_method

        # Initial dense layer to expand dimensions
        self.input_dense = nn.Linear(d_timeseries, d_hidden)
        if input_dense_init_method is not None:
            apply_init_method([self.input_dense], input_dense_init_method)

        # GRU layer
        self.gru_layers = nn.ModuleList([
            nn.GRU(input_size=d_hidden, hidden_size=d_hidden, num_layers=1, batch_first=True) for _ in range(n_layers)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        if model_head == 'cls_mlp':
            self.cls_head = ClassifierHead_MLP(d_hidden * n_timesteps, d_mlp2, num_classes, dropout_rate,
                                               do_flatten=True)
        elif model_head == 'cls_avgpool':
            self.cls_head = ClassifierHead_AvgPool(d_hidden, d_mlp2, num_classes, dropout_rate,
                                                   init_method=model_head_init_method)
        elif model_head == 'proj_avgpool':
            self.cls_head = ProjectionHead_AvgPool(d_hidden, d_mlp2)
        elif model_head == 'none':
            self.cls_head = nn.Identity()
        else:
            raise ValueError('Unknown cls_head_arch')

    def forward(self, x, attn_mask=None, padding_mask=None):
        # Add offsets and scaling
        torch_offsets = torch.tensor(self.offsets, device=x.device, dtype=x.dtype)
        torch_scalings = torch.tensor(self.scalings, device=x.device, dtype=x.dtype)

        if self.subseq_len is not None and (attn_mask is not None or padding_mask is not None):
            raise ValueError('subseq_len cannot be used with attn_mask or key_padding_mask')

        x = (x + torch_offsets) * torch_scalings
        x = self.input_dense(x)

        # GRU forward pass with conditional dropout
        for i, gru_layer in enumerate(self.gru_layers):
            x, _ = gru_layer(x)
            if i >= self.dropout_ffn_from_layer:
                x = self.dropout(x)

        # Apply the classification head

        x = self.cls_head(x.contiguous(), mask=padding_mask)

        return x


class VanillaCNN(nn.Module):
    def __init__(self, input_shape = (10,10),
                 num_classes = 10):
        super(VanillaCNN, self).__init__()

        self.resize = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)

        self.pool = nn.MaxPool2d(2)

        # Calculate the size of the flattened features
        self.flat_features = self._get_flat_features()

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.resize(x)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = x.view(-1, self.flat_features)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def _get_flat_features(self):
        # Helper method to calculate the size of the flattened features
        with torch.no_grad():
            x = torch.zeros(1, 1, 56, 56)  # Dummy input
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            return x.numel()


class GrayscaleResNet18(nn.Module):
    def __init__(self, input_shape = (10,10),
                 num_classes = 10):
        super(GrayscaleResNet18, self).__init__()

        # Load pretrained ResNet18
        self.resnet = resnet18(weights=None)

        # Modify the first convolutional layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer if needed
        if num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(224, antialias=True),
            # transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def forward(self, x):
        # x is expected to be a grayscale image tensor of shape (B, 1, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.preprocess(x)
        return self.resnet(x)

class ClassifierHead_MLP(nn.Module):
    def __init__(self, d_flatten, d_mlp, num_classes, dropout_rate, do_flatten=True):
        super(ClassifierHead_MLP, self).__init__()
        self.do_flatten = do_flatten
        self.additional_dense = nn.Linear(d_flatten, d_mlp)
        self.output_dense = nn.Linear(d_mlp, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        if mask is not None:
            raise ValueError('mask is not supported for MLP head')
        if self.do_flatten:
            x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(x)
        x = F.relu(self.additional_dense(x))
        x = self.dropout(x)
        x = self.output_dense(x)
        return x

class ClassifierHead_AvgPool(nn.Module):
    def __init__(self, d_k, d_mlp, num_classes, dropout_rate, init_method=None):
        super(ClassifierHead_AvgPool, self).__init__()
        self.additional_dense = nn.Linear(d_k, d_mlp)
        self.output_dense = nn.Linear(d_mlp, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        if init_method is not None:
            apply_init_method([self.additional_dense, self.output_dense], init_method)

    def forward(self, x, mask=None):
        x = masked_mean(x, mask, dim=1)
        x = self.dropout(x)
        x = F.relu(self.additional_dense(x))
        x = self.dropout(x)
        x = self.output_dense(x)
        return x

class ProjectionHead_AvgPool(nn.Module):
    def __init__(self, d_k, d_mlp, skip_mean=False):
        super(ProjectionHead_AvgPool, self).__init__()
        self.additional_dense = nn.Linear(d_k, d_mlp)
        self.output_dense = nn.Linear(d_mlp, d_mlp)
        self.skip_mean = skip_mean

    def forward(self, x, mask=None):
        if not self.skip_mean:
            x = masked_mean(x, mask=mask, dim=1)
        x = F.relu(self.additional_dense(x))
        x = self.output_dense(x)
        return x




class TransformerLayer(nn.Module):
    def __init__(self, d_k, num_heads, d_mlp, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.query_projection = nn.Linear(d_k, d_k)
        self.key_projection = nn.Linear(d_k, d_k)
        self.value_projection = nn.Linear(d_k, d_k)

        self.attention = nn.MultiheadAttention(d_k, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_k)
        self.norm2 = nn.LayerNorm(d_k)

        self.ffn = nn.Sequential(
            nn.Linear(d_k, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_k)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, apply_dropout=False,attn_mask=None,key_padding_mask=None):
        # Projecting to query, key, and value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # Multi-Head Attention and Residual Connection
        att_out, _ = self.attention(query, key, value,attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        x = self.norm1(x + att_out)

        # Feed-Forward Network and Residual Connection
        ffn_out = self.ffn(x)
        if apply_dropout:
            ffn_out = self.dropout(ffn_out)
        x = self.norm2(x + ffn_out)

        return x


def masked_mean(x,mask=None,dim=1):
    #mask indicates which elements should be excluded from the mean
    if mask is None:
        x = torch.mean(x, dim=dim)
    else:
        #expand mask to the same dimentionality as x
        mask = mask.unsqueeze(-1)
        x = torch.sum(x * ~mask, dim=dim) / torch.sum(~mask, dim=dim)
    return x

def apply_init_method(layers, init_method):
    if init_method == 'xavier':
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    elif init_method == 'kaiming':
        for layer in layers:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    elif init_method == 'small_weights':
        for layer in layers:
            nn.init.uniform_(layer.weight, -0.01, 0.01)
            nn.init.zeros_(layer.bias)
    elif init_method == 'diag':
        for layer in layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError('Unknown init method')