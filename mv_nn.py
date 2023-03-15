from constant import *
import os, re, copy, time, random,torch,sys
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.utils import data
import matplotlib.pyplot as plt
from tqdm import tqdm

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE


class LSTM(nn.Module):
    def __init__(self, na_fill_type,input_size, hidden_size, num_layers, output_size,device):
        super(LSTM, self).__init__() 
        self.input_size = input_size   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self._na_fill_type=na_fill_type

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.device=device

     def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
    
    def forward(self, data):

        if self._na_fill_type!='mask':
            input=data
            batch_size = input.shape[0]

        else:
            input, input_lengths=data

            batch_size=len(input_lengths)

        hidden=self.init_hidden(batch_size)

        if self._na_fill_type!='mask':

            output, _ = self.lstm(input, hidden)

        else:
            pack_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True,enforce_sorted=False)

            pack_output,hidden=self.lstm(pack_input,hidden)

            output, _ =torch.nn.utils.rnn.pad_packed_sequence(pack_output,batch_first=True)
            y_pred = self.linear(output)[:, -1,:]
          return y_pred




class AdaRNN_noTL(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(self, na_fill_type,use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9, model_type='AdaRNN_noTL'):
        super(AdaRNN_noTL, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        # self.trans_loss = trans_loss
        self.len_seq = len_seq
        in_size = self.n_input
        self._na_fill_type=na_fill_type
        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck == True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight) #initiate the weight as xavier_normal instead of uniform distribution
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type in ['AdaRNN','adaRNN','ADARNN','adarnn','Adarnn']:
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i]*2, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def forward(self, data, len_win=0):
        out = self.gru_features(data)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck)
   
        else:
            fc_out = self.fc_out(fea[:, -1, :])

        fc_out = torch.sigmoid(fc_out)#<--added to get [0,1] results
        return fc_out
        # loss_transfer,

    def gru_features(self, data, predict=False):
        # the features output from the two layers of GRU using the multi-variate matrix features
        out = None
        out_lis = []
        out_weight_list = [] if (self.model_type in ['AdaRNN','adaRNN','ADARNN','adarnn','Adarnn']) else None
        if self._na_fill_type!='mask':
            # x_input = x
            x_input=data
            batch_size = x_input.shape[0]
        else:
            input, input_lengths=data
            batch_size=len(input_lengths)
            x_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True,enforce_sorted=False)

        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            if self._na_fill_type=='mask':
                out, _ =torch.nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
            x_input = out
            out_lis.append(out)

        return out, out_lis



    def predict(self, x):

        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out 
        
  
        
class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension (input data features)
    d_model:
        Dimension of the input vector (hidden units)
    d_output:
        Model output dimension. (output targets)
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,na_fill_type,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 24):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        self._na_fill_type=na_fill_type
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        

        # Embeddin module
        if self._na_fill_type!='mask':
            input=data
            batch_size = input.shape[0]
            # print(f'input shape: {input.shape}')
        else:
            input, input_lengths=data
            
             #use none to increase dimension on the desired place you want

            batch_size=len(input_lengths)
        K = input.shape[1]
        encoding = self._embedding(input)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        output=output.squeeze(2)
        # print(f'final output size: {output.shape}')
        return output
    
  
