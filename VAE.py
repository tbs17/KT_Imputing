import os, sys, torch, argparse,copy
import pandas as pd
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
from tqdm import tqdm
from predict_HealthMNIST import VAEoutput
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset
from parse_model_args import VAEArgs
from model_test import VAEtest
from validation import *

class ConvVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False, p_input=0.2, p=0.5):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim
        self.p_input = p_input
        self.p = p

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_1 = nn.Dropout2d(p=self.p)  # spatial dropout

        # second convolution layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_2 = nn.Dropout2d(p=self.p)

        self.fc1 = nn.Linear(32 * 9 * 9, 300)
        self.dropout1 = nn.Dropout(p=self.p)
        self.fc21 = nn.Linear(300, 30)
        self.dropout2 = nn.Dropout(p=self.p)
        self.fc211 = nn.Linear(30, self.latent_dim)
        self.fc221 = nn.Linear(30, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.dropout3 = nn.Dropout(p=self.p)
        self.fc31 = nn.Linear(30, 300)
        self.dropout4 = nn.Dropout(p=self.p)
        self.fc4 = nn.Linear(300, 32 * 9 * 9)

        self.dropout2d_3 = nn.Dropout2d(p=self.p)
        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        self.dropout2d_4 = nn.Dropout2d(p=self.p)
        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        # convolution
        z = F.relu(self.conv1(x))
        z = self.dropout2d_1(self.pool1(z))
        z = F.relu(self.conv2(z))
        z = self.dropout2d_2(self.pool2(z))

        # MLP
        z = z.view(-1, 32 * 9 * 9)
        h1 = self.dropout1(F.relu(self.fc1(z)))
        h2 = self.dropout2(F.relu(self.fc21(h1)))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = self.dropout3(F.relu(self.fc3(z)))
        x = self.dropout4(F.relu(self.fc31(x)))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = self.dropout2d_3(x.view(-1, 32, 9, 9))
        x = self.dropout2d_4(F.relu(self.deconv1(x)))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample_latent(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)



class LSTMVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, hidden_dim, num_layers, dropout, latent_dim, num_dim, vy_init=1.0, vy_fixed=False):
        super(LSTMVAE, self).__init__()


        # self.T=T
        self.hidden_dim=hidden_dim
        self.latent_dim = latent_dim
        self.num_dim = num_dim
        self.num_layers=num_layers
        self.dropout=dropout
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        min_log_vy = torch.Tensor([-8.0]) #this is class variable
        
        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy)) #vy is initiated by using the preset initial value of vy minus the min log vy is also pre-set
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init])) #variance is always dependent on the num of dimensions, if there are 10 variables, then we have 10 variance
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init)) #set it as as a NN learnable parameter

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        self.encoder_lstm = nn.LSTM(self.num_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.mu = torch.nn.Linear(in_features= self.hidden_dim, out_features= self.latent_dim)
        self.log_var = torch.nn.Linear(in_features= self.hidden_dim , out_features= self.latent_dim)
        

        self.init_hidden_decoder = torch.nn.Linear(in_features= self.latent_dim, out_features= self.hidden_dim )
        self.decoder_lstm = torch.nn.LSTM(input_size= self.num_dim, hidden_size= self.hidden_dim, batch_first = True, num_layers = self.num_layers)
        self.output = torch.nn.Linear(in_features= self.hidden_dim, out_features= self.num_dim)
        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1)) # if you don't want the optimizer to update them, then just register them as buffer

    @property
    def vy(self): #define how vy is transformed from log_vy taking in min_log_vy as a facotr
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):#set the vy up
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def init_hidden(self,batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
                # .float().to(self.device)
                # reinforce the tensor to be on the cuda for evaluation issues 
    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """

        batch_size = x.shape[0]
        # print(f'batch_size :{x.size()}')
        # numpy uses float64 by default and pytorch uses float32
        # print(f'batch_size:{batch_size}')
        hidden = self.init_hidden(batch_size)
        # print(f'data/hidden[0].dtype:{x.dtype}/{hidden[0].dtype}')
        # when float, it's data/hidden[0].dtype:torch.float32/torch.float32
        # when not changing anything：
        x ,hidden_encoder= self.encoder_lstm(x,hidden)
        # print(f'input shape after lstm and before: {x.shape}')
        # b/c we are looking for the mean and var of the latent space, it needs the output from the latent space (that is the hidden states)
        # print(f'hidden state shape: {hidden_encoder[0].shape}\n'
        # f'cell state shape: {hidden_encoder[1].shape}')
        # print(f'hidden shape:{hidden[0].shape}')
        mu=self.mu(hidden_encoder[0])
        log_var=self.log_var(hidden_encoder[0])
        # print(f'mu/log var shape:{mu.shape}/{log_var.shape}')
        return mu, log_var

    def decode(self, x,z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # h3 = F.relu(self.fc3(z))
        # h4 = F.relu(self.fc31(h3))
        # return torch.sigmoid(self.fc4(h4)) #sigmoid outputs a probability that [0,1]
        hidden_decoder=self.init_hidden_decoder(z)
        # print(f'hidden_decoder shape:{hidden_decoder.shape}')
        hidden_decoder=(hidden_decoder,hidden_decoder)

        output,hidden_decoder=self.decoder_lstm(x,hidden_decoder)
        # print(f'output shape after decoder_lstm: {output.shape}')
        
        x_hat=self.output(output)
        # print(f'x shape given z: {x_hat.shape}')
        reconstruction = torch.sigmoid(x_hat) 
        return reconstruction

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        
        mu, logvar = self.encode(x)

        z = self.sample_latent(mu, logvar)
        reconstruction=self.decode(x,z)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        # print(f'recon_x/x size:{recon_x.shape}/{x.shape}')
        batch_size=x.shape[0]
        T=x.shape[1]
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        # print(f'recon_x/x/mask shape:{recon_x.shape}/{x.shape}/{mask.shape}')
        # recon_reshape=recon_x.view(-1, self.num_dim)
        # x_reshape= x.view(-1, self.num_dim)
        # mask_reshape=mask.view(-1, self.num_dim)
        # print(f'recon_x/x/mask reshape:{recon_x.shape}/{x.shape}/{mask.shape}')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        # se shape: se shape:torch.Size([120, 11], subjects per batch =6 and T=20
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        # original:mask_sum = torch.sum(data_mask_n.view(-1, 37), dim=1)

        mask_sum[mask_sum == 0] = 1 #when the sum is 0, reset it to 1, otherwise, it will be the sum of the missing value (n*1)
        # print(f'mask_sum shape:{mask_sum.shape}')torch.Size([11])
        mse = torch.sum(se, dim=1) / mask_sum #missing values will then be counted into calculating the mse as its denominator involves the total number of data points
        # print(f'mse shape:{mse.shape}')#
        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        # print(f'nll/mse shape before summing:{nll.shape}/{mse.shape}')
        # nll/mse shape before summing:torch.Size([33280, 11])/torch.Size([33280]) with bs=256,dim=11
        # other times, the data comes in, it will count its valid values excluding NAs, so there will be nll/mse shape before summing:torch.Size([9880, 11])/torch.Size([9880]) or torch.Size([8320, 11])/torch.Size([8320])
        nll=torch.sum(nll, dim=1) 
        # print(f'nll/mse shape after summing(nll for 1 sum):{nll.shape}/{mse.shape}')
        # ll/mse shape after summing(nll for 1 sum):torch.Size([33280])/torch.Size([33280])
        nll=torch.sum(nll.reshape(batch_size,-1),dim=1) #
        mse=torch.sum(mse.reshape(batch_size,-1),dim=1)
        # print(f'nll/mse shape after summing:{nll.shape}/{mse.shape}')
        # nll/mse shape after summing:torch.Size([256])/torch.Size([256])
        # print(f'mse/nll shape:{mse.shape}/{nll.shape}')# mse and nll shape:[256]
        return mse, nll




class SimpleVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        self.fc1 = nn.Linear(num_dim, 300)
        self.fc21 = nn.Linear(300, 30)
        self.fc211 = nn.Linear(30, latent_dim)
        self.fc221 = nn.Linear(30, latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.fc31 = nn.Linear(30, 300)
        self.fc4 = nn.Linear(300, num_dim)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc21(h1))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h4))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.num_dim))
        z = self.sample_latent(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        # original:mask_sum = torch.sum(data_mask_n.view(-1, 37), dim=1)

        mask_sum[mask_sum == 0] = 1 #when the sum is 0, reset it to 1, otherwise, it will be the sum of the missing value (n*1)
        mse = torch.sum(se, dim=1) / mask_sum #missing values will then be counted into calculating the mse as its denominator involves the total number of data points

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


if __name__ == "__main__":
    """
    This is used for pre-training.
    """

    # create parser and set variables
    opt = VAEArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert loss_function=='mse' or loss_function=='nll', ("Unknown loss function " + loss_function)
    assert ('T' in locals() and T is not None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    
    # set up dataset
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                             mask_file=mask_file, root_dir=data_source_path,
                                             transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDatasetConv(data_file=csv_file_data,
                                              label_file=csv_file_label,
                                              mask_file=mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                         mask_file=mask_file, root_dir=data_source_path,
                                         transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDataset(data_file=csv_file_data,
                                          label_file=csv_file_label,
                                          mask_file=mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())
        elif dataset_type == 'Physionet':
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path,data_format='3d')
    else:
        if dataset_type == 'Physionet':
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path,data_format='3d')
            validation_dataset=PhysionetDataset(data_file=csv_file_val_data, root_dir=data_source_path,data_format='3d')
    if run_tests:
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='3d')
    print('Length of dataset:  {}'.format(len(dataset)))
    Q = len(dataset[0]['label'])

    # set up Data Loader
    # min(len(dataset),256)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    # print(f'...training batch size is 256 or the len of dataset if its smaller, equals varying length?....')
    print(f'...loading data with  batch size :{batch_size}...')
    vy = torch.Tensor(np.ones(num_dim) * vy_init)

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim, num_dim, vy, vy_fixed).to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim, num_dim, vy, vy_fixed).to(device)

    elif type_nnet=='rnn':
        if type_rnn=='lstm':
            print(f'Using {type_nnet}-{type_rnn}')
            nnet_model=LSTMVAE( hidden_dim, num_layers, dropout, latent_dim, num_dim, vy, vy_fixed).to(device)

    
    
    optimiser = torch.optim.Adam(nnet_model.parameters(), lr=1e-3)

    print(nnet_model.vy)
    out_path=f'{results_path}-{marker}'
    if training:
        if model_params:
            try:
                nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')),strict=False)
                print(f'... pre-trained model loaded...')
            except:
                print(f'...loading pre-trained model failed...')

        net_train_loss = np.empty((0, 1))
        
        best_val_loss=np.inf
        best_epo_list=[]
        os.makedirs(save_path,exist_ok=True)
        for epoch in range(1, epochs + 1):

            # start training VAE
            nnet_model.train()
            train_loss = 0
            recon_loss_sum = 0
            nll_loss = 0
            kld_loss = 0

            for batch_idx, sample_batched in enumerate(dataloader):
                if dataset_type == 'Physionet':
                    data = sample_batched['data']
                else:
                    data = sample_batched['digit']
                data = data.to(device)                                  # send to GPU
                mask = sample_batched['mask']
                mask = mask.to(device)
                label = sample_batched['label'].to(device)

                optimiser.zero_grad()                                   # clear gradients

                recon_batch, mu, log_var = nnet_model(data)

                [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
                if type_nnet=='rnn':
                    KLD=-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=0)
                # print(f'KLD size:{KLD.shape}')
                    KLD=KLD.sum(-1)               
                else:
                    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                if loss_function == 'nll':
                    loss = torch.sum(nll + KLD)
                elif loss_function == 'mse':
                    # print(f'recon_loss/kld shape:{recon_loss.shape}/{KLD.shape}') #
                    # 
                    # recon_loss/kld shape neesd to be:torch.Size([256)/torch.Size([256]) if batch size =256
                    loss = torch.sum(recon_loss + KLD)

                loss.backward()                                         # compute gradients
                train_loss += loss.item()
                recon_loss_sum += recon_loss.sum().item()
                nll_loss += nll.sum().item()
                kld_loss += KLD.sum().item()
                
                optimiser.step()
            if epoch%save_cadence==0:print('====> Epoch: {} - Average loss: {:.4f}  - KLD loss: {:.3f}  - NLL loss: {:.3f}  - Recon loss: {:.3f}'.format(epoch, train_loss, kld_loss, nll_loss, recon_loss_sum))
            val_loss, kld_loss, nll_loss, recon_loss_sum=vae_validate(nnet_model,validation_dataset,batch_size,num_workers,dataset_type,type_nnet,loss_function)
            if epoch%save_cadence==0:print('====> Epoch: {}  Average val loss: {:.4f}  - val KLD loss: {:.3f}  -val NLL loss: {:.3f}  -val Recon loss: {:.3f}'.format(epoch,val_loss, kld_loss, nll_loss, recon_loss_sum))
            if val_loss<best_val_loss:
                prev_best=copy.deepcopy(best_val_loss)
                best_val_loss=val_loss
                best_epo_list.append(epoch)
                filepath=f'{save_path}/{marker}_vae_epoch{epoch}_val_loss{best_val_loss:.4f}.pth'
                
                torch.save(nnet_model.state_dict(),filepath)
                if len(best_epo_list)>=2 and best_epo_list[-1]>best_epo_list[-2]:
                    os.remove(f'{save_path}/{marker}_vae_epoch{best_epo_list[-2]}_val_loss{prev_best:.4f}.pth')
                net_train_loss = np.append(net_train_loss, train_loss)
            
            if epoch % save_cadence == 0:
                    print(nnet_model.vy)

                
            if epoch%save_cadence==0:print(f'..best epoch is {best_epo_list[-1]} ...')
            if epoch%save_cadence==0:print(f'...saving final model into {save_path}/{marker}_vae_epoch{best_epo_list[-1]}_val_loss{best_val_loss:.4f}.pth...') 
    else:
        assert model_params!='', 'please specify your model path to test!'
        # try:
        if model_params:
            nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')),strict=False)
            # nnet_model = nnet_model.double().to(device) 
        # except:
            # print('..loading pre-trained model failed..')
 

    if run_tests:
        print(f'...running test and get decoder loss...')
        # VAEtest(batch_size,num_workers,dataset_type,test_dataset, nnet_model, type_nnet, id_covariate)
        X_df,recon_df,gen_w_stu = VAEoutput(batch_size,num_workers,out_path,ref_df_path,dataset_type,nnet_model, test_dataset,type_nnet)


    print(nnet_model.vy)