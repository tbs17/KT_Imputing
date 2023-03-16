import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_def import HealthMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDatasetConv,PhysionetDataset
from utils import batch_predict, batch_predict_varying_T
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
def predict_gp(kernel_component, full_kernel_inverse, z):
    """
    Function to compute predictive mean

    """
    mean = torch.matmul(torch.matmul(kernel_component, full_kernel_inverse), z)
    return mean

def MSE_test(T,num_workers,dataset_type,csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
             covar_module, likelihoods, save_path, latent_dim, prediction_x, prediction_mu):
    
    """
    Function to compute Mean Squared Error of test set.
    
    """
    print("Running tests with a test set")
    results_path=save_path.split('_')[0]
    marker=save_path.split('_')[1]
    if '_' in marker:
        domain=marker.split('_')[0]
    else:
        domain=marker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                              csv_file_label=csv_file_test_label,
                                              mask_file=test_mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'Physionet':
            test_dataset= PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='2d')
        else:
            test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor()) 
    elif type_nnet=='rnn':
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='2d')
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
       

            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data']
            else:
                data = sample_batched['digit']
      
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))
            if(prediction_x.shape[0] > 6040):
                r = np.random.choice(prediction_x.shape[0], 6000, replace=False) + 40
                ind = np.concatenate((np.arange(40), r))
                prediction_x = prediction_x[ind]
                prediction_mu = prediction_mu[ind]

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            for i in range(0, latent_dim):
                covar_module[i].eval()
                likelihoods[i].eval()
                K1 = covar_module[i](prediction_x.to(device), prediction_x.to(device)).evaluate() \
                                     + likelihoods[i].noise * torch.eye(prediction_mu.shape[0]).to(device)
                LK1 = torch.linalg.cholesky(K1)
                iK1 = torch.cholesky_solve(torch.eye(prediction_mu.shape[0], dtype=torch.double).to(device), LK1).to(device)
                kernel_component = covar_module[i](test_x.to(device), prediction_x.to(device)).evaluate()
                pred_means = predict_gp(kernel_component, iK1, prediction_mu[:, i])
                Z_pred = torch.cat((Z_pred, pred_means.view(-1, 1)), 1)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            print(f'...save recon_loss and recon_loss_GP test results under {results_path}/{marker}_result_error.csv...')
            np.savetxt(os.path.join(results_path, f'{marker}_result_error.csv'), pred_results)


def MSE_test_GPapprox_old(num_workers,dataset_type,csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                      covar_module0, covar_module1, likelihoods, save_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, P, T, id_covariate, varying_T=False, save_file='result_error.csv'):

    """
    Function to compute Mean Squared Error of test set with GP approximationö
    
    """
    results_path=save_path.split('-')[0]
    marker=save_path.split('-')[1]
    if '_' in marker:
        domain=marker.split('_')[0]
    else:
        domain=marker
    print(f'results_path/marker/domain:{results_path},{marker},{domain}')
    print("...Running tests with a test set...")
    # dataset_type = 'HealthMNIST'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            test_dataset = RotatedMNISTDatasetConv(data_file=csv_file_test_data,
                                                   label_file=csv_file_test_label,
                                                   mask_file=test_mask_file, root_dir=data_source_path,
                                                   transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'Physionet':
            test_dataset= PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='2d')
        else:
            test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())
    elif type_nnet=='rnn':
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='2d')
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(dataloader_test,total=len(dataloader_test))):


            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data']
            else:
                data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            # obtain the Z based on the testing data and posterior of Z from prediction samples
            # 
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
  
            if type_nnet=='rnn':
                Z_pred=torch.stack([Z_pred,Z_pred],dim=0)

            recon_Z = nnet_model.decode(data,Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            print(f'...save recon_loss and recon_loss_GP test results under {results_path}/{marker}_result_error_best.csv...')
            save_file=f'{marker}_{save_file}'
            np.savetxt(os.path.join(results_path, save_file), pred_results)
    return Z_pred, pred_results


def MSE_test_GPapprox(num_workers,dataset_type,csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet, nnet_model,
                      covar_module0, covar_module1, likelihoods, save_path, latent_dim, prediction_x, prediction_mu,
                      zt_list, subjects_per_batch, P_pred,T, id_covariate, varying_T=False, save_file='result_error.csv'):

    """
    Function to compute Mean Squared Error of test set with GP approximationö
    
    """
    results_path=save_path.split('-')[0]
    marker=save_path.split('-')[1]
    if '_' in marker:
        domain=marker.split('_')[0]
    else:
        domain=marker
    print(f'results_path/marker/domain:{results_path},{marker},{domain}')
    print("...Running tests with a test set...")
    # dataset_type = 'HealthMNIST'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                                  csv_file_label=csv_file_test_label,
                                                  mask_file=test_mask_file, root_dir=data_source_path,
                                                  transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            test_dataset = RotatedMNISTDatasetConv(data_file=csv_file_test_data,
                                                   label_file=csv_file_test_label,
                                                   mask_file=test_mask_file, root_dir=data_source_path,
                                                   transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'Physionet':
            test_dataset= PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path)
        else:
            test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())
    elif type_nnet=='rnn':
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,data_format='2d')
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    
    if varying_T:
        pass
    else:
        if type_nnet=='rnn':
            P_test=len(test_dataset)//T
            print(f'...varying_T is false, you are using subject sampler to sample {P_val} val students...')
            batch_size=subjects_per_batch*T
            actual_data_len=P_test*T
            N_batches=actual_data_len//batch_size
            valid_len=N_batches*batch_size
            print(f'..loading data with actual_data_len: {actual_data_len} and batch size=subjects_per_batch*T {batch_size} and N_batches=actual_data_len//batch_size={N_batches}...')
    
            dataloader_test = DataLoader(test_dataset, batch_sampler=BatchSampler(SubjectSampler(test_dataset, P_test, T), batch_size, drop_last=True), num_workers=num_workers)
        else:
            dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,num_workers=num_workers)
    len_loader=len(dataloader_test)
    cadence=len_loader//5
    print(f'...len of dataloader is :{len_loader} with printing cadence  {cadence}....')
    nnet_model.eval()
    covar_module0.eval()
    covar_module1.eval()
    likelihoods.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(dataloader_test,total=N_batches)):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            mask = sample_batched['mask']
            if type_nnet=='rnn':
                data = sample_batched['data']
                data=data.reshape(subjects_per_batch,T,data.shape[-1])
                mask=mask.reshape(subjects_per_batch,T,mask.shape[-1])
            else:
                data = sample_batched['digit']
            data = data.double().to(device)
            mask = mask.double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            

            test_x = label.type(torch.DoubleTensor).to(device)
            # obtain the Z based on the testing data and posterior of Z from prediction samples
            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            if varying_T:
                Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
            else:
                Z_pred=batch_predict(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, 
                 zt_list, P_pred, T, id_covariate, eps=1e-6)
            if type_nnet=='rnn':
                if varying_T:
                    pass
                else:
                    Z_pred=Z_pred.reshape(subjects_per_batch,T,latent_dim).mean(0)

                    Z_pred=torch.stack([Z_pred,Z_pred],dim=0)
            
                    data=data.reshape(T,subjects_per_batch,data.shape[-1])
            
    

            recon_Z = nnet_model.decode(data,Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
        print(f'Decoder Test Data recon_loss/recon_loss_GP: {str(torch.mean(recon_loss))}/{str(torch.mean(recon_loss_GP))}' )
        print(f'...save recon_loss and recon_loss_GP test results under {results_path}/{marker}_result_error.csv...')
        out_file=f'{marker}_{save_file}'
        np.savetxt(os.path.join(results_path, out_file), pred_results)
    return Z_pred, pred_results


def VAEtest(batch_size,num_workers,dataset_type,test_dataset, nnet_model, type_nnet, id_covariate):
    """
    Function to compute Mean Squared Error using just a VAE.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label'].to(device)
            if dataset_type=='Physionet':
                data = sample_batched['data']
            else:
                data = sample_batched['digit']
            data = data.to(device)
            mask = sample_batched['mask'].to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)
            # print(data.shape)
            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
    print('final Decoder loss: ' + str(torch.mean(recon_loss)))



def HMNIST_test(num_workers,nnet_model, type_nnet, latent_dim, prediction_x, prediction_mu, covar_module0, covar_module1, 
                likelihoods, csv_file_test_data, csv_file_test_label, test_mask_file, zt_list, P, T, id_covariate, 
                varying_T=False):
    """
    Function to compute Mean Squared Error of test set.
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=test_mask_file,
                                          root_dir='./data',
                                          transform=transforms.ToTensor())
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
