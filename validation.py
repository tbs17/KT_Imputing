from torch.utils.data import DataLoader

import torch

from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from elbo_functions import deviance_upper_bound, elbo
from utils import batch_predict_varying_T
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
def validation_dubo(type_nnet,latent_dim, covar_module0, covar_module1, likelihood, train_xt, m, log_v, z, P, T, eps):
    """
    Efficient KL divergence using the variational mean and variance instead of a sample from the latent space (DUBO).
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch_dtype = torch.double
    v = torch.exp(log_v)
    # print(f'P/T length:{P}/{T}')
    # P=len(train_xt)//T
    # reshape_dim=P* T
    # train_xt=train_xt[:reshape_dim]
    # v = v[:reshape_dim]
    # m=m[:reshape_dim]
    if type_nnet=='rnn':
        T=1
        P=len(train_xt)
    x_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[1], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.linalg.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[1], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(stacked_x_st, stacked_x_st).evaluate().transpose(0,1)
    B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + torch.eye(T, dtype=torch.double).to(device) * likelihood.noise_covar.noise.unsqueeze(dim=2)).transpose(0,1)
    LB_st = torch.linalg.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device), LB_st)
    # print(f'm/v shape:{m.shape}/{v.shape}')
    dubo_sum = torch.tensor([0.0]).double().to(device)
    for i in range(latent_dim):
        m_st = torch.reshape(m[:, i], [P, T, 1]).to(device)
        v_st = torch.reshape(v[:, i], [P, T]).to(device)
        K0xz_st = torch.reshape(K0xz[i], [P, T, K0xz.shape[2]]).to(device)
        iB_K0xz = torch.matmul(iB_st[i], K0xz_st).to(device)
        K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz[i], 0, 1), torch.reshape(iB_K0xz, [P*T, K0xz.shape[2]])).to(device)
        W = K0zz[i] + K0zx_iB_K0xz
        W = (W + W.T) / 2
        LW = torch.linalg.cholesky(W).to(device)
        logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz[i]))).to(device)
        logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st[i], dim1=-2, dim2=-1))).to(device)
        logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
        logDetSigma = -logDetK0zz + logDetB + logDetW
        # print(f'B_st[i] and m_st shape:{B_st[i].shape}/{m_st.shape}') #B_st[i] and m_st shape:torch.Size([10, 20, 20])/torch.Size([10, 20, 1])
        iB_m_st = torch.linalg.solve(B_st[i],m_st).to(device) #<---bug on the old function removed causing runtime error, also needs to change the order of the original A and B matrices
    #  
        t=torch.linalg.solve(B_st[i],m_st)
        # print(f'torch.linalg.solve(m_st,B_st[i]) shape: {t.shape}')
        # print(f'iB_m_st.shape:{iB_m_st.shape}') #torch.Size([10, 20, 1])
        qF1 = torch.sum(m_st*iB_m_st).to(device)
        p = torch.matmul(K0xz[i].T, torch.reshape(iB_m_st, [P * T])).to(device)
        qF2 = torch.sum(torch.linalg.solve_triangular( LW,p[:,None], upper=False)[0] ** 2).to(device) #the updated function torch.linalg.solve_triangular has its arguments reversed and does not return a copy of one of the inputs.
        qF = qF1 - qF2
        tr = torch.sum(iB_st[i] * K0_st[i]) - torch.sum(K0zx_iB_K0xz * iK0zz[i])
        logDetD = torch.sum(torch.log(v[:, i])).to(device)
        tr_iB_D = torch.sum(torch.diagonal(iB_st[i], dim1=-2, dim2=-1)*v_st).to(device)
        D05_iB_K0xz = torch.reshape(iB_K0xz*torch.sqrt(v_st)[:,:,None], [P*T, K0xz.shape[2]])
        K0zx_iB_D_iB_K0zx = torch.matmul(torch.transpose(D05_iB_K0xz,0,1), D05_iB_K0xz).to(device)
        tr_iB_K0xz_iW_K0zx_iB_D = torch.sum(torch.diagonal(torch.cholesky_solve(K0zx_iB_D_iB_K0zx, LW))).to(device)
        tr_iSigma_D = tr_iB_D - tr_iB_K0xz_iW_K0zx_iB_D
        dubo = 0.5*(tr_iSigma_D + qF - P*T + logDetSigma - logDetD + tr)
        dubo_sum = dubo_sum + dubo
    return dubo_sum



def validate(varying_T,subjects_per_batch,num_workers,dataset_type,nnet_model, type_nnet, dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, 
             zt_list, P,T, weight,id_covariate, loss_function, eps=1e-6):
             
    """
    Obtain KL divergence of validation set.
    
    :param nnet_model: neural network model
    :param type_nnet: type of encoder/decoder
    :param dataset: dataset to use
    :param type_KL: type of KL divergence computation
    :param num_samples: number of samples
    :param latent_dim: number of latent dimensions
    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihoods: GPyTorch likelihood model
    :param zt_list: list of inducing points
    :param T: number of timepoints
    :param weight: value for the weight
    :param train_mu: mean on training set
    :param id_covariate: covariate number of the id
    :param loss_function: selected loss function
    :param eps: jitter
    :return: KL divergence between variational distribution 
    """

    # print("Testing the model with a validation set")
    # T=16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (type_KL == 'GPapprox_closed' or type_KL == 'GPapprox')

    # set up Data Loader for training
    if type_nnet=='rnn':
   
        if varying_T:
        
            N_batches = (P + subjects_per_batch - 1)//subjects_per_batch
            actual_data_len=subjects_per_batch*N_batches
            print(f'varying_T is true, you are using VaryingLengthSubjectSampler to form a  dataloader with n_batches= (P + subjects_per_batch - 1)//subjects_per_batch :{N_batches}')
            dataloader = DataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=num_workers)    
        else:
            
            
            P_val=len(dataset)//T
            # print(f'...varying_T is false, you are using subject sampler to sample {P_val} val students...')
            batch_size=subjects_per_batch*T
            actual_data_len=P_val*T
            N_batches=actual_data_len//batch_size
            valid_len=N_batches*batch_size
            # print(f'..loading data with actual_data_len: {actual_data_len} and batch size=subjects_per_batch*T {batch_size} and N_batches=actual_data_len//batch_size={N_batches}...')
    
            dataloader = DataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P_val, T), batch_size, drop_last=True), num_workers=num_workers)

            
    else:
        dataloader = DataLoader(dataset, batch_size=T, shuffle=False, drop_last=True,num_workers=num_workers)
    # print(f'length of validation dataset is {len(dataset)}')
    len_loader=N_batches
    # print(f'length of validation dataloader after sampling is {N_batches}')
    cadence=len_loader//2
    Q = len(dataset[0]['label'])
    # P = len(dataset) // T
    
    full_mu = torch.zeros(actual_data_len, latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_log_var = torch.zeros(actual_data_len, latent_dim, dtype=torch.double, requires_grad=True).to(device)
    full_labels = torch.zeros(actual_data_len, Q, dtype=torch.double, requires_grad=False).to(device)

    recon_loss_sum = 0
    nll_loss_sum = 0
    nnet_model.eval()
    covar_module0.eval()
    covar_module1.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(dataloader,total=N_batches)):
            label=sample_batched['label']
            indices = sample_batched['idx']
            mask = sample_batched['mask']
            if dataset_type=='Physionet':
                data = sample_batched['data']
            else:
                data = sample_batched['digit']
            if type_nnet=='rnn':

                mask=mask.reshape(subjects_per_batch,T,mask.shape[-1])
                data = data.reshape(subjects_per_batch,T,data.shape[-1])

            data= data.double().to(device)
            mask = mask.double().to(device)
            full_labels[indices] = label.double().to(device)
            # print(f'indices within a batch:{len(indices)}')
            # covariates = torch.cat((full_labels[indices, :id_covariate], full_labels[indices, id_covariate+1:]), dim=1)
            recon_batch, mu, log_var = nnet_model(data)
            # if batch_idx%cadence==0:print(f'===validation batch: {batch_idx}: recon_batch/data/mask shape:{recon_batch.shape}/{data.shape}/{mask.shape}==')
            if type_nnet=='rnn':
                # print(f'mu after NN shape:{mu.shape}')
                mu=mu.mean(0)
                # print(f'mu after NN shape after mean:{mu.shape}')
                mu=torch.stack([mu for i in range (T)],dim=1).reshape(-1,latent_dim)
                # print(f'mu after NN shape after mean after stack:{mu.shape}')
                
                # print(f'log_var after NN shape:{log_var.shape}')
                log_var=log_var.mean(0)
                # print(f'log_var after NN shape after mean:{log_var.shape}')
                log_var=torch.stack([log_var for i in range (T)],dim=1).reshape(-1,latent_dim)
                # print(f'log_var after NN shape after mean after stack:{log_var.shape}')
            
            full_mu[indices] = mu
            full_log_var[indices] = log_var

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll = torch.sum(nll)

            recon_loss_sum = recon_loss_sum + recon_loss.item()
            nll_loss_sum = nll_loss_sum + nll.item()
    if type_nnet=='rnn':

        full_log_var=full_log_var[:valid_len]
        full_mu=full_mu[:valid_len]
        full_labels=full_labels[:valid_len]
        
    gp_losses = 0
    gp_loss_sum = 0
    param_list = []
    # print(f'full_mu/full_log_var/full_labels shape:{full_mu.shape}/{full_log_var.shape}/{full_labels.shape}')
    if type_nnet=='rnn':
        full_mu=full_mu.reshape(N_batches*subjects_per_batch,T,full_mu.shape[-1]).mean(1)
        full_log_var=full_log_var.reshape(N_batches*subjects_per_batch,T,full_log_var.shape[-1]).mean(1)
        full_labels=full_labels.reshape(N_batches*subjects_per_batch,T,full_labels.shape[-1]).mean(1)
    if isinstance(covar_module0, list):
        if type_KL == 'GPapprox':
            for sample in range(0, num_samples):
                Z = nnet_model.sample_latent(full_mu, full_log_var)
                for i in range(0, latent_dim):
                    Z_dim = Z[:, i]
                    gp_loss = -elbo(covar_module0[i], covar_module1[i], likelihoods[i], full_labels, Z_dim,
                                    zt_list[i].to(device), P, T, eps)
                    gp_loss_sum = gp_loss.item() + gp_loss_sum
            gp_loss_sum /= num_samples

        elif type_KL == 'GPapprox_closed':
            for i in range(0, latent_dim):
                mu_sliced = full_mu[:, i]
                log_var_sliced = full_log_var[:, i]
                gp_loss = deviance_upper_bound(covar_module0[i], covar_module1[i],
                                               likelihoods[i], full_labels,
                                               mu_sliced, log_var_sliced,
                                               zt_list[i].to(device), P,
                                               T, eps)
                gp_loss_sum = gp_loss.item() + gp_loss_sum
    else:
        if type_KL == 'GPapprox_closed':
            gp_loss = validation_dubo(type_nnet,latent_dim, covar_module0, covar_module1,
                                      likelihoods, full_labels,
                                      full_mu, full_log_var,
                                      zt_list, P_val, T, eps)
            gp_loss_sum = gp_loss.item()

    if loss_function == 'mse':
        gp_loss_sum /= latent_dim
        net_loss_sum = weight*gp_loss_sum + recon_loss_sum
    elif loss_function == 'nll':
        net_loss_sum = gp_loss_sum + nll_loss_sum

    #Do logging
    print('Validation set - Loss: %.3f  - GP loss: %.3f  - NLL loss: %.3f  - Recon Loss: %.3f' % (
        net_loss_sum, gp_loss_sum, nll_loss_sum, recon_loss_sum))

    return net_loss_sum




def vae_validate(nnet_model,validation_dataset,batch_size,num_workers,dataset_type,type_nnet,loss_function):
    nnet_model.eval()
    val_loss = 0
    recon_loss_sum = 0
    nll_loss = 0
    kld_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for batch_idx, sample_batched in enumerate(validation_dataloader):
            if dataset_type == 'Physionet':
                data = sample_batched['data']
            else:
                data = sample_batched['digit']
            data = data.to(device)                                  # send to GPU
            mask = sample_batched['mask']
            mask = mask.to(device)
            label = sample_batched['label'].to(device)
            recon_batch, mu, log_var = nnet_model(data)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            # print('Validation Decoder loss: ' + str(torch.mean(recon_loss)))
            if type_nnet=='rnn':
                KLD=-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=0)
                KLD=KLD.sum(-1)
            
            else:
                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            if loss_function == 'nll':
                loss = torch.sum(nll + KLD)
            elif loss_function == 'mse':
                loss = torch.sum(recon_loss + KLD)
            val_loss+=loss.item()
            recon_loss_sum += recon_loss.sum().item()
            nll_loss += nll.sum().item()
            kld_loss += KLD.sum().item()
    
    return val_loss, kld_loss, nll_loss, recon_loss_sum
