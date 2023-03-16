import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from dataset_def import HealthMNISTDatasetConv, HealthMNISTDataset,PhysionetDataset
from utils import batch_predict, batch_predict_varying_T, min_max_scaler
from torch.utils.data.sampler import BatchSampler
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader

def gen_rotated_mnist_plot(X, recon_X, labels, seq_length=16, num_sets=3, save_file='recon.pdf'):
    """
    Function to generate rotated MNIST digits plots.
    
    """
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    fig.set_size_inches(9, 1.5 * num_sets)
    for j in range(num_sets):
        begin = seq_length * j
        end = seq_length * (j + 1)
        time_steps = labels[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')

def gen_rotated_mnist_seqrecon_plot_old(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate rotated MNIST digits.
    
    """
    num_sets = 4
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(3 * num_sets, 3 * num_sets)

    for j in range(num_sets):
        begin = seq_length_train * j
        end = seq_length_train * (j + 1)
        time_steps = labels_train[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')

        begin = seq_length_full * j
        end = seq_length_full * (j + 1)
        time_steps = labels_recon[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')


def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate Health MNIST digits.
    
    """    
    num_sets = 8
    fig, ax = plt.subplots(4 * num_sets - 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)

    for j in range(num_sets):
        begin_data = seq_length_train*j
        end_data = seq_length_train*(j+1)

        begin_label = seq_length_full*2*j
        mid_label = seq_length_full*(2*j+1)
        end_label = seq_length_full*2*(j+1)
        
        time_steps = labels_train[begin_data:end_data, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j, int(t)].imshow(np.reshape(X[begin_data + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[begin_label:mid_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 1, int(t)].imshow(np.reshape(recon_X[begin_label + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[mid_label:end_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 2, int(t)].imshow(np.reshape(recon_X[mid_label + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')

def recon_complete_gen_old(num_workers,dataset_type,generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                       covar_module1, likelihoods, latent_dim, ref_df_path, prediction_x, 
                       prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)
    recon_df=pd.DataFrame()
    X_df=pd.DataFrame()
    label_df=pd.DataFrame()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data=sample_batched['data']
            else:
                data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)
            if type_nnet=='rnn':
                Z_pred=torch.stack([Z_pred,Z_pred],dim=0)
                # print(f'data/z_pred shape:{data.shape}/{Z_pred.shape}')
            recon_Z = nnet_model.decode(data,Z_pred)

            
            # filename = 'recon_complete' if epoch == -1 else 'recon_complete_best'
            
            marker=results_path.split('-')[-1]
            # save_dir='/'.join(results_path.split('/')[:-1])
            save_dir=results_path.split('-')[0]
            if '_' in marker:
                domain=marker.split('_')[0]
            else:
                domain=marker
            print(f'save_dir/marker/domain:{save_dir},{marker},{domain}')

            
            orig_cols=['seq_number', 'assessment_duration','bktcount', 'count', 'quest_difficulty',
       'item_difficulty', 'usmo_difficulty', 'quest_ref_1', 'item_ref_1',
       'usmoid_1', 'question_type_1']
            if type_nnet=='rnn':
                data=data.reshape(-1,data.shape[2])
                recon_Z=recon_Z.reshape(-1,recon_Z.shape[2])
            X=pd.DataFrame(data.detach().cpu().numpy())
            X.columns=orig_cols
            recon_X=pd.DataFrame(recon_Z.detach().cpu().numpy())#recon_Z[0:160, :]
            recon_X.columns=orig_cols

            orig_df_path,label_orig_path=ref_df_path.split('-')[0],ref_df_path.split('-')[1]
            print(f'reference data path {orig_df_path}\n'
            f'reference label path:{label_orig_path}')
            orig_df=pd.read_csv(orig_df_path)
            is_reverse=True
            X_transformed=min_max_scaler(X,orig_df,is_reverse)
            recon_X_transformed=min_max_scaler(recon_X,orig_df,is_reverse)
            X_df=X_df.append(X_transformed)
            recon_df=recon_df.append(recon_X_transformed)

            # append label data...
            labels_recon=pd.DataFrame(label.detach().cpu().numpy())
            labels_recon.columns=['event_time', 'LE_SCHOOL_ID', 'SPECIAL_ED', 'student_id',
    'FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL', 'correct',
    'score_rate']
            label_df=label_df.append(labels_recon)
        print('>>>\nBelow is the original data stats\n')
        display(X_df.head())
        display(X_df.describe())
        print('>>>\nBelow is the original data nunique\n')
        display(X_df.nunique())
        
        print('>>>\nBelow is the generated data stats\n')
        display(recon_df.head())
        display(recon_df.describe())
        print('>>>\nBelow is the generated data nunique\n')
        display(recon_df.nunique())
        out_dir=f'{save_dir}/generate'
        os.makedirs(out_dir,exist_ok=True)
        print(f'..saving reconstructed x and covariate info used into {out_dir}/{marker}_X.csv and {out_dir}/{marker}_ReconX.csv and {out_dir}/{marker}_Reconlabel.csv ....')
        X_df.to_csv(f'{out_dir}/{marker}_X.csv',index=False)
        recon_df.to_csv(f'{out_dir}/{marker}_ReconX.csv',index=False)
        # X_df.to_csv(f'{out_dir}/{marker}_X.csv',index=False)
        label_df.to_csv(f'{out_dir}/{marker}_Reconlabel.csv',index=False)
        print(f'..saving matched reconstructed x and covariate info into {out_dir}/{marker}_ReconX_matched.csv ....')
        # data/k12/split_data/geom_no_outlr_data_readings.csv
        
        label_orig=pd.read_csv(label_orig_path)
        label_orig_n=label_orig[['event_time','student_id']].iloc[:recon_df.shape[0]]
        gen_w_stu=pd.concat([recon_df,label_orig_n],axis=1)
        gen_w_stu['score_rate']=np.nan
        gen_w_stu.to_csv(f'{out_dir}/{marker}_ReconX_matched.csv',index=False)
    return X_df,recon_df,gen_w_stu




def recon_complete_gen(subjects_per_batch,num_workers,dataset_type,generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                       covar_module1, likelihoods, latent_dim, ref_df_path, prediction_x, 
                       prediction_mu,  zt_list, P_pred, T, id_covariate, varying_T=False):
   
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(' Length of Generating data:  {}'.format(len(generation_dataset)))
    if type_nnet=='rnn':
        
        # dataloader_full = DataLoader(generation_dataset, batch_size=batch_size, shuffle=False, drop_last=True,num_workers=num_workers)
        if varying_T:
            n_batches = (P_gen + subjects_per_batch - 1)//subjects_per_batch
            print(f'varying_T is true, you are using VaryingLengthSubjectSampler to form a  dataloader with n_batches= (P + subjects_per_batch - 1)//subjects_per_batch :{n_batches}')
            dataloader_full = DataLoader(generation_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(generation_dataset, id_covariate), subjects_per_batch), num_workers=num_workers)    
        else:
            P_gen=len(generation_dataset)//T
            print(f'varying_T is false, you are using subject sampler to extract {P_gen} students')
            batch_size=subjects_per_batch*T
            actual_data_len=P_gen*T
            N_batches=actual_data_len//batch_size
            print(f'..loading data with batch size=subjects_per_batch*T {batch_size} and N_batches=actual_data_len//batch_size={N_batches}...')

            dataloader_full = DataLoader(generation_dataset, batch_sampler=BatchSampler(SubjectSampler(generation_dataset, P_gen, T), batch_size, drop_last=True), num_workers=num_workers)
    else:
        dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)
    len_loader=N_batches
    cadence=len_loader//2
    print(f'...len of dataloader is :{len_loader} with printing cadence  {cadence}....')
    recon_df=pd.DataFrame()
    X_df=pd.DataFrame()
    label_df=pd.DataFrame()
                
    marker=results_path.split('-')[-1]
    save_path=results_path.split('-')[0]
    save_dir='/'.join(save_path.split('/')[:-1])
    if '_' in marker:
        domain=marker.split('_')[0]
    else:
        domain=marker
    print(f'save_dir/marker/domain:{save_dir},{marker},{domain}')
    orig_cols=['seq_number', 'assessment_duration','bktcount', 'count', 'quest_difficulty',
       'item_difficulty', 'usmo_difficulty', 'quest_ref_1', 'item_ref_1',
       'usmoid_1', 'question_type_1']
    orig_df_path,label_orig_path=ref_df_path.split('-')[0],ref_df_path.split('-')[1]
    print(f'reference data path {orig_df_path}\n'
    f'reference label path:{label_orig_path}')
    orig_df=pd.read_csv(orig_df_path)
    nnet_model.eval()
    covar_module0.eval()
    covar_module1.eval()
    likelihoods.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(dataloader_full,total=N_batches)):

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data=sample_batched['data']
            mask = sample_batched['mask']
            if type_nnet=='rnn':
                data=data.reshape(subjects_per_batch,T,data.shape[-1])
                mask=mask.reshape(subjects_per_batch,T,mask.shape[-1])
            else:
                data = sample_batched['digit']
            data = data.double().to(device)
            mask = mask.to(device)


            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
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
            
            if type_nnet=='rnn':
                data=data.reshape(-1,data.shape[2])
                recon_Z=recon_Z.reshape(-1,recon_Z.shape[2])

            X=pd.DataFrame(data.detach().cpu().numpy())
            X.columns=orig_cols
            recon_X=pd.DataFrame(recon_Z.detach().cpu().numpy())#recon_Z[0:160, :]
            recon_X.columns=orig_cols

            X_df=X_df.append(X)
            recon_df=recon_df.append(recon_X)

            # append label data...
            labels_recon=pd.DataFrame(label.detach().cpu().numpy())
            labels_recon.columns=['event_time', 'LE_SCHOOL_ID', 'SPECIAL_ED', 'student_id',
    'FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL', 'correct',
    'score_rate']
            label_df=label_df.append(labels_recon)
        is_reverse=True
        X_df=min_max_scaler(X_df,orig_df,is_reverse)
        recon_df=min_max_scaler(recon_df,orig_df,is_reverse)
        print('>>>\nBelow is the original data stats\n')
        display(X_df.head())
        display(X_df.describe())
        print('>>>\nBelow is the original data nunique\n')
        display(X_df.nunique())
        
        print('>>>\nBelow is the generated data stats\n')
        display(recon_df.head())
        display(recon_df.describe())
        print('>>>\nBelow is the generated data nunique\n')
        display(recon_df.nunique())
        out_dir=f'{save_dir}/generate'
        os.makedirs(out_dir,exist_ok=True)
        recon_df=recon_df.reset_index(drop=True)
        X_df=X_df.reset_index(drop=True)
        print(f'..saving reconstructed x and covariate info used into {out_dir}/{marker}_X.csv and {out_dir}/{marker}_ReconX.csv and {out_dir}/{marker}_Reconlabel.csv ....')
        X_df.to_csv(f'{out_dir}/{marker}_X.csv',index=False)
        recon_df.to_csv(f'{out_dir}/{marker}_ReconX.csv',index=False)
        label_df.to_csv(f'{out_dir}/{marker}_Reconlabel.csv',index=False)
        print(f'..saving matched reconstructed x and covariate info into {out_dir}/{marker}_ReconX_matched.csv ....')

        
        label_orig=pd.read_csv(label_orig_path)
        label_orig_n=label_orig[['event_time','student_id']].iloc[:recon_df.shape[0]].reset_index(drop=True)
        gen_w_stu=pd.concat([recon_df,label_orig_n],axis=1)
        gen_w_stu['score_rate']=np.nan
        gen_w_stu.to_csv(f'{out_dir}/{marker}_ReconX_matched.csv',index=False)
    return X_df,recon_df,gen_w_stu



def variational_complete_gen(dataset_type,generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                             covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                             prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data'].double().to(device)
            else:
                data = sample_batched['digit'].double().to(device)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)
            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete_' + str(epoch) + '.pdf'))

def VAEoutput(batch_size,num_workers,results_path,ref_df_path,dataset_type,nnet_model, dataset,type_nnet):
    """
    Function to obtain output of VAE.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Batch size must be a multiple of T
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    X_df=pd.DataFrame()
    recon_df=pd.DataFrame()
    label_df=pd.DataFrame()
    nnet_model.eval()
    orig_df_path,label_orig_path=ref_df_path.split('-')[0],ref_df_path.split('-')[1]
    print(f'reference data path {orig_df_path}\n'
    f'reference label path:{label_orig_path}')
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(dataloader,total=len(dataloader))):
            # no mini-batching. Instead get a mini-batch of size 4000
            label = sample_batched['label'].to(device)
            if dataset_type=='Physionet':
                data = sample_batched['data'].to(device)
            else:
                data = sample_batched['digit'].to(device)

            recon_Z, mu, log_var = nnet_model(data)

   
            marker=results_path.split('-')[-1]
       
            save_dir=results_path.split('-')[0]
            if '_' in marker:
                domain=marker.split('_')[0]
            else:
                domain=marker
   

            
            orig_cols=['seq_number', 'assessment_duration','bktcount', 'count', 'quest_difficulty',
       'item_difficulty', 'usmo_difficulty', 'quest_ref_1', 'item_ref_1',
       'usmoid_1', 'question_type_1']
            if type_nnet=='rnn':
                data=data.reshape(-1,data.shape[2])
                recon_Z=recon_Z.reshape(-1,recon_Z.shape[2])
            X=pd.DataFrame(data.detach().cpu().numpy())
            X.columns=orig_cols
            recon_X=pd.DataFrame(recon_Z.detach().cpu().numpy())#recon_Z[0:160, :]
            recon_X.columns=orig_cols


            X_df=X_df.append(X)
            recon_df=recon_df.append(recon_X)

            # append label data...
            labels_recon=pd.DataFrame(label.detach().cpu().numpy())
            labels_recon.columns=['event_time', 'LE_SCHOOL_ID', 'SPECIAL_ED', 'student_id',
    'FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL', 'correct',
    'score_rate']
            label_df=label_df.append(labels_recon)
        orig_df=pd.read_csv(orig_df_path)
        is_reverse=True
        X_df=min_max_scaler(X_df,orig_df,is_reverse)
        recon_df=min_max_scaler(recon_df,orig_df,is_reverse)
        print('>>>\nBelow is the original data stats\n')
        display(X_df.head())
        display(X_df.describe())
        print('>>>\nBelow is the original data nunique\n')
        display(X_df.nunique())
        
        print('>>>\nBelow is the generated data stats\n')
        display(recon_df.head())
        display(recon_df.describe())
        print('>>>\nBelow is the generated data nunique\n')
        display(recon_df.nunique())
        out_dir=f'{save_dir}/generate'
        os.makedirs(out_dir,exist_ok=True)
        print(f'..saving reconstructed x and covariate info used into {out_dir}/{marker}_X.csv and {out_dir}/{marker}_ReconX.csv and {out_dir}/{marker}_Reconlabel.csv ....')
        X_df.to_csv(f'{out_dir}/{marker}_X.csv',index=False)
        recon_df.to_csv(f'{out_dir}/{marker}_ReconX.csv',index=False)

        label_df.to_csv(f'{out_dir}/{marker}_Reconlabel.csv',index=False)
        print(f'..saving matched reconstructed x and covariate info into {out_dir}/{marker}_ReconX_matched.csv ....')

        
        label_orig=pd.read_csv(label_orig_path)
        label_orig_n=label_orig[['event_time','student_id']].iloc[:recon_df.shape[0]].reset_index(drop=True)
        recon_df=recon_df.reset_index(drop=True)
        gen_w_stu=pd.concat([recon_df,label_orig_n],axis=1)
        gen_w_stu['score_rate']=np.nan
        gen_w_stu.to_csv(f'{out_dir}/{marker}_ReconX_matched.csv',index=False)
    return  X_df,recon_df,gen_w_stu   

def predict_generate(num_workers,dataset_type,csv_file_test_data, csv_file_test_label, csv_file_test_mask, dataset, generation_dataset, nnet_model, results_path, covar_module0, covar_module1, likelihoods, type_nnet, latent_dim, data_source_path, prediction_x, prediction_mu, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to perform prediction and visualise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    print('Length of dataset:  {}'.format(len(dataset)))

    # set up Data Loader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers)

    # Get values for GP initialisation:
    Z = torch.tensor([]).to(device)
    mu = torch.tensor([]).to(device)
    log_var = torch.tensor([]).to(device)
    data_train = torch.tensor([]).to(device)
    label_train = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
          
            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data'].double().to(device)
            else:
                data = sample_batched['digit'].double().to(device)

            label_train = label
            data_train = data
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            gen_rotated_mnist_plot(data[40:100, :].cpu(), recon_batch[40:100, :].cpu(), label[40:100, :].cpu(), seq_length=20,
                                   save_file=os.path.join(results_path, 'recon_train.pdf'))
            break

    if type_nnet == 'conv':
        test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                              csv_file_label=csv_file_test_label,
                                              mask_file=csv_file_test_mask, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type=='Physionet':
            test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=csv_file_test_mask, root_dir=data_source_path,
                                          transform=transforms.ToTensor())
        else:
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path)
    elif type_nnet=='rnn':
        if dataset_type == 'Physionet':
            test_dataset = PhysionetDataset(data_file=csv_file_test_data, root_dir=data_source_path,T=T)
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
  
            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data'].double().to(device)
            else:
                data = sample_batched['digit'].double().to(device)

            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            np.savetxt(os.path.join(results_path, 'result_error.csv'), pred_results)

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
       

            label_id = sample_batched['idx']
            label = sample_batched['label']
            if dataset_type=='Physionet':
                data = sample_batched['data'].double().to(device)
            else:
                data = sample_batched['digit'].double().to(device)
      
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete.pdf'))

