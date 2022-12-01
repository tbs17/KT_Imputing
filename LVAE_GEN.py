from multiprocessing import allow_connection_pickling
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from timeit import default_timer as timer

from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE, LSTMVAE
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset
from elbo_functions import elbo, KL_closed, deviance_upper_bound
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test, MSE_test_GPapprox
from predict_HealthMNIST import recon_complete_gen
from parse_model_args import ModelArgs
from training import hensman_training, minibatch_training, standard_training, variational_inference_optimization
from validation import validate
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader
from torch.utils.data.sampler import BatchSampler

eps = 1e-6

if __name__ == "__main__":
    """
    Root file for running L-VAE.
    
    Run command: python LVAE.py --f=path_to_config-file.txt 
    """

    # STEP 1:======= create parser and set variables========
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)





    assert not(hensman and mini_batch)
    assert loss_function=='mse' or loss_function=='nll', ("Unknown loss function " + loss_function)
    assert not varying_T or hensman, "varying_T can't be used without hensman"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))


    # STEP2: ================set up dataset=================

    if run_tests or generate_images:
        if dataset_type == 'HealthMNIST' and type_nnet == 'conv':
            prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_prediction_data,
                                                        csv_file_label=csv_file_prediction_label,
                                                        mask_file=prediction_mask_file, root_dir=data_source_path,
                                                        transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))
        elif dataset_type == 'RotatedMNIST':
            prediction_dataset = RotatedMNISTDatasetConv(data_file=csv_file_prediction_data,
                                                            label_file=csv_file_prediction_label,
                                                            mask_file=prediction_mask_file, root_dir=data_source_path,
                                                            transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))
        elif dataset_type == 'Physionet':
            if type_nnet=='simple':
                prediction_dataset = PhysionetDataset(data_file=csv_file_prediction_data, root_dir=data_source_path,data_format='2d')
            elif type_nnet=='rnn':
                prediction_dataset = PhysionetDataset(data_file=csv_file_prediction_data, root_dir=data_source_path,data_format='2d')
                pred_data=np.load(os.path.join(data_source_path,csv_file_prediction_data),allow_pickle=True)
                P_pred_varyL=len(np.unique(pred_data['outcome_attrib'][:,id_covariate]))
                print(f'..there are {P_pred_varyL} total students in the prediction set...')
    else:
        prediction_dataset = None

    #Set up generation dataset
    if generate_images:
        if type_nnet == 'conv':
            generation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_generation_data,
                                                        csv_file_label=csv_file_generation_label,
                                                        mask_file=generation_mask_file,
                                                        root_dir=data_source_path,
                                                        transform=transforms.ToTensor())

        elif type_nnet == 'simple':
            if dataset_type=='HealthMNIST':
                generation_dataset = HealthMNISTDataset(csv_file_data=csv_file_generation_data,
                                                    csv_file_label=csv_file_generation_label,
                                                    mask_file=generation_mask_file,
                                                    root_dir=data_source_path,
                                                    transform=transforms.ToTensor())
            elif dataset_type == 'Physionet':
                generation_dataset =PhysionetDataset(data_file=csv_file_generation_data, root_dir=data_source_path,data_format='2d')
        elif type_nnet=='rnn':
            if dataset_type == 'Physionet':
                generation_dataset = PhysionetDataset(data_file=csv_file_generation_data, root_dir=data_source_path,data_format='2d')
                gen_data=np.load(os.path.join(data_source_path,csv_file_generation_data),allow_pickle=True)
                P_gen_varyL=len(np.unique(gen_data['outcome_attrib'][:,id_covariate]))
                print(f'..there are {P_gen_varyL} total students in the generation set...')

    else:
        generation_dataset = None


    if varying_T:
        if P!=None:
            P_pred=int(P.split('-')[2])
            P_gen=int(P.split('-')[3])
        else:
        
            P_gen=P_gen_varyL
            P_pred=P_pred_varyL
    else:
        if P!=None:
            P_pred=int(P.split('-')[2])
            P_gen=int(P.split('-')[3])
        else:
            P_pred=P_pred_varyL
            P_gen=P_gen_varyL

    N = len(prediction_dataset)
    print('Length of prediction_dataset :  {}'.format(N))

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)

    # print(f' dataset dtype:{dataset.dtype}')
    Q = len(prediction_dataset[0]['label'])

    # len(dataset[0]['label']) #dataset is a list, can be used with idx, Q is the total number of variables
    # ----------end on setting up train/val/test/pred/gen dataset------------
    print(f'Q is the number of covariates:{Q}')

    # STEP3: =============set up model and send to GPU if available================
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed,
                                p_input=dropout_input, p=dropout).double().to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim, num_dim, vy_init, vy_fixed).to(device)
    elif type_nnet=='rnn':
        if type_rnn=='lstm':
            print(f'Using {type_nnet}-{type_rnn}')
            nnet_model=LSTMVAE( hidden_dim, num_layers, dropout, latent_dim, num_dim, vy_init, vy_fixed).to(device)

    print('...initialize GP and covar_module....')
    covar_module = []
    covar_module0 = []
    covar_module1 = []
    likelihoods = []
    gp_models = []
    Z = torch.zeros(N, latent_dim, dtype=torch.double).to(device)

    train_x = torch.zeros(N, Q, dtype=torch.double).to(device)

    marker_prev=marker.split('-')[0]
    marker_cur=marker.split('-')[1]
    if len(marker.split('-'))>2:
        model_flag=marker.split('-')[2]
    print(f'marker_prev:{marker_prev}\n'
    f'marker_cr: {marker_cur}')
    if hensman:
        likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
            noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device) # A Likelihood in GPyTorch specifies the mapping from latent function values f(X) to observed labels y. 
    # if input is a sample from f(x), then output will be the conditional distribution of p(y|f(x))
    # if the input is a distribution f(x), the output will be the marginal distribution of p(y|x)
    # in the case of regression this might be a Gaussian distribution, as y(x) is equal to f(x) plus Gaussian noise
        if constrain_scales:
            likelihoods.noise = 1
            likelihoods.raw_noise.requires_grad = False

        covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                                cat_kernel, bin_kernel, sqexp_kernel,
                                                                cat_int_kernel, bin_int_kernel,
                                                                covariate_missing_val, id_covariate)

        gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                                covar_module0 + covar_module1).to(device)

    try:
    # print(gp_model)
        print('!!!!remember to specify strict=False in the load_state_dict, otherwise, will have layer mismatch!!!')
        gp_model.load_state_dict(torch.load(os.path.join(gp_model_folder, f'{marker_prev}_gp_model.pth'), map_location=torch.device(device)),strict=False)
        zt_list = torch.load(os.path.join(gp_model_folder, f'{marker_prev}_zt_list.pth'), map_location=torch.device(device))
        print('...Loaded GP models and zt_list...')
        covar_module0.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_best_covar_module0.pth'), map_location=torch.device(device)),strict=False)
        #only when the model on cpu is allowable to eval
        print('...Loaded best_covar_module0 model...')
        covar_module1.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_best_covar_module1.pth'), map_location=torch.device(device)),strict=False)
        #only when the model on cpu is allowable to eval
        print('...Loaded best_covar_module1 model...')
        likelihoods.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_best_likelihoods.pth'), map_location=torch.device(device)),strict=False)
        #only when the model on cpu is allowable to eval
        print('...Loaded best_likelihoods module...')
    except:
        print('...GP model/zt_list/best_covar_module0/best_covar_module1/likelihoods loading failed!...')
        pass

    if type_KL == 'closed':
        covar_modules = [covar_module]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_modules = [covar_module0, covar_module1]
    # 

    print('===we don not want training process, go directly to infer====')

    nnet_model.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_{model_flag}-vae_model.pth'), map_location=torch.device(device)),strict=False)
    #only when the model on cpu is allowable to eval
    print('!!change model parameters to double, otherwise, input and model parameter gonna have dtype mismatch!!')
    nnet_model = nnet_model.double().to(device) 
    print('...Loaded nnet model...')

 
    # ===step 5: generate data======
    # data_batches=10
    

    if results_path and generation_dataset:
        print('...Generate using best model...')
        if varying_T:
            n_batches = (P_pred + subjects_per_batch - 1)//subjects_per_batch
            print(f'varying_T is true, you are using VaryingLengthSubjectSampler to form a  dataloader with n_batches= (P + subjects_per_batch - 1)//subjects_per_batch :{n_batches}')

            prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(
            VaryingLengthSubjectSampler(prediction_dataset, id_covariate), batch_size=subjects_per_batch,drop_last=True),
                                            num_workers=num_workers)
        else:
            
            P_pred=len(prediction_dataset)//T
            batch_size=subjects_per_batch*T
            actual_data_len=P_pred*T
            N_batches=actual_data_len//batch_size
            valid_len=N_batches*batch_size
            P_pred_actual=valid_len//T
            print(f'..loading {P_pred} prediction students and prediction data with actual_data_len/valid_len: {actual_data_len}/{valid_len} and batch size=subjects_per_batch*T {batch_size} and N_batches=actual_data_len//batch_size={N_batches}...')
            prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=BatchSampler(SubjectSampler(prediction_dataset, P_pred, T), batch_size, drop_last=True), num_workers=num_workers)
            # DataLoader(prediction_dataset, batch_sampler=BatchSampler(SubjectSampler(prediction_dataset, P, T), batch_size=subjects_per_batch*T, drop_last=True),
            #                                 num_workers=num_workers)

        print(f'prediction dataloader length:{N_batches}')
        full_mu = torch.zeros(actual_data_len, latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(actual_data_len, Q, dtype=torch.double).to(device)
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(tqdm(prediction_dataloader,total=N_batches)):
                label_id = sample_batched['idx']
                label=sample_batched['label']  
                if type_nnet=='rnn':
                    data = sample_batched['data']
                    data=data.reshape(subjects_per_batch,T,data.shape[-1])
                else:
                    data = sample_batched['digit']
                prediction_x[label_id] = label.double().to(device)
                data=data.double().to(device)

                mu, log_var = nnet_model.encode(data)
                if type_nnet=='rnn':
                    mu = mu.mean(0)
                    mu=torch.stack([mu for i in range(T)],dim=1).reshape(-1,latent_dim)
                
                full_mu[label_id] = mu
        if type_nnet=='rnn':
            
            print(f'valid samples:{valid_len}')
            prediction_x=prediction_x[:valid_len]
            full_mu=full_mu[:valid_len]


        out_path=f'{save_path}-{marker_cur}'
        print(f'out_path:{out_path}')
        

        if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
    
            
            X_df,recon_df,gen_w_stu=recon_complete_gen(subjects_per_batch,num_workers,dataset_type,generation_dataset, nnet_model, type_nnet, out_path, 
            covar_module0, covar_module1, likelihoods, latent_dim, ref_df_path, prediction_x, full_mu, zt_list, P_pred_actual,T, id_covariate, varying_T)

    if memory_dbg:
            print("Max memory allocated during cadence generation: {:.2f} MBs, we reset max mem alloc and empty cache".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.empty_cache()
