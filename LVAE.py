from multiprocessing import allow_connection_pickling
import os
from re import T
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
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path,data_format='2d')

    elif type_nnet=='rnn':
        if dataset_type == 'Physionet':
         
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path,data_format='2d')
            train_data=np.load(os.path.join(data_source_path,csv_file_data),allow_pickle=True)
            P_train_varyL=len(np.unique(train_data['outcome_attrib'][:,id_covariate]))
            print(f'..there are {P_train_varyL} total students in the train set...')

    #Set up prediction dataset
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


    if run_validation:
        if dataset_type == 'HealthMNIST':
            validation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_validation_data,
                                                        csv_file_label=csv_file_validation_label,
                                                        mask_file=validation_mask_file, root_dir=data_source_path,
                                                        transform=transforms.ToTensor())
            print('Length of validation dataset:  {}'.format(len(validation_dataset)))
        elif dataset_type == 'RotatedMNIST':
            validation_dataset = RotatedMNISTDatasetConv(data_file=csv_file_validation_data,
                                                            label_file=csv_file_validation_label,
                                                            mask_file=validation_mask_file, root_dir=data_source_path,
                                                            transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(validation_dataset)))
        elif dataset_type == 'Physionet':
            if type_nnet=='simple':
                validation_dataset = PhysionetDataset(data_file=csv_file_validation_data, root_dir=data_source_path,data_format='2d')
            elif type_nnet=='rnn':
                validation_dataset = PhysionetDataset(data_file=csv_file_validation_data, root_dir=data_source_path,data_format='2d')
                val_data=np.load(os.path.join(data_source_path,csv_file_validation_data),allow_pickle=True)
                P_val_varyL=len(np.unique(val_data['outcome_attrib'][:,id_covariate]))
                print(f'..there are {P_val_varyL} total students in the validation set...')
    else:
        validation_dataset = None
    if varying_T:
        if P!=None:
            P_train=int(P.split('-')[0])
            P_val=int(P.split('-')[1])
            P_pred=int(P.split('-')[2])
            # P_gen=int(P.split('-')[3])
        else:
            P_train=P_train_varyL
            P_val=P_val_varyL
            # P_gen=P_gen_varyL

    else:
        if P!=None:
            P_train=int(P.split('-')[0])
            P_val=int(P.split('-')[1])
            P_pred=int(P.split('-')[2])
            # P_gen=int(P.split('-')[3])
        else:
            P_train=P_train_varyL
            P_val=P_val_varyL
            P_pred=P_pred_varyL
            # P_gen=P_gen_varyL

    P=f'{P_train}-{P_val}'
    

    print('Length of training dataset:  {}'.format(len(dataset)))
    N = len(dataset)

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)


    Q = len(dataset[0]['label'])
    

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
    
    if varying_T:
        n_batches = (P_train + subjects_per_batch - 1)//subjects_per_batch
        print(f'varying_T is true, you are using VaryingLengthSubjectSampler to form a  dataloader with n_batches= (P + subjects_per_batch - 1)//subjects_per_batch :{n_batches}')
        setup_dataloader = DataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=num_workers)    

    else:
        P_train=len(dataset)//T
        batch_size=subjects_per_batch*T
        actual_data_len=P_train*T
        N_batches=actual_data_len//batch_size
        valid_len=N_batches*batch_size
        print(f'..loading training data with actual_data_len: {actual_data_len} and batch size=subjects_per_batch*T {batch_size} and N_batches=actual_data_len//batch_size={N_batches}...')
        setup_dataloader = DataLoader(dataset, batch_sampler=BatchSampler(SubjectSampler(dataset, P_train, T), batch_size, drop_last=True), num_workers=num_workers)
 
    
        # step4.1-----Get values for GP initialisation-------:

    # STEP3.1 ==========Load pre-trained encoder/decoder parameters if present=========
    if training:
        print('\n====loading pre-trained weights to get mu and logvar from pre-trained weights===')
        try:
            nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
            print('...Loaded pre-trained values...')
        except:
            print('....Loading pre-trained values failed, could be b/c the pre-training parameter are different from your current setup....')
    else:
        pass
    nnet_model = nnet_model.double().to(device)

    # STEP4: ===============set up Data Loader for GP initialisation, train GP models=================



    # # step4.1-----Get values for GP initialisation-------:
  
    Z = torch.zeros(actual_data_len, latent_dim, dtype=torch.double).to(device)

    train_x = torch.zeros(actual_data_len, Q, dtype=torch.double).to(device)

    nnet_model.eval()
    label_id_list=[]
    with torch.no_grad():
        print('...we need to generate the mu and logvar and general Z from training set...')
        for batch_idx, sample_batched in enumerate(tqdm(setup_dataloader,total=N_batches)):
 
            label_id = sample_batched['idx']
            train_x[label_id] = sample_batched['label'].double().to(device)
            if dataset_type=='Physionet':
                data = sample_batched['data']
            else:

                data = sample_batched['digit']
            if type_nnet=='rnn':
                data=data.reshape(subjects_per_batch,T,data.shape[-1])
            data=data.double().to(device)
            covariates = torch.cat((train_x[label_id, :id_covariate], train_x[label_id, id_covariate+1:]), dim=1)
            
            mu, log_var = nnet_model.encode(data)

            if type_nnet=='rnn':
         
                sampled_z=nnet_model.sample_latent(mu, log_var).mean(dim=0)
                Z[label_id] =torch.stack([sampled_z for i in range (T)],dim=1).reshape(-1,latent_dim) #(32,64)
                
            else:
                Z[label_id] = nnet_model.sample_latent(mu, log_var)

    if type_nnet=='rnn':

        train_x=train_x[:valid_len]
        Z=Z[:valid_len]
        
    print(f'Z/train_x shape after dropping last batch:{Z.shape}/{train_x.shape}')

    print('\n====SET UP GP MODEL====')
    covar_module = []
    covar_module0 = []
    covar_module1 = []
    zt_list = []
    likelihoods = []
    gp_models = []
    adam_param_list = []


    # step4.2-----training for GP prior, covar_module0 and covar_module1---------
    marker_prev=marker.split('-')[0]
    marker_cur=marker.split('-')[1]
    if len(marker.split('-'))>2:
        model_flag=marker.split('-')[2]
    print(f'marker_prev:{marker_prev}\n'
    f'marker_cr: {marker_cur}')
    # step4.2.1---hensman training style set up----
    if hensman:
        likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
            noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device) 
        if constrain_scales:
            likelihoods.noise = 1
            likelihoods.raw_noise.requires_grad = False

        covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                                cat_kernel, bin_kernel, sqexp_kernel,
                                                                cat_int_kernel, bin_int_kernel,
                                                                covariate_missing_val, id_covariate)

        gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                                covar_module0 + covar_module1).to(device)

        # initialise inducing points
        zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
        print(f'zt_list.shape:{zt_list.shape}')
        for i in range(latent_dim):

            
            train_x_len=train_x.shape[0]
            target_list=torch.cat((train_x[0:int(0.5*M)], train_x[int(train_x_len/2):int(train_x_len/2)+int(0.5*M)]), dim=0)
            
            zt_list[i]=target_list.clone().detach()


        adam_param_list.append({'params': covar_module0.parameters()})
        adam_param_list.append({'params': covar_module1.parameters()})




        covar_module0.train().double()
        covar_module1.train().double()
        likelihoods.train().double()

    # set up loading if pretrained gp model parameters, zt_list,m, H

        try:
            gp_model.load_state_dict(torch.load(os.path.join(gp_model_folder, f'{marker_prev}_gp_model.pth'), map_location=torch.device(device)))
            zt_list = torch.load(os.path.join(gp_model_folder, f'{marker_prev}_zt_list.pth'), map_location=torch.device(device))
            print('...Loaded GP models and zt_list...')
        except:
            print('...GP model loading failed!...')
            pass

        m = torch.randn(latent_dim, M, 1).double().to(device).detach()
        H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()

        if natural_gradient:
            print('...it is natural gradient, start loading gradient for m,H...')
            H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)
        
        try:
            m = torch.load(os.path.join(gp_model_folder,f'{marker_prev}_m.pth'), map_location=torch.device(device)).detach()#<----originally, they need to detach
            H = torch.load(os.path.join(gp_model_folder,f'{marker_prev}_H.pth'), 
            map_location=torch.device(device)).detach()#<----originally, they need to detach
            print('...Loaded natural gradient values m/H...')
        except:
            print('...Loading natural gradient values failed!...')
            pass

        if not natural_gradient:
            print ('it is not natural gradient, creating m, H')
            adam_param_list.append({'params': m})
            adam_param_list.append({'params': H})
            m.requires_grad_(True)
            H.requires_grad_(True)



            # step4.2.2---non-hensman training style----
    else:
        print('it is not hensman training, creating GP model different ways')
        for i in range(0, latent_dim):
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device))

            if constrain_scales:
                likelihoods[i].noise = 1
                likelihoods[i].raw_noise.requires_grad = False

            # set up additive GP prior
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                additive_kernel0, additive_kernel1 = generate_kernel_approx(cat_kernel, bin_kernel, sqexp_kernel,
                                                                            cat_int_kernel, bin_int_kernel,
                                                                            covariate_missing_val, id_covariate)
                covar_module0.append(additive_kernel0.to(device))           # additive kernel without id covariate
                covar_module1.append(additive_kernel1.to(device))           # additive kernel with id covariate
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                                covar_module0[i] + covar_module1[i]).to(device))

                z_init=torch.cat((train_x[20:20+int(2/3*M)], train_x[10000:10000+int(2/3*M)]), dim=0)
       
                zt = torch.nn.Parameter(z_init.clone().cpu().double().detach(), requires_grad=False)
                zt_list.append(zt)
                adam_param_list.append({'params': covar_module0[i].parameters()})
                adam_param_list.append({'params': covar_module1[i].parameters()})
   
            else:
                additive_kernel = generate_kernel(cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel,
                                                    covariate_missing_val)
                covar_module.append(additive_kernel.to(device))             # additive kernel GP prior
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                                covar_module[i]).to(device))
                adam_param_list.append({'params': gp_models[i].parameters()})

            gp_models[i].train().double()
            likelihoods[i].train().double()

        

            try:
                print('try to load gp_model from local')
                for i in range(0, latent_dim):
                    gp_model_name = f'{marker_prev}_gp_model' + str(i) + '.pth'
                    zt_list_name = f'{marker_prev}_zt_list' + str(i) + '.pth'
                    gp_models[i].load_state_dict(torch.load(os.path.join(gp_model_folder,gp_model_name), map_location=torch.device(device)))
                    zt_list[i] = torch.load(os.path.join(gp_model_folder,zt_list_name), map_location=torch.device('cpu'))
            except:
                print('fail to load gp_model from local')
                pass

    # step4.2.3---start actual training after GP model----

    nnet_model.train()
    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3) 



    if type_KL == 'closed':
        covar_modules = [covar_module]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_modules = [covar_module0, covar_module1]

    start = timer()
    out_path=f'{save_path}-{marker_cur}'


    if type_nnet=='rnn':
        if dataset_type == 'Physionet':
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path,data_format='2d')
    print(f'm.shape:{m.shape}')
    if training:
        print('\n====start training actual vae training after constructing the gp model===\n')
        if hensman:
            print('...using hensman setup for vae...')
            # print(f'm/H shape before training:{m.shape}/{H.shape}')
            generation_dataset=None
            prediction_dataset=None
            _ = hensman_training(num_workers,dataset_type,nnet_model, type_nnet, epochs, dataset,
                                    optimiser, type_KL, num_samples, latent_dim,
                                    covar_module0, covar_module1, likelihoods, m,
                                    H, zt_list, P_train, T, varying_T, Q, weight,
                                    id_covariate, loss_function, natural_gradient, natural_gradient_lr,
                                    subjects_per_batch, memory_dbg, eps,
                                    out_path, validation_dataset,
                                    generation_dataset, prediction_dataset, gp_model, csv_file_test_data=csv_file_test_data,
                                    csv_file_test_label=csv_file_test_label, test_mask_file=test_mask_file,
                                    data_source_path=data_source_path)

            m, H = _[5], _[6]
            best_model,best_covar_module0, best_covar_module1,best_likelihoods=_[8],_[9],_[10],_[11]
     
        elif mini_batch:
            print('using mini_batch training for vae')
            _ = minibatch_training(dataset_type,nnet_model, type_nnet, epochs, dataset,
                                    optimiser, type_KL, num_samples, latent_dim,
                                    covar_module0, covar_module1, likelihoods,
                                    zt_list, P, T, Q, weight,
                                    id_covariate, loss_function, memory_dbg, eps, out_path,
                                    validation_dataset, generation_dataset, prediction_dataset)
            best_model,best_covar_module0, best_covar_module1=_[-3],_[-2],_[-1]
        elif variational_inference_training:
            print('using variational_inference_training  for vae')
            variational_inference_optimization(dataset_type,nnet_model, type_nnet, epochs, dataset, prediction_dataset,
                                                optimiser, latent_dim, covar_module0, covar_module1,
                                                likelihoods, zt_list, P, T, Q, weight, constrain_scales,
                                                id_covariate, loss_function, memory_dbg, eps,
                                                out_path, save_path, gp_model_folder, generation_dataset)
        else:
            print('using standard_training  for vae')
            _ = standard_training(dataset_type,nnet_model, type_nnet, epochs, dataset,
                                    optimiser, type_KL, num_samples, latent_dim,
                                    covar_modules, likelihoods, zt_list,
                                    id_covariate, P, T, Q, weight, constrain_scales,
                                    loss_function, memory_dbg, eps, validation_dataset,
                                    generation_dataset, prediction_dataset)
        end=timer()
        min,sec=divmod((end-start),60)
        print(f"\nDuration of training: {min} min and {sec:.2f} seconds\n")

        if memory_dbg:
            print("Max memory allocated during training: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.empty_cache()

        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]
        print('Best results in epoch: ' + str(_[7]))


        # # step4.2.4---saving-----

        print(f'...Saving diagnostics.pkl, plot_values, final-vae model, gp_model, zt_list, m, H into {save_path}...')

        pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                        os.path.join(save_path, f'{marker_cur}_diagnostics.pkl'))

        pd.to_pickle([train_x, mu, log_var, Z, label_id], os.path.join(save_path, f'{marker_cur}_plot_values.pkl'))
    
        
        if hensman:

            pass

        else:
            for i in range(0, latent_dim):
                gp_model_name = f'{marker_cur}_gp_model' + str(i) + '.pth'
                zt_list_name = f'{marker_cur}_zt_list' + str(i) + '.pth'
                torch.save(gp_models[i].state_dict(), os.path.join(save_path, gp_model_name))
                try:
                    torch.save(zt_list[i], os.path.join(save_path, zt_list_name))
                except:
                    pass

        if memory_dbg:
            print("Max memory allocated during saving and post-processing: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.empty_cache()   


    else:

        print('===we don not want training process, go directly to infer====')
        _=np.load(os.path.join(gp_model_folder,f'{marker_prev}_diagnostics.pkl'),allow_pickle=True)
        penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]
        print('...Loaded diagnostics...')
        nnet_model.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_{model_flag}-vae_model.pth'), map_location=torch.device(device)))

        print('...Loaded nnet model...')
        covar_module0.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_best_covar_module0.pth'), map_location=torch.device(device)))

        print('...Loaded best_covar_module0 model...')
        covar_module1.load_state_dict(torch.load(os.path.join(gp_model_folder,f'{marker_prev}_best_covar_module1.pth'), map_location=torch.device(device)))

        print('...Loaded best_covar_module1 model...')
        best_model=nnet_model
        best_covar_module0=covar_module0
        best_covar_module1=covar_module1

