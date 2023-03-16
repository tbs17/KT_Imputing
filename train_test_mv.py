from constant import *
from data_process import *
import os, re, copy, time, random,torch,json,datetime,time
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.utils import data
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from mv_nn_transfer import *
from mv_nn import *
from ast import literal_eval
from generalize_kt import *
from train_test_sv import *
from data_process import *
from kt_utils import *
# -----get combinations for features to train-----

def trans_seed_model_runner(params):
    variate_type=params['variate_type']
    lag_type=params['lag_type']
    model_name=params['model_name']

    batch_size=params['batch_size']
    model_transform=params['model_transform']
    best_src_shift=params['best_src_shift']
    best_tar_shift=params['best_tar_shift']
    start_shift_step=best_tar_shift
    total_time_steps=best_tar_shift+1
    start=params['start']
    end=params['end']
    lamb=params['lamb']
    domain_src=params['domain_src']
    domain_tar=params['domain_tar']  
    pred_type=params['pred_type']#<_--
    target_var_pos=params['target_var_pos']#<--
    tag_pos=params['tag_pos']#<--
    na_fill_type=params['na_fill_type']#<_---
    df=pd.DataFrame()
    for i in range(1,6):
        seed=i
        set_seed(seed)
        print(f'\n...start running model for seed {seed}...\n')
        params={
            'domain_src':domain_src,
            'domain_tar':domain_tar,
            'src_name':domain_src,
            'tar_name':domain_tar,
            'start':start,
            'end':end,
            'start_shift':start_shift_step,
            'start_shift_step':start_shift_step,
            'total_time_steps':total_time_steps,
            'model_name':model_name,
            'model_transform':model_transform,
            'save_flag':False,
            'save_data_flag':False,
            'save_model_flag':False,
            'batch_size':batch_size,
            'best_src_shift':best_src_shift,
            'best_tar_shift':best_tar_shift,
            'target_var_pos':target_var_pos,
            'tag_pos':tag_pos,
            'pred_type':pred_type,
            'seed':seed,
            'lamb':lamb,
            'na_fill_type':na_fill_type
            }
        if variate_type=='sv':

            results=sv_transfer_runner(params)
            
        else:
    
            results=mv_transfer_runner(params)
        
        results['data']=domain_src
        results['variate_type']=variate_type
        results['seed']=seed
        results['run_type']='model_tr'
        results['lag_type']=lag_type
        df=df.append(results)
        display(df.head())
        out_dir='TRANS_MODEL/MODEL_TR'
        os.makedirs(out_dir,exist_ok=True)
        df.to_csv(f'{out_dir}/{variate_type}_{domain_src}_{model_name}_{start_shift_step}_{total_time_steps}_{na_fill_type}_seed_model_tr.csv',index=False)
    print(f'obtain {df.shape[0]} models')
    display(df.head())
    return df


def get_combo(test_features,marker,combo_range):
    from itertools import combinations
    from ast import literal_eval
    start,end=combo_range
    cnt=0
    adict={'len':[],'combo':[]}
    for L in range(start,end):
        # if L<3:
        for subset in combinations(test_features,L):
            cnt+=1
            print(L,subset)            
            adict['len'].append(L)
            adict['combo'].append(subset)


    print(cnt) 
    dat=pd.DataFrame.from_dict(adict)
    display(dat)
    dat.to_csv(f'TRANS_DATA/combination_{marker}.csv',index=False)
    return dat


# -------------below is the easy version which is the kt orginal format version input trainer-----------


from network.util_network import ScheduledOptim, NoamOpt
class NoamOptimizer: #this is the adapted optimizer from adam from all you need is attention paper

    def __init__(self, model, lr, model_size, warmup,weight_decay):
        self._adam = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #weight decay is like L2 norm
        self._opt = NoamOpt(
            model_size=model_size, factor=1, warmup=warmup, optimizer=self._adam)

    def step(self, loss):
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()


def pprint(log_file,*text):
    # print with UTC+8 time
    import datetime
    time = '['+str(datetime.datetime.utcnow() +
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *text, flush=True)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(time, *text, flush=True, file=f)


# =============below is mv version model retriever/runner/seed_runner============

def seed_model_runner_mv(params):
    marker=params['marker']
    start_shift=params['start_shift']
    total_time_step=params['total_time_step']
    model_name=params['model_name']
    data_name=params['data_name']
    batch_size=params['batch_size']
    root_dir=params['root_dir']
    save_data_flag=params['save_data_flag']
    save_model_flag=params['save_model_flag']
    model_transform=params['model_transform']
    na_fill_type=params['na_fill_type']
    seed_max=params['seed_max']
    need_convert=params['need_convert']
    data_path=params['data_path']
    seq_size=params['seq_size']
    batch_size=params['batch_size']
    test_features=params['test_features']
    model_path=params['model_path']
    training=params['training']
    test_loader=params['test_loader']
    seed_run=params['seed_run']
    augment_data_path=params['augment_data_path']
    print(f'we will use {na_fill_type} na fill method and {marker} data')
    df=pd.DataFrame()
    if seed_run:
        for i in range(1,seed_max+1):
            seed=i*10
            set_seed(seed)
            print(f'\n...start running model for seed {seed}...\n')

            params={
            
            'model_name':model_name,
            'data_name':data_name,
            'start_shift_step':start_shift,
            'total_time_steps':total_time_step,
            'save_data_flag':save_data_flag,
            'save_model_flag':save_model_flag,
            'model_transform':model_transform,
            'seed':seed,
            'num_stu':None,
            'na_fill_type':na_fill_type,
            'need_convert': need_convert,
            'converted_data_path':'',
            'root_dir':root_dir,
            'data_path':data_path,
            'save_dir':'',
            'test_features':test_features,
            'seq_size':seq_size,
            'batch_size':batch_size,
            'marker':marker,
            'model_path':model_path,
            'training':training,
            'test_loader':test_loader,
            'augment_data_path':augment_data_path

            #    'TRANS_DATA/alg2_40stu_seqL150_8features_None_mask_transformerTrue.npz'
            } 
            results=mv_best_model_retriever_new(params)
            results['data']=data_name
            results['model_index']=i
            df=df.append(results)
    else:
            params={
            
            'model_name':model_name,
            'data_name':data_name,
            'start_shift_step':start_shift,
            'total_time_steps':total_time_step,
            'save_data_flag':save_data_flag,
            'save_model_flag':save_model_flag,
            'model_transform':model_transform,
            'seed':111,
            'num_stu':None,
            'na_fill_type':na_fill_type,
            'need_convert': need_convert,
            'converted_data_path':'',
            'root_dir':root_dir,
            'data_path':data_path,
            'save_dir':'',
            'test_features':test_features,
            'seq_size':seq_size,
            'batch_size':batch_size,
            'marker':marker,
            'model_path':model_path,
            'training':training,
            'test_loader':test_loader,
            'augment_data_path':augment_data_path

            #    'TRANS_DATA/alg2_40stu_seqL150_8features_None_mask_transformerTrue.npz'
            } 
            results=mv_best_model_retriever_new(params)
            results['data']=data_name
            results['model_index']=1 
            
           
        # results['lag_type']=lag_type
            df=df.append(results)
    display(df.head())
    out_dir='TRANS_MODEL/MODEL_RUN'
    os.makedirs(out_dir,exist_ok=True)
    print(f'...saving results under {out_dir}/{data_name}_{model_name}_{start_shift}_{total_time_step}_{na_fill_type}_seed_model_run_{marker}.csv ')
    df.to_csv(f'{out_dir}/{data_name}_{model_name}_{start_shift}_{total_time_step}_{na_fill_type}_seed_run{seed_run}_{marker}.csv',index=False)
    
    display(df.head())

    avg_mse=df['MSE'].mean()
    avg_rmse=df['RMSE'].mean()
    avg_l1=df['L1'].mean()
    print(f'...after running {seed_max} seeds, avg performance mse/rmse/l1is:{avg_mse:.5f}/{avg_rmse:.5f}/{avg_l1:.5f}!..')
    return df



def mv_best_model_retriever_new(params):
    seed=params['seed']
    start0=time.time()
    marker=params['marker']
    model_name=params['model_name']
    data_name=params['data_name']
    start_shift_step=params['start_shift_step']
    total_time_steps=params['total_time_steps']
    gpu_core=0
    save_data_flag=params['save_data_flag']
    save_model_flag=params['save_model_flag']
    need_convert=params['need_convert']
    num_stu=params['num_stu']
    data_path=params['data_path']
    root_dir=params['root_dir']
    print(f'data_path: {data_path}')
    model_transform_flag=params['model_transform']
    na_fill_type=params['na_fill_type']
    converted_data_path=params['converted_data_path']
    save_dir=params['save_dir']
    seq_size=params['seq_size']
    batch_size=params['batch_size']
    test_features=params['test_features']
    print(f'we will use {na_fill_type} na fill method')
 
    seed=int(params['seed'])
    model_path=params['model_path']
    training=params['training']
    test_loader=params['test_loader']
    augment_data_path=params['augment_data_path']
    if model_transform_flag==0:
        flat_flag=True
        model_transform=False
    else:
        flat_flag=False
        model_transform=True
    df=pd.read_csv(data_path)

    log_file_name='run_best'
    out_file_name=f'{data_name}_{model_name}_{na_fill_type}'
    out_dir=f'{data_name}_{model_name}_time_feat_combo'

    print(f'....Step wise feature adding for the {model_name}......')
    data_params={
        'data':df,
        'batch_size' : batch_size,
        'seq_size':seq_size,
        'total_time_step':total_time_steps, 
        'remove_feature_list':'', 
        'domain_src':f'{data_name}',
        'num_stu':num_stu,
        'marker':None,
        'model_transform':model_transform,
        'save_data_flag':save_data_flag,
        'save_model_flag':save_model_flag,
        'NUM_WORKER' : 0,
        'start_shift_step':start_shift_step,
        'na_fill_type':na_fill_type,
        'need_convert':need_convert,
        'converted_data_path':converted_data_path,
        'save_dir':save_dir,
        'test_features':test_features,
        'marker':marker,
        'test_loader':test_loader,
        'augment_data_path':augment_data_path
    }



    # :::train para:::
    train_params={

    'learning_rate' : 2e-4,
    'n_epochs' : 5,
    'weight_decay':5e-4,
    'warm_up_step_count':20,
    'label_exist':True,
    'flat_flag':flat_flag,
    'log_file_name':log_file_name,
    'out_file_name':out_file_name,
        'out_dir':out_dir,
        'gpu_core':gpu_core,
        'seed':seed,
        'training':training
        # 'is_continue':is_continue

    }


    # Model parameters
    model_params={
        'hidden_dim' : 128 , # Lattent dim
        'q' : 8 , # Query size
        'v' : 8 , # Value size
        'h' : 8 , # Number of heads
        'N' : 4 , # Number of encoder and decoder to stack
        'attention_size' : 4,  # Attention window size
        'dropout' : 0.2,  # Dropout rate
        'pe' : None,  # Positional encoding
        'chunk_mode' : None,
        'input_dim' :len(test_features),  # From dataset
        'output_dim' :1,  # From dataset
        'num_layers':2,
        'src_seqL':seq_size,
        'tar_seqL':seq_size,
        'n_hiddens':[64,64],
        'class_num':1,
        'model_name':model_name,
        'transfer_net':False,
        'model_path':model_path
    }

    time_feat_combo_run=time_feature_combo_run_new(data_params,train_params,model_params)

    end0=time.time()

    print(f'===MP takes {round((end0-start0)/60,2)} mins to finish!')
    # now = datetime.now()
    # print("now =", now)
    return time_feat_combo_run


def time_feature_combo_run_new(data_params,train_params,model_params):
    from itertools import combinations
    from ast import literal_eval
    data=data_params['data']
    augment_data_path=data_params['augment_data_path']

    seq_size=data_params['seq_size']
    total_time_step=data_params['total_time_step']
    shift_step_list=range(1,total_time_step)
    start_shift_step=int(data_params['start_shift_step'])
    remove_feature_list=data_params['remove_feature_list']
    test_feature_list=data_params['test_features']
    domain_src=data_params['domain_src']
    stu_num=data_params['num_stu']
    test_loader=data_params['test_loader']
    model_transform=data_params['model_transform']
    save_data_flag=data_params['save_data_flag']
    save_model_flag=data_params['save_model_flag']
    save_dir=data_params['save_dir']
    marker=data_params['marker']
    NUM_WORKER = data_params['NUM_WORKER']
    batch_size = data_params['batch_size']
    # dat=data_params['combo_file']
    na_fill_type=data_params['na_fill_type']
    need_convert=data_params['need_convert']
    converted_data_path=data_params['converted_data_path']
    gpu_core=int(train_params['gpu_core'])
    seed=int(train_params['seed'])
    training=train_params['training']
    cnt=0
    df=pd.DataFrame()
    device = torch.device(f"cuda:{gpu_core}")
    ratios=[0.7,0.1,0.2]

    

    if start_shift_step>1:
        shift_step_list=range(start_shift_step,total_time_step)
    else:
        shift_step_list=shift_step_list
    for j in shift_step_list:

        print(f'....creating data at shift step {j} for feature list: \n{test_feature_list} ')
        if training:
            print(f'..generating data loader in the new way...')
            train_loader,val_loader,test_loader=generate_loader(ratios,batch_size,augment_data_path,save_dir,data,na_fill_type,seq_size,j, test_feature_list,domain_src,stu_num,marker,model_transform,save_data_flag)
            dataloaders={
                'train':train_loader,
                'val':val_loader,
                'test':test_loader
            }
        else:
            test_loader=test_loader
        
        set_seed(seed)
        out_dir=train_params['out_dir']
        # is_continue=train_params['is_continue']      
        log_file_name=train_params['log_file_name']+f'{start_shift_step}-{total_time_step}.log'
        out_file_name=train_params['out_file_name']+f'{start_shift_step}-{total_time_step}.csv'
        learning_rate = train_params['learning_rate']
        n_epochs = train_params['n_epochs']
        weight_decay=train_params['weight_decay']
        warm_up_step_count=train_params['warm_up_step_count']
        label_exist=train_params['label_exist']
        flat_input=train_params['flat_flag']
        


        #:::: Model parameters::::
        input_dim = len(test_feature_list) # From dataset
        feature_len=len(test_feature_list)
        output_dim = model_params['output_dim']
        num_layers=model_params['num_layers']
        src_seqL,tar_seqL=model_params['src_seqL'],model_params['tar_seqL']
        hidden_dim = model_params['hidden_dim']  # Lattent dim
        q = model_params['q']  # Query size
        v = model_params['v']  # Value size
        h = model_params['h']  # Number of heads
        N = model_params['N']  # Number of encoder and decoder to stack
        attention_size = model_params['attention_size']  # Attention window size
        dropout = model_params['dropout']  # Dropout rate
        pe = model_params['pe']  # Positional encoding
        chunk_mode = model_params['chunk_mode']   
        n_hiddens=model_params['n_hiddens']
        class_num=model_params['class_num']
        model_name=model_params['model_name']
        model_path=model_params['model_path']
#             if j%30==0:
        print(f'================predict for seqL={src_seqL}--{test_feature_list}--shiftStep{j}=================')
        
        if model_name in ['lstm','LSTM','Lstm']:
            model=LSTM(na_fill_type,input_dim, hidden_dim, num_layers, output_dim,device).to(device)
            print(f'this is lstm model,you are training with {model_name} model')
        elif model_name in ['adaRNN','adarnn','AdaRNN']:
            print(f'this is adaRNN model,you are training with {model_name} model')
            model=AdaRNN_noTL(na_fill_type,use_bottleneck=True, bottleneck_width=64, n_input=input_dim, n_hiddens=n_hiddens,  n_output=class_num, dropout=dropout, len_seq=tar_seqL, model_type=model_name).to(device)
        else:
            print(f'this is transformer model,you are training with {model_name} model')
            model = Transformer(na_fill_type,input_dim, hidden_dim, output_dim, q, v, h, N, attention_size=attention_size,
            dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)

        optimizer = NoamOptimizer(model=model, lr=learning_rate, model_size=hidden_dim, warmup=warm_up_step_count,weight_decay=weight_decay)
        
        checkpoint_format=f'TRANS_MODEL/BKT_MV/{out_dir}/stuNUm_{stu_num}_{marker}_feature{feature_len}_shiftStep{j}_{model_name}_{domain_src}_in{input_dim}_hid{hidden_dim}_{num_layers}numLayers_lr{learning_rate}_bs{batch_size}_seqL{src_seqL}_dw{weight_decay}_EP{n_epochs}_{na_fill_type}'

        direct_folder=checkpoint_format.split('/stuNUm')[0]
  
        os.makedirs(direct_folder,exist_ok=True)
        
        if save_model_flag:
            
            log_file=os.path.join(checkpoint_format,log_file_name)
            os.makedirs(checkpoint_format,exist_ok=True)
 
        
        print(f'>>>>train model for {checkpoint_format}<<<<<<')
        if training:
            best_epo_path,best_loss,best_loss_r,best_loss_l1=train_test_reg_mv(log_file, dataloaders, model, optimizer,checkpoint_format,data_params,train_params,model_params)

        else:
            model.load_state_dict(torch.load(model_path),strict=False)
            best_loss,best_loss_r,best_loss_l1=test_reg_mv(model,test_loader,data_params,train_params,model_params)
            best_epo_path=None
        new_df=pd.DataFrame({'model_name':[model_name],'features':[test_feature_list],'shift':[j],'na_fill_type':[na_fill_type],
                            'ckpt':[best_epo_path],'MSE':[best_loss],'RMSE':[best_loss_r],'L1':[best_loss_l1],'seed':[seed],'marker':[marker]})
        df=df.append(new_df)

        df.to_csv(os.path.join(direct_folder,out_file_name),index=False)
        display(df.tail())

    print(f'=======we have run total {cnt} feature combos!=====')
    display(df.tail(10))
    return df

def test_reg_mv(model,test_loader,data_params,train_params,model_params):
    gpu_core=int(train_params['gpu_core'])
    label_exist=train_params['label_exist']
    transfer_net=model_params['transfer_net']
    flat_input=train_params['flat_flag']#<--

    na_fill_type=data_params['na_fill_type']
    model.eval()
    criterion, criterion_l1=nn.MSELoss(),nn.L1Loss()
 
    running_loss,running_r_loss,running_l1_loss=[],[],[]
    device=torch.device(f'cuda:{gpu_core}')
    with torch.no_grad():
        for data in tqdm(test_loader,total=len(test_loader)):
          
            if label_exist:
                if na_fill_type=='mask':
                    feature, _,label_reg,feature_lens=data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    # , feature_lens.to(device).long()
                    input=(feature,feature_lens)
                else:
                    feature, _,label_reg = data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    input=feature
            else:
                feature,label_reg=data
            if flat_input:
                label_reg=label_reg.reshape(-1,1)
            
            
            if transfer_net:
                reg_pred = model.predict(input)
            else:
                reg_pred = model(input)
            loss=criterion(reg_pred.to(device),label_reg)
            loss_r=torch.sqrt(loss)
            loss_l1=criterion_l1(reg_pred.to(device),label_reg)
            running_loss.append(loss.detach().item())
            running_r_loss.append(loss_r.detach().item())
            running_l1_loss.append(loss_l1.detach().item())
        
        epoch_loss=np.mean(running_loss)
        epoch_loss_r=np.mean(running_r_loss)
        epoch_loss_l1=np.mean(running_l1_loss)
 
    return epoch_loss, epoch_loss_r, epoch_loss_l1


def train_test_reg_mv(log_file,data_loader, model, optimizer,checkpoint_format,data_params,train_params,model_params):
    label_exist=train_params['label_exist']
    n_epochs=train_params['n_epochs']
    flat_input=train_params['flat_flag']#<--
    save_model_flag=data_params['save_model_flag']
    na_fill_type=data_params['na_fill_type']
    gpu_core=int(train_params['gpu_core'])
    seed=int(train_params['seed'])
    device=torch.device(f'cuda:{gpu_core}')
    train_loader,val_loader,test_loader=data_loader['train'],data_loader['val'],data_loader['test']
    stop = 0
    best_mean_loss,best_loss, best_loss_r, best_loss_l1=10,10,10,10
    best_epo_list=[]
    loss_train_list,r_loss_train_list,l1_loss_train_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_val_list,r_loss_val_list,l1_loss_val_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_test_list,r_loss_test_list,l1_loss_test_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    criterion, criterion_l1=nn.MSELoss(),nn.L1Loss()
    model_name=checkpoint_format.split('/')[-1]
    for e in range(1,n_epochs+1):
        stop += 1
        running_train_loss,running_train_loss_r,running_train_loss_l1,running_val_loss=[],[],[],[]
        model.train()
        for data in tqdm(train_loader, total=len(train_loader)):

            if label_exist:
                if na_fill_type=='mask':
                    feature, _,label_reg,feature_lens=data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    input=(feature,feature_lens)
                else:
                    feature, _,label_reg = data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    input=feature
            else:
                feature,label_reg=data
            if flat_input:
                label_reg=label_reg.reshape(-1,1)
                
            
#             print(device)
            reg_pred = model(input)
            train_loss = criterion(reg_pred.to(device), label_reg) 
            train_loss_r=torch.sqrt(train_loss)
            train_loss_l1=criterion_l1(reg_pred.to(device), label_reg)
            optimizer.step(train_loss)
          
            running_train_loss.append(train_loss.item())
            running_train_loss_r.append(train_loss_r.item())
            running_train_loss_l1.append(train_loss_l1.item())
        epoch_train_loss=np.mean(running_train_loss)
        epoch_train_loss_r=np.mean(running_train_loss_r)
        epoch_train_loss_l1=np.mean(running_train_loss_l1)
        

        loss_train_list[e-1]=epoch_train_loss
        r_loss_train_list[e-1]=epoch_train_loss_r
        l1_loss_train_list[e-1]=epoch_train_loss_l1
        epoch_val_loss, epoch_val_loss_r, epoch_val_loss_l1=test_reg_mv(model,val_loader,data_params,train_params,model_params)
        loss_val_list[e-1]=epoch_val_loss
        r_loss_val_list[e-1]=epoch_val_loss_r
        l1_loss_val_list[e-1]=epoch_val_loss_l1
        epoch_test_loss, epoch_test_loss_r, epoch_test_loss_l1=test_reg_mv(model,test_loader,data_params,train_params,model_params)
        loss_test_list[e-1]=epoch_test_loss
        r_loss_test_list[e-1]=epoch_test_loss_r
        l1_loss_test_list[e-1]=epoch_test_loss_l1
        avg_l1_r_loss=(epoch_test_loss_r+epoch_test_loss_l1)/2
    
        if  avg_l1_r_loss< best_mean_loss:

            prev_best=copy.deepcopy(best_loss_r)
            best_mean_loss=avg_l1_r_loss
            best_loss=epoch_test_loss
            best_loss_r=epoch_test_loss_r
            best_loss_l1=epoch_test_loss_l1
            best_epo_list.append(e)
            # pprint(log_file,best_epo_list)
            
            filepath = f'seed{seed}_epoch{e}_test_rmse{best_loss_r:.4f}.pkl'
            if save_model_flag:torch.save(model.state_dict(), os.path.join(checkpoint_format,filepath))
            
            if len(best_epo_list)>=2 and best_epo_list[-1]>best_epo_list[-2] and save_model_flag:
                os.remove(os.path.join(checkpoint_format,f'seed{seed}_epoch{best_epo_list[-2]}_test_rmse{prev_best:.4f}.pkl'))

        if stop >= 20:
            break
    
    if len(best_epo_list)>0:
        best_epo_path=model_name+f'seed{seed}_epoch{best_epo_list[-1]}_test_rmse{best_loss_r:.4f}.pkl'
        best_path=os.path.join(checkpoint_format,best_epo_path)
        print(
        f'>>>best epoch is {best_epo_list[-1]}, at folder: {checkpoint_format}\n'
        f'path:{best_epo_path}\n'
        f'with best Test MAE/RMSE/L1 loss: {best_loss:.4f}/{best_loss_r:.4f}/{best_loss_l1:.4f}')
    
        fig,ax=plt.subplots() #make it explicit, usually show is using plt, save is to save the figure, the rest is to plot on the ax
        ax.plot(r_loss_train_list, 'o-', label='train_rmse')
        ax.plot(r_loss_val_list, 'o-', label='val_rmse')
        ax.plot(r_loss_test_list, 'o-', label='test_rmse')
        ax.plot(l1_loss_train_list, '--', label='train_L1')
        ax.plot(l1_loss_val_list, '--', label='val_L1')
        ax.plot(l1_loss_test_list, '--', label='test_L1')
        # ax.title(f'{model_name} model shiftStep {})
        ax.legend()
        if save_model_flag:

            fig.savefig(os.path.join(checkpoint_format,f'seed{seed}_{model_name}.jpg'))

            plt.show()
    else:
        best_path,best_loss,best_loss_r,best_loss_l1=None,None,None,None,
    return best_path,best_loss,best_loss_r,best_loss_l1




# ------------------below is transfer mv trainer--------------------
def mv_transfer_runner(params):
    start0=time.time()
    start=params['start']
    end=params['end']
    model_name=params['model_name']
    src_name=params['src_name']#<---
    tar_name=params['tar_name']#<--
    start_shift_step=params['start_shift_step']
    total_time_steps=params['total_time_steps']
    batch_size=params['batch_size']
    seed=int(params['seed'])
    gpu_core=0
    save_data_flag=params['save_data_flag']
    save_model_flag=params['save_model_flag']
    best_src_shift=params['best_src_shift']
    na_fill_type=params['na_fill_type']
    print(f'we will use {na_fill_type} na fill method')
    src_path=f'TRANS_DATA/KT_data_2017-07-01_2019-06-30_{src_name}_eng.csv'
    tar_path=f'TRANS_DATA/KT_data_2017-07-01_2019-06-30_{tar_name}_eng.csv'
    combo_file='TRANS_DATA/combination_no_prof_complete_2_feat.csv'
    # 'TRANS_DATA/combination_no_prof_complete.csv'
    feature_file='TRANS_DATA/test_features.json'
    model_transform_flag=params['model_transform']
    if model_transform_flag==0:
        flat_flag=True
        model_transform=False
    else:
        flat_flag=False
        model_transform=True
    src_df=pd.read_csv(src_path)
    tar_df=pd.read_csv(tar_path)
    dat=pd.read_csv(combo_file)
    log_file_name='run_TR'
    out_file_name=f'{src_name}_to_{tar_name}_{model_name}_{start}_{best_src_shift}_{total_time_steps-1}_TR_{na_fill_type}'
    out_dir=f'{src_name}_to_{tar_name}_{model_name}_TR'

    dat['combo']=dat['combo'].apply(lambda x:literal_eval(x))
    dat['seq']=dat.index
    with open(feature_file,'r') as f:
        features=json.load(f)
    test_features=features['test_features']
    print(f'....Step wise feature adding for the {model_name}......')
    data_params={
        'src_data':src_df,
        'tar_data':tar_df,
        'batch_size' : batch_size,
        'seq_size':150,
        'total_time_step':total_time_steps, 
        'remove_feature_list':'', 
        'test_features':test_features,
        'domain_src':f'{src_name}',
        'domain_tar':f'{tar_name}',
        'num_stu':None,
        'marker':None,
        'model_transform':model_transform,
        'save_data_flag':save_data_flag,
        'save_model_flag':save_model_flag,
        'NUM_WORKER' : 2,
        'combo_file':dat,
        'start':start,
        'end':end,
        'start_shift_step':start_shift_step,
        'na_fill_type':na_fill_type
    }



    # :::train para:::
    train_params={

    'learning_rate' : 2e-4,
    'n_epochs' : 5,
    'weight_decay':5e-4,
    'warm_up_step_count':20,
    'label_exist':True,
    'flat_flag':flat_flag,
    'log_file_name':log_file_name,
    'out_file_name':out_file_name,
    'out_dir':out_dir,
    'seed':seed,
    'lamb':10,
    'len_win':2,#only valid for adaRNN transfer net
    'pre_epoch':5, #only valid for adaRNN transfer net
    'gpu_core':gpu_core,
    "best_src_shift":best_src_shift
    }


    # Model parameters
    model_params={
        'hidden_dim' : 128 , # Lattent dim
        'q' : 8 , # Query size
        'v' : 8 , # Value size
        'h' : 8 , # Number of heads
        'N' : 4 , # Number of encoder and decoder to stack
        'attention_size' : 4,  # Attention window size
        'dropout' : 0.2,  # Dropout rate
        'pe' : None,  # Positional encoding
        'chunk_mode' : None,
        'input_dim' :len(test_features),  # From dataset
        'output_dim' :1,  # From dataset
        'num_layers':2,
        'src_seqL':150,
        'tar_seqL':150,
        'n_hiddens':[64,64],
        'class_num':1,
        'model_name':model_name,
        'transfer_loss_type':'mmd',
        'transfer_net':True
    }

    transfer_res=mv_transfer_trainer(data_params,train_params,model_params)

    end0=time.time()

    print(f'===MP takes {round((end0-start0)/60,2)} mins to finish!')
    # now = datetime.now()
    # print("now =", now)
    return transfer_res

def mv_transfer_trainer(data_params,train_params,model_params):
    src_data=data_params['src_data']#<---
    tar_data=data_params['tar_data']#<--
    seq_size=data_params['seq_size']
    total_time_step=data_params['total_time_step']
    shift_step_list=range(1,total_time_step)
    start_shift_step=int(data_params['start_shift_step'])
    remove_feature_list=data_params['remove_feature_list']
    test_features=data_params['test_features']
    domain_src=data_params['domain_src']
    domain_tar=data_params['domain_tar']#<----
    stu_num=data_params['num_stu']
    marker=data_params['marker']
    model_transform=data_params['model_transform']
    save_data_flag=data_params['save_data_flag']
    save_model_flag=data_params['save_model_flag']
    NUM_WORKER = data_params['NUM_WORKER']
    batch_size = data_params['batch_size']
    dat=data_params['combo_file']
    na_fill_type=data_params['na_fill_type']
    gpu_core=int(train_params['gpu_core'])
    best_src_shift=int(train_params['best_src_shift'])#<--
    start=int(data_params['start'])#<--
    end=int(data_params['end'])#<--
    cnt=0
    df=pd.DataFrame()
    device = torch.device(f"cuda:{gpu_core}")
    run_df=dat.iloc[dat['seq'][start:end]]
    
    for i in run_df.index:
        cnt+=1
        L=run_df.loc[i,'len']
        subset=run_df.loc[i,'combo']
        test_feature_list=list(subset)
        feat_arr_src,label_arr_src, label_reg_arr_src=convert_npz_lagged_bkt_shift_mask(src_data,na_fill_type,seq_size,best_src_shift, 
                            remove_feature_list, test_feature_list,domain_src,stu_num,marker,model_transform,save_data_flag)
        print(feat_arr_src.shape)
        print(label_reg_arr_src.shape)
        train_loader_src,val_loader_src,test_loader_src=get_mv_data_mask(feat_arr_src,label_arr_src, label_reg_arr_src,  
                                                                    batch_size,NUM_WORKER)

        if start_shift_step!=0:
            shift_step_list=range(start_shift_step,total_time_step)
        else:
            shift_step_list=shift_step_list
        for j in shift_step_list:
#             if j % 30==0:
            print(f'....creating data for {i}th index, {L} length features at shift step {j}: \n{subset} ')
            # marker=f'tempo_shift{j}_feat{f}'

            
            feat_arr_tar,label_arr_tar, label_reg_arr_tar=convert_npz_lagged_bkt_shift_mask(tar_data,na_fill_type,seq_size,j, remove_feature_list, 
                                                    test_feature_list,domain_tar,stu_num,marker,model_transform,save_data_flag)
            print(feat_arr_tar.shape)
            print(label_reg_arr_tar.shape)
        
            # entire_tar_loader=get_mv_entire_data(feat_arr_tar,label_arr_tar, label_reg_arr_tar,  batch_size,NUM_WORKER)
    
            train_loader_tar,val_loader_tar,test_loader_tar=get_mv_data_mask(feat_arr_tar,label_arr_tar, label_reg_arr_tar,  batch_size,NUM_WORKER)
            dataloaders={
                'train_src':train_loader_src,
                'val_src':val_loader_src,
                'test_src':test_loader_src,
                'train_tar':train_loader_tar,
                'val_tar':val_loader_tar,
                'test_tar':test_loader_tar,

            }
            # ===train para===
            seed=train_params['seed']
            set_seed(seed)         
            out_dir=train_params['out_dir']      
            log_file_name=train_params['log_file_name']+f'{start}-{end}.log'
            out_file_name=train_params['out_file_name']+f'_{start}({start_shift_step})-{end}({total_time_step}).csv'
            learning_rate = train_params['learning_rate']
            n_epochs = train_params['n_epochs']
            weight_decay=train_params['weight_decay']
            warm_up_step_count=train_params['warm_up_step_count']
            label_exist=train_params['label_exist']
            flat_input=train_params['flat_flag']
            lamb=train_params['lamb']#<---

            #:::: Model parameters::::
            input_dim = len(test_feature_list) # From dataset
            output_dim = model_params['output_dim']
            num_layers=model_params['num_layers']
            src_seqL,tar_seqL=model_params['src_seqL'],model_params['tar_seqL']
            hidden_dim = model_params['hidden_dim']  # Lattent dim
            q = model_params['q']  # Query size
            v = model_params['v']  # Value size
            h = model_params['h']  # Number of heads
            N = model_params['N']  # Number of encoder and decoder to stack
            attention_size = model_params['attention_size']  # Attention window size
            dropout = model_params['dropout']  # Dropout rate
            pe = model_params['pe']  # Positional encoding
            chunk_mode = model_params['chunk_mode']   
            n_hiddens=model_params['n_hiddens']
            class_num=model_params['class_num']
            model_name=model_params['model_name']
            transfer_loss_type=model_params['transfer_loss_type']#<---
            transfer_net=model_params['transfer_net']

            if model_name in ['lstm','LSTM','Lstm']:
                model=LSTM_TR(na_fill_type,input_dim, hidden_dim, num_layers, output_dim,transfer_loss=transfer_loss_type).to(device)
                print(f'this is lstm model,you are training with {model_name} model')
            elif model_name in ['adaRNN','adarnn','AdaRNN']:
                print(f'this is adaRNN model,you are training with {model_name} model')
                model=AdaRNN(na_fill_type,use_bottleneck=True, bottleneck_width=64, n_input=input_dim, n_hiddens=n_hiddens,  n_output=class_num, dropout=dropout, len_seq=tar_seqL, model_type=model_name,trans_loss=transfer_loss_type).to(device)
            else:
                print(f'this is transformer model,you are training with {model_name} model')
                model = Transformer_TR(na_fill_type,input_dim, hidden_dim, output_dim, q, v, h, N,transfer_loss_type=transfer_loss_type, attention_size=attention_size,
                dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)

            # print(f'!!!!Optimizer for transfer net needs to be adam or sgd!!!')
            optimizer = NoamOptimizer(model=model, lr=learning_rate, model_size=hidden_dim, warmup=warm_up_step_count,weight_decay=weight_decay)

            checkpoint_format=f'TRANS_MODEL/BKT_TR/{out_dir}/procNum{start}-{end}/stuNUm_{stu_num}_feature{test_feature_list}_shiftStep{j}_SEED{seed}_{model_name}_{domain_src}_in{input_dim}_hid{hidden_dim}_{num_layers}numLayers_lr{learning_rate}_bs{batch_size}_seqL{src_seqL}_dw{weight_decay}_EP{n_epochs}'
            direct_folder=checkpoint_format.split('/stuNUm')[0]
            direct_folder0=direct_folder.split('/procNum')[0]
            os.makedirs(direct_folder0,exist_ok=True)

            if save_model_flag:
                
                log_file=os.path.join(checkpoint_format,log_file_name)
                os.makedirs(checkpoint_format,exist_ok=True)
            else:
                
                log_file=os.path.join(direct_folder0,log_file_name)
                os.makedirs(direct_folder0,exist_ok=True)

            print(f'>>>>train model for {checkpoint_format}<<<<<<')
            if model_name in ['adaRNN','AdaRNN','adarnn','ADARNN']:
                print(f'this is trainer for AdaRNN, you are transfer training {model_name} model')
                transfer_loss_train_list,best_epo_path,best_loss,best_loss_r,best_loss_l1=train_adaRNN_tr_mv(log_file,dataloaders,model,optimizer,checkpoint_format,
                                                                                                             data_params,train_params,model_params)
            else:
                transfer_loss_train_list,best_epo_path,best_loss,best_loss_r,best_loss_l1=train_test_reg_mv_transfer(log_file,dataloaders,model, optimizer,checkpoint_format,data_params,train_params,model_params)
            # transfer_loss_train_list,best_epo_path,best_loss,best_loss_r,best_loss_l1=train_test_reg_mv_transfer(log_file,dataloaders,model, optimizer,checkpoint_format,data_params,train_params,model_params)
            new_df=pd.DataFrame({'source_domain':[domain_src],'target_domain':[domain_tar],'model_name':[model_name],'seqLen':[src_seqL],'src_shift':[j],
                                    'target_shift':[j], 'ckpt':[best_epo_path],'MSE':[best_loss],'RMSE':[best_loss_r],'L1':[best_loss_l1]})
            df=df.append(new_df)
            df.to_csv(os.path.join(direct_folder0,out_file_name),index=False)
    display(df.head(20))
    return df


def train_test_reg_mv_transfer(log_file,dataloaders,model, optimizer,checkpoint_format,data_params,train_params,model_params):
    weight_decay=train_params['weight_decay']
    n_epochs = train_params['n_epochs']
    model_transform=data_params['model_transform']
    label_exist=train_params['label_exist']
    save_model_flag=data_params['save_model_flag']
    gpu_core=int(train_params['gpu_core'])
    source_loader, target_train_loader, val_loader, target_test_loader = dataloaders['train_src'],  dataloaders['train_tar'],dataloaders['val_src'],dataloaders['test_tar']
 
    stop = 0
    best_mean_loss,best_loss, best_loss_r, best_loss_l1=10,10,10,10
    best_epo_list=[]
    loss_train_list,r_loss_train_list,l1_loss_train_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_val_list,r_loss_val_list,l1_loss_val_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_test_list,r_loss_test_list,l1_loss_test_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    transfer_loss_train_list=np.zeros(n_epochs)
    criterion, criterion_l1=nn.MSELoss(),nn.L1Loss()
    n_batch=min(len(source_loader),len(target_train_loader))
    device=torch.device(f'cuda:{gpu_core}')
    if model_transform==True:
        flat_input=False
    else:
        flat_input=True
    for e in range(1,n_epochs+1):
        stop += 1
        running_train_loss,running_train_loss_r,running_train_loss_l1,running_transfer_loss=[],[],[],[],[]
        model.train()
        for (src, tar) in tqdm(zip(source_loader, target_train_loader),total=n_batch):
            
            # optimizer.zero_grad()
            feature_s, label_s, label_reg_s =src
            feature_t, label_t, label_reg_t =tar
            if flat_input:
                label_reg_s=label_reg_s.reshape(-1,1)
                label_reg_t=label_reg_t.reshape(-1,1)
            feature=torch.cat((feature_s,feature_t),dim=0)
            # print(f'feature shape :{feature.shape}') #[512,4,16]
            label_reg=torch.cat((label_reg_s,label_reg_t),dim=0)#[512,1]
            # print(f'feature shape :{label_reg.shape}')
            feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
            pred_all, loss_transfer = model(feature)
            pred_s = pred_all[0:pred_all.shape[0]//2]
            pred_t = pred_all[pred_all.shape[0]//2:]
            loss_s = criterion(pred_s, label_reg[:label_reg.shape[0]//2])
            loss_t = criterion(pred_t, label_reg[label_reg.shape[0]//2:])
            loss_s_r=torch.sqrt(loss_s)
            loss_t_r=torch.sqrt(loss_t)
            loss_l1_s = criterion_l1(pred_s, label_reg[:label_reg.shape[0]//2])
            loss_l1_t = criterion_l1(pred_t, label_reg[label_reg.shape[0]//2:])
            
            total_loss = loss_s + loss_t + weight_decay * loss_transfer
            total_loss_r=loss_s_r+loss_t_r
            total_loss_l1=loss_l1_s+loss_l1_t
            # total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step(total_loss)
            running_train_loss.append(total_loss.item())
            running_train_loss_r.append(total_loss_r.item())
            running_train_loss_l1.append(total_loss_l1.item())
            running_transfer_loss.append(loss_transfer.item())
        epoch_train_loss=np.mean(running_train_loss)
        epoch_train_loss_r=np.mean(running_train_loss_r)
        epoch_train_loss_l1=np.mean(running_train_loss_l1)
        epoch_transfer_loss=np.mean(running_transfer_loss)

        loss_train_list[e-1]=epoch_train_loss
        r_loss_train_list[e-1]=epoch_train_loss_r
        l1_loss_train_list[e-1]=epoch_train_loss_l1
        transfer_loss_train_list[e-1]=epoch_transfer_loss
        epoch_val_loss, epoch_val_loss_r, epoch_val_loss_l1=test_reg_mv(model,val_loader,data_params,train_params,model_params)
        loss_val_list[e-1]=epoch_val_loss
        r_loss_val_list[e-1]=epoch_val_loss_r
        l1_loss_val_list[e-1]=epoch_val_loss_l1
        epoch_test_loss, epoch_test_loss_r, epoch_test_loss_l1=test_reg_mv(model,target_test_loader,data_params,train_params,model_params)
        loss_test_list[e-1]=epoch_test_loss
        r_loss_test_list[e-1]=epoch_test_loss_r
        l1_loss_test_list[e-1]=epoch_test_loss_l1
        pprint(log_file,f'epoch {e:2d}, mean train/val/test MAE loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f}/{epoch_test_loss:.4f},'
        f'mean train/val/test RMSE loss: {epoch_train_loss_r:.4f}/{epoch_val_loss_r:.4f}/{epoch_test_loss_r:.4f},'
        f'mean train/val/test L1 loss: {epoch_train_loss_l1:.4f}/{epoch_val_loss_l1:.4f}/{epoch_test_loss_l1:.4f}'
        f'mean transfer loss: {epoch_transfer_loss:.4f}')
        avg_l1_r_loss=(epoch_test_loss_r+epoch_test_loss_l1)/2
        # st.fmean(epoch_test_loss_r,epoch_test_loss_l1) #fmean() is a feature for python>=3.8
        if  avg_l1_r_loss< best_mean_loss:
            pprint(log_file,f'>>>Found epoch {e:2d} is better than last one!')
            prev_best=copy.deepcopy(best_loss_r)
            best_mean_loss=avg_l1_r_loss
            best_loss=epoch_test_loss
            best_loss_r=epoch_test_loss_r
            best_loss_l1=epoch_test_loss_l1
            best_epo_list.append(e)
            # print(best_epo_list)
            model_name=checkpoint_format.split('/')[-1]
            filepath = model_name+f'_epoch{e}_test_rmse{best_loss_r:.4f}.pkl'
            if save_model_flag:torch.save(model.state_dict(), os.path.join(checkpoint_format,filepath))
            
            if len(best_epo_list)>=2 and best_epo_list[-1]>best_epo_list[-2] and save_model_flag:
                os.remove(os.path.join(checkpoint_format,model_name+f'_epoch{best_epo_list[-2]}_test_rmse{prev_best:.4f}.pkl'))

        if stop >= 20:
            break
    best_epo_path=model_name+f'_epoch{best_epo_list[-1]}_test_rmse{best_loss_r:.4f}.pkl'
    best_path=os.path.join(checkpoint_format,best_epo_path)
    pprint(log_file,
    f'>>>best epoch is {best_epo_list[-1]}, at folder: {checkpoint_format}\n'
    f'path:{best_epo_path}\n'
    f'best Test MAE/RMSE/L1 loss: {best_loss:.4f}/{best_loss_r:.4f}/{best_loss_l1:.4f}')
    fig,ax=plt.subplots() #make it explicit, usually show is using plt, save is to save the figure, the rest is to plot on the ax
    ax.plot(r_loss_train_list, 'o-', label='train_rmse')
    ax.plot(r_loss_val_list, 'o-', label='val_rmse')
    ax.plot(r_loss_test_list, 'o-', label='test_rmse')
    ax.plot(l1_loss_train_list, '--', label='train_L1')
    ax.plot(l1_loss_val_list, '--', label='val_L1')
    ax.plot(l1_loss_test_list, '--', label='test_L1')
    ax.legend()
    if save_model_flag:
        fig.savefig(os.path.join(checkpoint_format,f'{model_name}.jpg'))
        plt.show()
    return transfer_loss_train_list,best_path,best_loss,best_loss_r,best_loss_l1



# --------------below is AdaRNN special  transfer trainer------------------
def test_epoch(model, gpu_core,test_loader, prefix='Test'):
    model.eval()
    # total_loss = 0
    # total_loss_1 = 0
    # total_loss_r = 0
# the downside of using total is it's not taking the loss out of cuda, hence not in the numpy format. later, it wil cause device conflicts
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    device=torch.device(f'cuda:{gpu_core}')
    running_loss,running_r_loss,running_l1_loss=[],[],[]
    with torch.no_grad():
        for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
            feature, label_reg = feature.to(device).float(), label_reg.to(device).float()
            
            pred = model.predict(feature)
            loss = criterion(pred.to(device), label_reg)
            loss_r = torch.sqrt(loss)
            loss_l1 = criterion_1(pred.to(device), label_reg)
            # total_loss += loss.detach().item()
            #detach only detach from the current graph not necessarily go to the cpu and make it numpy
            # .item extracts loss value as python float
            # total_loss_1 += loss_1.detach().item()
            # total_loss_r += loss_r.detach().item()
            running_loss.append(loss.detach().item())
            running_r_loss.append(loss_r.detach().item())
            running_l1_loss.append(loss_l1.detach().item())
    epoch_loss=np.mean(running_loss)
    epoch_loss_r=np.mean(running_r_loss)
    epoch_loss_l1=np.mean(running_l1_loss)
    return epoch_loss, epoch_loss_r, epoch_loss_l1


def train_adaRNN_tr_mv(log_file,dataloaders,model,optimizer,checkpoint_format,data_params,train_params,model_params):
    


    num_layers=model_params['num_layers']
    len_seq=data_params['seq_size']
    len_win=train_params['len_win']#<--
    weight_decay=train_params['weight_decay']
    pre_epoch=train_params['pre_epoch']#<--
    n_epochs = train_params['n_epochs']
    save_model_flag=data_params['save_model_flag']
    gpu_core=int(train_params['gpu_core'])
    stop=0
    best_epo_list=[]
    # train_loader_alg2,val_loader,test_loader_alg2=data_loader['train_alg2'],data_loader['val_alg2'],data_loader['test_alg2']
    # train_loader_geom,val_loader_geom,test_loader_geom=data_loader['train_geom'],data_loader['val_geom'],data_loader['test_geom']
    source_loader, target_train_loader, val_loader, target_test_loader = dataloaders['train_src'],  dataloaders['train_tar'],dataloaders['val_src'],dataloaders['test_tar']
    best_mean_loss,best_loss, best_loss_r, best_loss_l1=10,10,10,10
    loss_train_list,r_loss_train_list,l1_loss_train_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_val_list,r_loss_val_list,l1_loss_val_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    loss_test_list,r_loss_test_list,l1_loss_test_list=np.zeros(n_epochs),np.zeros(n_epochs),np.zeros(n_epochs)
    transfer_loss_train_list=np.zeros(n_epochs)
    device=torch.device(f'cuda:{gpu_core}')
    n_batch=min(len(source_loader),len(target_train_loader))
    for e in range(n_epochs):
        pprint(log_file,'Epoch:', e)
        pprint(log_file,'training...')
        model.train()
        criterion = nn.MSELoss()
        criterion_l1 = nn.L1Loss()
        loss_all = []
        loss_1_all = []
        dist_mat = torch.zeros(num_layers, len_seq).to(device)
        running_train_loss,running_train_loss_r,running_train_loss_l1,running_val_loss,running_transfer_loss=[],[],[],[],[]
        for (src,tar) in tqdm(zip(source_loader,target_train_loader), total=n_batch):
            # optimizer.zero_grad()
            feature_s, label_s, label_reg_s =src
            feature_t, label_t, label_reg_t =tar
            feature=torch.cat((feature_s,feature_t),dim=0)
            label=torch.cat((label_s,label_t),dim=0)
            # .reshape(-1,1)
            label_reg=torch.cat((label_reg_s,label_reg_t),dim=0)
            # .reshape(-1,1)
            # print(f'feature/label/label_reg shape after stacking: {feature.shape}/{label.shape}/{label_reg.shape}')
            feature, label, label_reg = feature.to(device).float(
            ), label.to(device).long(), label_reg.to(device).float()
            total_loss = torch.zeros(1).to(device)
            # print(f'\n input feature/label_reg shape: {feature.shape}/{label_reg.shape}')
            # torch.Size([512, 4, 16])/torch.Size([512, 1]) for =512, seq_size=4, len_win=2
            if e < pre_epoch:
                pred_all, loss_transfer, out_weight_list = model.forward_pre_train(
                    feature, len_win=len_win) #feature shape:32*300*16
            else:
                pred_all, loss_transfer, dist, weight_mat = model.forward_Boosting(
                    feature, weight_mat)
                dist_mat = dist_mat + dist
            pred_s = pred_all[0:pred_all.shape[0]//2]
            pred_t = pred_all[pred_all.shape[0]//2:]
            # print(f'pred_s/t shape: {pred_s.shape}/{pred_t.shape}')
            loss_s = criterion(pred_s, label_reg[:label_reg.shape[0]//2])
            loss_t = criterion(pred_t, label_reg[label_reg.shape[0]//2:])
            loss_s_r=torch.sqrt(loss_s)
            loss_t_r=torch.sqrt(loss_t)
            loss_l1_s = criterion_l1(pred_s, label_reg[:label_reg.shape[0]//2])
            loss_l1_t = criterion_l1(pred_t, label_reg[label_reg.shape[0]//2:])
            
            total_loss = loss_s + loss_t + weight_decay * loss_transfer
            total_loss_r=loss_s_r+loss_t_r
            total_loss_l1=loss_l1_s+loss_l1_t
            # total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step(total_loss)
            running_train_loss.append(total_loss.item())#detach is to detach the gradient from the loss, in training, you want the gradient attached to the loss
            running_train_loss_r.append(total_loss_r.item())
            running_train_loss_l1.append(total_loss_l1.item())
            running_transfer_loss.append(loss_transfer.item())

        epoch_train_loss=np.mean(running_train_loss)
        epoch_train_loss_r=np.mean(running_train_loss_r)
        epoch_train_loss_l1=np.mean(running_train_loss_l1)
        epoch_transfer_loss=np.mean(running_transfer_loss)
        loss_train_list[e-1]=epoch_train_loss
        r_loss_train_list[e-1]=epoch_train_loss_r
        l1_loss_train_list[e-1]=epoch_train_loss_l1
        transfer_loss_train_list[e-1]=epoch_transfer_loss
        epoch_val_loss, epoch_val_loss_r, epoch_val_loss_l1=test_epoch(model, gpu_core,val_loader, prefix='Valid')
        loss_val_list[e-1]=epoch_val_loss
        r_loss_val_list[e-1]=epoch_val_loss_r
        l1_loss_val_list[e-1]=epoch_val_loss_l1
        epoch_test_loss, epoch_test_loss_r, epoch_test_loss_l1=test_epoch(model,gpu_core, target_test_loader, prefix='Test')
        loss_test_list[e-1]=epoch_test_loss
        r_loss_test_list[e-1]=epoch_test_loss_r
        l1_loss_test_list[e-1]=epoch_test_loss_l1
        pprint(log_file,f'epoch {e:2d}, mean train/val/test MAE loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f}/{epoch_test_loss:.4f},'
        f'mean train/val/test RMSE loss: {epoch_train_loss_r:.4f}/{epoch_val_loss_r:.4f}/{epoch_test_loss_r:.4f},'
        f'mean train/val/test L1 loss: {epoch_train_loss_l1:.4f}/{epoch_val_loss_l1:.4f}/{epoch_test_loss_l1:.4f}'
        f'mean transfer loss: {epoch_transfer_loss:.4f}')
        avg_l1_r_loss=(epoch_test_loss_r+epoch_test_loss_l1)/2
        # st.fmean(epoch_test_loss_r,epoch_test_loss_l1) #fmean() is a feature for python>=3.8
        if  avg_l1_r_loss< best_mean_loss:
            pprint(log_file,f'>>>Found epoch {e:2d} is better than last one!')
            prev_best=copy.deepcopy(best_loss_r)
            best_mean_loss=avg_l1_r_loss
            best_loss=epoch_test_loss
            best_loss_r=epoch_test_loss_r
            best_loss_l1=epoch_test_loss_l1
            best_epo_list.append(e)
            # print(best_epo_list)
            model_name=checkpoint_format.split('/')[-1]
            filepath = model_name+f'_epoch{e}_test_rmse{best_loss_r:.4f}.pkl'
            if save_model_flag:torch.save(model.state_dict(), os.path.join(checkpoint_format,filepath))
            
            if len(best_epo_list)>=2 and best_epo_list[-1]>best_epo_list[-2] and save_model_flag:
                os.remove(os.path.join(checkpoint_format,model_name+f'_epoch{best_epo_list[-2]}_test_rmse{prev_best:.4f}.pkl'))

        if stop >= 20:
            break
    best_epo_path=model_name+f'_epoch{best_epo_list[-1]}_test_rmse{best_loss_r:.4f}.pkl'
    best_path=os.path.join(checkpoint_format,best_epo_path)
    pprint(log_file,
    f'>>>best epoch is {best_epo_list[-1]}, at folder: {checkpoint_format}\n'
    f'path:{best_epo_path}\n'
    f'best Test MAE/RMSE/L1 loss: {best_loss:.4f}/{best_loss_r:.4f}/{best_loss_l1:.4f}')
    fig,ax=plt.subplots() #make it explicit, usually show is using plt, save is to save the figure, the rest is to plot on the ax
    ax.plot(r_loss_train_list, 'o-', label='train_rmse')
    ax.plot(r_loss_val_list, 'o-', label='val_rmse')
    ax.plot(r_loss_test_list, 'o-', label='test_rmse')
    ax.plot(l1_loss_train_list, '--', label='train_L1')
    ax.plot(l1_loss_val_list, '--', label='val_L1')
    ax.plot(l1_loss_test_list, '--', label='test_L1')
    ax.legend()

    if save_model_flag:
        
        fig.savefig(os.path.join(checkpoint_format,f'{model_name}.jpg'))
        plt.show()
    return transfer_loss_train_list,best_path,best_loss,best_loss_r,best_loss_l1
