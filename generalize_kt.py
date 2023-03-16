
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.nn as nn
import os
from network.util_network import ScheduledOptim, NoamOpt
from torch.utils.data import TensorDataset,DataLoader
from torch.utils import data
from sklearn.metrics import roc_auc_score
from data_process import *
import torch, re
from torch.utils.data import Dataset
from constant import *
from train_test_mv import *
from train_test_sv import *
from kt_utils import *
from data_process import *
from timeit import default_timer as timer


def test_reg_mv(model,test_loader,data_params,train_params,model_params):
    gpu_core=int(train_params['gpu_core'])
    label_exist=train_params['label_exist']
    transfer_net=model_params['transfer_net']
    flat_input=train_params['flat_flag']#<--
    # import statistics as st
    model.eval()
    criterion, criterion_l1=nn.MSELoss(),nn.L1Loss()
    len_dataset = len(test_loader.dataset)
    running_loss,running_r_loss,running_l1_loss=[],[],[]
    device=torch.device(f'cuda:{gpu_core}')
    with torch.no_grad():
        for data in test_loader:
          
            if label_exist:
                feature, _,label_reg = data
            else:
                feature,label_reg=data
            if flat_input:
                label_reg=label_reg.reshape(-1,1)
            
            feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
            if transfer_net:
                reg_pred = model.predict(feature)
            else:
                reg_pred = model(feature)
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



def predict_reg_mv(model,test_loader,data_params,train_params,model_params):
    gpu_core=int(train_params['gpu_core'])
    label_exist=train_params['label_exist']
    transfer_net=model_params['transfer_net']
    flat_input=train_params['flat_flag']#<--
    # import statistics as st
    na_fill_type=data_params['na_fill_type']
    model_transform=data_params['model_transform']
    test_feature_list=data_params['test_features']
    model.eval()
    criterion, criterion_l1=nn.MSELoss(),nn.L1Loss()
    # len_dataset = len(test_loader.dataset)
    running_loss,running_r_loss,running_l1_loss=[],[],[]
    device=torch.device(f'cuda:{gpu_core}')
    pred_df=pd.DataFrame()
    with torch.no_grad():
        for data in tqdm(test_loader,total=len(test_loader)):
          
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
            
            
            if transfer_net:
                reg_pred = model.predict(input)
            else:
    
                reg_pred = model(input)
            if model_transform:
                reg_pred=reg_pred[:,-1].unsqueeze(1)
            reg_pred=reg_pred.cpu().detach().numpy()
            if na_fill_type!='mask':
                pred_feat=input
            else:
                pred_feat,_=input
            pred_feat=pred_feat.cpu().detach()
          
            dim1_idx,dim2_idx,dim3_idx=torch.nonzero(pred_feat, as_tuple=True) #find all the indices of nonzero values for each dimension
            target_id=dim2_idx.numpy()[-1]
            pred_feat=pred_feat[:,target_id,:]
            pred_feat=pred_feat.numpy()

            feat_df=pd.DataFrame(pred_feat,columns=test_feature_list)
      

            label_df=pd.DataFrame(reg_pred,columns=['score_rate'])
            reg_pred_df=pd.concat([feat_df,label_df],axis=1)
            pred_df=pred_df.append(reg_pred_df)
   

    return pred_df
 
# ====below is the new generalizer for mv data using all features============
def mv_generalize_runner_new(params):
    model_path=params['model_path']
 
    domain_src=params['domain_src']

    model_name=params['model_name']
    model_transform_flag=params['model_transform']
    marker=params['marker']
    na_fill_type=params['na_fill_type']
    seed=int(params['seed'])
    is_predict=params['is_predict']
    data_path=params['data_path']
    root_dir=params['root_dir']
    batch_size=params['batch_size']
    seq_size=params['seq_size']
    save_flag_model=params['save_flag_model']
    save_flag_data=params['save_flag_data']
    start_shift_step=params['start_shift_step']
    total_time_steps=params['total_time_steps']
    test_features=params['test_features']
    save_dir=params['save_dir']
    orig_data_path=params['orig_data_path']
    save_mixed_data=params['save_mixed_data']
    
    out_dir=model_path.split('/stuNUm')[0]
 

    orig_data=pd.read_csv(orig_data_path)
    df=pd.read_csv(data_path)
    start=timer()
    if model_transform_flag==0:
        flat_flag=True
        model_transform=False
    else:
        flat_flag=False
        model_transform=True
    data_params={
        'data':df,
        'batch_size' : batch_size,
        'seq_size':seq_size,
        'total_time_step':total_time_steps, 
        'remove_feature_list':'', 
        'num_stu':None,
        'marker':marker,
        'model_transform':model_transform,
        'save_flag_data':save_flag_data,
        'save_flag_model':save_flag_model,
        'NUM_WORKER' : 0,
        'domain_src':domain_src,
        'domain_tar':domain_src,
        'start_shift_step':start_shift_step,
        'na_fill_type':na_fill_type,
        'log_file_name':'run_predict.log',
        'out_file_name':f'predict_{domain_src}_{model_name}_MV_{total_time_steps}-{start_shift_step}_{len(test_features)}Feats',
        'out_dir':out_dir,
        'is_predict':is_predict,
        'save_dir':save_dir,
        'test_features':test_features,
   
    }
    train_params={
        'label_exist':True,
        'flat_flag':flat_flag,
          'gpu_core':0,

          'seed':seed,
    }
    with open (f'{root_dir}/scripts/question_num.json','r') as f:
        QUESTION_NUM=json.load(f)
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
      
        'output_dim' :1,  # From dataset
        'num_layers':2,
        'src_seqL':seq_size,
        'tar_seqL':seq_size,
        'n_hiddens':[64,64],
        'class_num':1,
        'model_name':model_name,
        'checkpoint':model_path,
        'transfer_net':False,

        
    }
    predict_result=mv_generalize_new(data_params,train_params,model_params)
    cols_keep=predict_result.columns
    orig_data_n=orig_data[cols_keep]
    data_new=pd.concat([orig_data_n,predict_result])
    print(f' mixed with original data shape:{data_new.shape}')
    display(data_new.head())
    save_path=data_path.split('.csv')[0].split('/')[-1]
    
    if save_mixed_data:
        print(f'..saving the mixed data under {save_path}_{na_fill_type}_{marker}.csv... ')
        data_new.to_csv(f'TRANS_DATA/{save_path}_{na_fill_type}_{marker}.csv',index=False)
    end=timer()
    min,sec=divmod((end-start),60)
    print(f'..It took {min} and {sec:.2f} seconds to finish generalizing...')
    return predict_result,data_new



def mv_generalize_new(data_params,train_params,model_params):
    from itertools import combinations
    from ast import literal_eval
    data=data_params['data']
    seq_size=data_params['seq_size']
    total_time_step=data_params['total_time_step']
    shift_step_list=range(1,total_time_step)
    start_shift_step=int(data_params['start_shift_step'])
    remove_feature_list=data_params['remove_feature_list']
    test_features=data_params['test_features']
    domain_src=data_params['domain_src']
    domain_tar=data_params['domain_tar']
    stu_num=data_params['num_stu']
    marker=data_params['marker']
    model_transform=data_params['model_transform']
    save_flag_model=data_params['save_flag_model']
    save_flag_data=data_params['save_flag_data']
    NUM_WORKER = data_params['NUM_WORKER']
    batch_size = data_params['batch_size']
    
    na_fill_type=data_params['na_fill_type']
    save_dir=data_params['save_dir']
    gpu_core=int(train_params['gpu_core'])
    

    
    df=pd.DataFrame()
    device = torch.device(f"cuda:{gpu_core}")

    is_predict=data_params['is_predict']
    if is_predict:
        pred_df=pd.DataFrame()
        pred_file_name=data_params['out_file_name']+f'_pred_{na_fill_type}_{marker}.csv'
    cnt=0
    
    test_feature_list=test_features
    if start_shift_step>1 :
        shift_step_list=range(start_shift_step,total_time_step)
    else:
        shift_step_list=shift_step_list
    for j in shift_step_list:

        print(f'....creating data at {j} shift step for features:: \n{test_feature_list} ')

        
        
                    
        test_loader,processed_data=generate_indiv_loader(batch_size,save_dir,data,na_fill_type,seq_size,j, test_feature_list,domain_src,stu_num,marker,model_transform,save_flag_data)
 
        seed=int(train_params['seed'])
        set_seed(seed)
        
        log_file_name=data_params['log_file_name']+f'_{start_shift_step}-{total_time_step}.log'
        out_file_name=data_params['out_file_name']+f'_{start_shift_step}-{total_time_step}.csv'
    
        #:::: Model parameters::::
        input_dim = len(test_feature_list) # From dataset
        output_dim = model_params['output_dim']
        num_layers=model_params['num_layers']

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
        checkpoint=model_params['checkpoint']
        

        print(f'================predict for --{test_feature_list}--shiftStep{j}=================')
        
        if model_name in ['lstm','LSTM','Lstm']:
            print(f'This is LSTM model, you are predicting with {model_name} model')
            model=LSTM(na_fill_type,input_dim, hidden_dim, num_layers, output_dim,device).to(device)
        elif model_name in ['adaRNN','adarnn','AdaRNN']:
            print(f'This is adaRNN model, you are predicting with {model_name} model')
            model=AdaRNN_noTL(na_fill_type,use_bottleneck=True, bottleneck_width=64, n_input=input_dim, n_hiddens=n_hiddens,  n_output=class_num, dropout=dropout, len_seq=seq_size, model_type=model_name).to(device)
        else:
            print(f'This is transformer model, you are predicting with {model_name} model')
            model = Transformer(na_fill_type,input_dim, hidden_dim, output_dim, q, v, h, N, attention_size=attention_size,
            dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
        model.load_state_dict(torch.load(checkpoint),strict=False)
        criterion = nn.MSELoss()
        criterion_l1=nn.L1Loss()
        direct_folder=checkpoint.split('/stuNUm')[0]
        os.makedirs(direct_folder,exist_ok=True)
        
        if save_flag_model:
            
            log_file=os.path.join(checkpoint,log_file_name)

        print(f'>>>>predict model from {checkpoint}<<<<<<')
        if is_predict:
            pred_df=predict_reg_mv(model,test_loader,data_params,train_params,model_params)
            processed_data=processed_data.reset_index(drop=True)
            print(f'processed_data/pred df shape:{processed_data.shape}/{pred_df.shape}')
            pred_len=pred_df.shape[0]
            pred_df_n=pred_df[['score_rate']].reset_index(drop=True)
            concat_df=pd.concat([processed_data.iloc[:pred_len],pred_df_n],axis=1)
            display(concat_df.head())
            int_cols=['student_id', 'bktcount','seq_number', 'assessment_duration', 'count','quest_ref_1', 'item_ref_1', 'usmoid_1', 'question_type_1']
            for col in int_cols:
                concat_df[col]=concat_df[col].astype(int)
            print(f' after concat with input data: {concat_df.shape}')
            display(concat_df.head())
            print(f'..saving the predicted data under {direct_folder}/{pred_file_name}...')
            concat_df.to_csv(os.path.join(direct_folder,pred_file_name),index=False)

