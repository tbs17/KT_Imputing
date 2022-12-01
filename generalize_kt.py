
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

def gen_seed_model_runner(params):
    variate_type=params['variate_type']
    lag_type=params['lag_type']
    file_path=params['file_path']
    domain_src=params['domain_src']
    predict_domain=params['predict_domain']
    model_name=params['model_name']
    model_transform=params['model_transform']
    start_shift_step=params['start_shift_step']
    total_time_steps=params['total_time_steps']

    target_var_pos=params['target_var_pos']#<--
    tag_pos=params['tag_pos']#<--
    pred_type=params['pred_type']



    df=pd.DataFrame()
    for i in range(2,6):
        seed=i
        print(f'\n...start running model for seed {seed}...\n')
        params={
            'file_path':file_path,
            'domain_src':domain_src,
            'predict_domain':predict_domain,
            'model_name':model_name,
            'model_transform':model_transform,
            'start_shift_step':start_shift_step,
            'total_time_steps':total_time_steps,
            'is_continue':False,
            'target_var_pos':target_var_pos,
            'tag_pos':tag_pos,
            'pred_type':pred_type,
            'seed':seed
            }
        if variate_type=='sv':

            results=sv_generalize_runner(params)
        else:

            results=mv_generalize_runner(params)
        results['data']=domain_src
        results['variate_type']=variate_type
        results['seed']=seed
        results['run_type']='model_gen'
        results['lag_type']=lag_type
        df=df.append(results)
        display(df.head())
        out_dir='TRANS_MODEL/MODEL_GEN'
        os.makedirs(out_dir,exist_ok=True)
        df.to_csv(f'{out_dir}/{variate_type}_{domain_src}_{model_name}_{start_shift_step}_{total_time_steps}_seed_model_gen.csv',index=False)
    print(f'obtain {df.shape[0]} models')
    display(df.head())
    return df

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
                    # print('...label exists and mask fill...')#<---
                    feature, _,label_reg,feature_lens=data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    # , feature_lens.to(device).long()
                    input=(feature,feature_lens)
                else:
                    # print('...label exists and other fill...')
                    feature, _,label_reg = data
                    feature,label_reg=feature.to(device).float(),label_reg.to(device).float()
                    input=feature
            else:
                # print('...label does not exist ...')
                feature,label_reg=data
            if flat_input:
                label_reg=label_reg.reshape(-1,1)
            
            
            if transfer_net:
                reg_pred = model.predict(input)
            else:
                # print(f'...it is not transfer net, predict normally...')#<---
                reg_pred = model(input)
            if model_transform:
                reg_pred=reg_pred[:,-1].unsqueeze(1)
            reg_pred=reg_pred.cpu().detach().numpy()
            if na_fill_type!='mask':
                pred_feat=input
            else:
                pred_feat,_=input
            pred_feat=pred_feat.cpu().detach()
            # print(f'feat_arr/reg_pred shape:{pred_feat.shape}/{reg_pred.shape}')#torch.Size([256, 644, 6])/(256, 1)
            dim1_idx,dim2_idx,dim3_idx=torch.nonzero(pred_feat, as_tuple=True) #find all the indices of nonzero values for each dimension
            target_id=dim2_idx.numpy()[-1]
            pred_feat=pred_feat[:,target_id,:]
            pred_feat=pred_feat.numpy()
            # print(f'pred_feat shape:{pred_feat.shape}')#(32, 6)
            # pred_feat=pred_feat.reshape(-1,pred_feat.shape[2])
            feat_df=pd.DataFrame(pred_feat,columns=test_feature_list)
            # reg_pred=reg_pred
            # .squeeze(1)
            # print(f'reg_pred shape:{reg_pred.shape}')#(32,1)

            label_df=pd.DataFrame(reg_pred,columns=['score_rate'])
            reg_pred_df=pd.concat([feat_df,label_df],axis=1)
            pred_df=pred_df.append(reg_pred_df)
   
    # display(pred_df.tail())
    return pred_df
 

def sv_generalize_runner(params):
    file_path=params['file_path']
    domain_src=params['domain_src']
    predict_domain=params['predict_domain']
    model_name=params['model_name']
    model_transform_flag=params['model_transform']
    start_shift_step=params['start_shift_step']
    total_time_steps=params['total_time_steps']
    save_flag=False
    target_var_pos=params['target_var_pos']#<--
    tag_pos=params['tag_pos']#<--
    pred_type=params['pred_type']
    seed=int(params['seed'])
    feat_type='single'
    best_model_path,best_index=get_best_model_path(file_path,feat_type)
    out_dir=best_model_path.split('/stuNUm')[0]
    data_params={
    'data_folder':'.',
    'batch_size':256,
    'domain_src':f'{domain_src}_bkt_v2', 
    'domain_tar' : f'{predict_domain}_bkt_v2', 
    'src_seqL':150,
    'tar_seqL':150,

    'NUM_WORKER':2,
    'target_var_pos':target_var_pos,
    'tag_pos':tag_pos,
    'test_stu_num':None,
    'save_flag':save_flag,
    'start_shift':start_shift_step,
    'total_time_step':total_time_steps,
    
    'log_file_name':'run_predict.log',
    'out_file_name':f'predict_{predict_domain}_sv_{model_name}_best_{start_shift_step}_{total_time_steps}.csv',
    'out_dir':out_dir,
    }
    train_params={
        'seed':seed,
    }
    with open ('scripts/question_num.json','r') as f:
        QUESTION_NUM=json.load(f)
    model_params={

        'input_dim' :150,  # From dataset
        'output_dim' :1,  # From dataset
        'hidden_dim' : 128 , # Lattent dim
        'attn_dim':256,
        'fc_dim':512,
        'dropout' : 0.2,  # Dropout rate
        'num_layers':2,
        'model_name':model_name,
        'checkpoint':best_model_path,
        'question_num':2566,
        'pred_type':pred_type
    }
    predict_result=sv_generalize(data_params,train_params,model_params)
    return predict_result

def sv_generalize(data_params,train_params,model_params):

    data_folder = data_params['data_folder']
    batch_size = data_params['batch_size']
    domain_src, domain_tar = data_params['domain_src'], data_params['domain_tar']
    src_seqL,tar_seqL=data_params['src_seqL'],data_params['tar_seqL']
    save_flag=data_params['save_flag']
    NUM_WORKER=data_params['NUM_WORKER']
    target_var_pos=data_params['target_var_pos']
    test_stu_num=data_params['test_stu_num']
    df=pd.DataFrame()
    start_shift=data_params['start_shift']
    total_time_step=data_params['total_time_step']
    target_var_pos=data_params['target_var_pos']#<--
    tag_pos=data_params['tag_pos']#<--
    pred_type=model_params['pred_type']
    for shift_step in range(start_shift,total_time_step):
        # if shift_step<4:
        #     batch_size=128
        # else:
        #     batch_size=batch_size
        print(f'test student number:{test_stu_num} for shift steps {shift_step}')
        test_loader=load_entire_data_sv_ez_shift(data_folder, domain_tar,test_stu_num,target_var_pos, tag_pos,shift_step,  batch_size,tar_seqL,NUM_WORKER, pred_type, phase='tar')
        
        seed=int(train_params['seed'])
        torch.manual_seed(seed) #torch random seed
        random.seed(seed) #python random seed
        np.random.seed(seed) #numpy random seed          
        log_file_name=data_params['log_file_name']
        out_file_name=data_params['out_file_name']
        out_dir=data_params['out_dir']
        input_dim = model_params['input_dim'] # From dataset
        output_dim = model_params['output_dim']
        num_layers=model_params['num_layers']
        hidden_dim = model_params['hidden_dim']  # Lattent dim
        attn_dim=model_params['attn_dim']
        fc_dim=model_params['fc_dim']
        dropout = model_params['dropout']  # Dropout rate
        model_name=model_params['model_name']
        checkpoint=model_params['checkpoint']
        question_num=model_params['question_num']
        if model_name in ['DKT','dkt']:

            model = DKT_reg_sv(input_dim, hidden_dim, question_num,num_layers, dropout).cuda()

        else:
            model=NPA_reg_sv(input_dim, hidden_dim, attn_dim, fc_dim,num_layers, question_num, dropout).cuda()
        
        model.load_state_dict(torch.load(checkpoint),strict=False)
        criterion = nn.MSELoss()
        criterion_l1=nn.L1Loss()
        # checkpoint=f'TRANS_MODEL/BKT_SV/{out_dir}/stuNUm_{stu_num}_feature{feat_type}_shiftStep{shift_step}_SEED{seed}_{model_name}_{domain_src}_in{input_dim}_hid{hidden_dim}_{num_layers}numLayers_lr{learning_rate}_bs{batch_size}_seqL{src_seqL}_dw{weight_decay}_EP{n_epochs}'
        direct_folder=checkpoint.split('/stuNUm')[0]
        os.makedirs(direct_folder,exist_ok=True)
        # if save_flag:
            
        log_file=os.path.join(direct_folder,log_file_name)

        epoch_test_loss, epoch_test_loss_r, epoch_test_loss_l1=test_reg(model, train_params,model_params,test_loader)
            # model, criterion, criterion_l1,target_data_loader,transfer_net=False)
        pprint(log_file,f'mean test MAE loss: {epoch_test_loss:.4f}, \
        mean test RMSE loss: {epoch_test_loss_r:.4f}, mean test L1 loss: {epoch_test_loss_l1:.4f}, ')
        new_df=pd.DataFrame({'source_domain':[domain_src],'target_domain':[domain_tar],'model_name':[model_name],'seqLen':[src_seqL],'shift':[shift_step],'domain_tar':[domain_tar],
                                'ckpt':[checkpoint],'MSE':[epoch_test_loss],'RMSE':[epoch_test_loss_r],'L1':[epoch_test_loss_l1]})
        df=df.append(new_df)
        df.to_csv(os.path.join(direct_folder,out_file_name),index=False)
        display(df.tail())
    print(f'=======we have run total {src_seqL-1} timeStep combos!=====')
    display(df.head(20))
    return df


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

 

def mv_generalize_old(data_params,train_params,model_params):
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

        
        feat_arr,label_arr, label_reg_arr,feat_lens,processed_data=convert_npz_lagged_bkt_shift_mask(save_dir,data,na_fill_type,seq_size,j, remove_feature_list, test_feature_list,domain_tar,stu_num,marker,model_transform,save_flag_data)

        feat=feat_arr
        label_arr=label_arr
        label_reg_arr=label_reg_arr
        print(feat_arr.shape)
        print(label_arr.shape)
        print(label_reg_arr.shape)
        # test_loader=get_mv_entire_data(feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER)
        is_predict=True
        test_loader=get_mv_data_mask(is_predict,na_fill_type,feat_lens,feat_arr,label_arr, label_reg_arr,  batch_size,NUM_WORKER)
        # dataloaders={
        #     'test':test_loader
        # }

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
            concat_df.to_csv(os.path.join(direct_folder,pred_file_name),index=False)
    return concat_df
        


# --------below is MV generalizer---------


def mv_generalize_runner(params):
    model_path=params['model_path']
    file_path=params['file_path']
    domain_src=params['domain_src']
    predict_domain=params['predict_domain']
    model_name=params['model_name']
    model_transform_flag=params['model_transform']
    data_marker=params['data_marker']
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
    is_continue=params['is_continue']
    save_dir=params['save_dir']
    feat_type='multi'
    # if is_predict:
    #     out_dir=model_path.split('/procNum')[0]
    #     best_index=254
    # else:
    if file_path:

        file_path=params['file_path']
        best_model_path,best_index=get_best_model_path(file_path,feat_type)
        print(f'best index is: {best_index}')
    else:
        best_model_path=model_path
        best_index=254
    out_dir=best_model_path.split('/procNum')[0]
    start,end=best_index,best_index+1
 
    # save_flag_data=False
    # save_flag_model=True
    
    if data_marker:
        predict_domain=data_marker
   
    combo_file=f'{root_dir}/TRANS_DATA/combination_no_prof_complete.csv'
    df=pd.read_csv(data_path)
    dat=pd.read_csv(combo_file)
    dat['combo']=dat['combo'].apply(lambda x:literal_eval(x))
    dat['seq']=dat.index
    feature_file=f'{root_dir}/TRANS_DATA/test_features.json'
    
    with open(feature_file,'r') as f:
        features=json.load(f)
    test_features=features['test_features']
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
        'marker':None,
        'model_transform':model_transform,
        'save_flag_data':save_flag_data,
        'save_flag_model':save_flag_model,
        'NUM_WORKER' : 2,
        'combo_file':dat,
        'domain_src':domain_src,
        'domain_tar':predict_domain,
        'start_shift_step':start_shift_step,
      
        'na_fill_type':na_fill_type,
        'log_file_name':'run_predict.log',
        'out_file_name':f'predict_{predict_domain}_{model_name}_MV_{total_time_steps-start_shift_step}lags_{best_index}thFeat',
        'out_dir':out_dir,
        'start':start,
        'end':end,
        'is_predict':is_predict,
        'save_dir':save_dir,
        'test_features':test_features
    }
    train_params={
        'label_exist':True,
        'flat_flag':flat_flag,
          'gpu_core':0,
          'is_continue':is_continue,
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
        # 'input_dim' :len(test_features),  # From dataset
        'output_dim' :1,  # From dataset
        'num_layers':2,
        'src_seqL':150,
        'tar_seqL':150,
        'n_hiddens':[64,64],
        'class_num':1,
        'model_name':model_name,
        'checkpoint':best_model_path,
        'transfer_net':False,

        
    }
    predict_result=mv_generalize(data_params,train_params,model_params)
    return predict_result


def mv_generalize(data_params,train_params,model_params):
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
    dat=data_params['combo_file']
    start=data_params['start']
    end=data_params['end']
    na_fill_type=data_params['na_fill_type']
    save_dir=data_params['save_dir']
    gpu_core=int(train_params['gpu_core'])
    is_continue=train_params['is_continue']
    remove_feature_list=''
    
    df=pd.DataFrame()
    device = torch.device(f"cuda:{gpu_core}")
    run_df=dat.iloc[dat['seq'][start:end]]
    is_predict=data_params['is_predict']
    if is_predict:
        pred_df=pd.DataFrame()
        pred_file_name=data_params['out_file_name']+f'_pred_{na_fill_type}.csv'
    cnt=0
    for i in run_df.index:
        cnt+=1
        L=run_df.loc[i,'len']
        subset=run_df.loc[i,'combo']
        test_feature_list=list(subset)
        data_params['test_features']=test_feature_list
        if start_shift_step>1 :
            shift_step_list=range(start_shift_step,total_time_step)
        else:
            shift_step_list=shift_step_list
        for j in shift_step_list:
    #             if j % 30==0:
            print(f'....creating data for {i}th index, {L} length features at shift step {j}: \n{subset} ')
            # marker=f'tempo_shift{j}_feat{f}'
            
            feat_arr,label_arr, label_reg_arr,feat_lens,processed_data=convert_npz_lagged_bkt_shift_mask(save_dir,data,na_fill_type,seq_size,j, remove_feature_list, test_feature_list,domain_tar,stu_num,marker,model_transform,save_flag_data)
            # data,na_fill_type,seq_size,shift_step, remove_feature_list, test_feature_list,course_name,num_stu,marker,model_transform,save_flag
            feat=feat_arr
            label_arr=label_arr
            label_reg_arr=label_reg_arr
            print(feat_arr.shape)
            print(label_arr.shape)
            print(label_reg_arr.shape)
            # test_loader=get_mv_entire_data(feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER)
            is_predict=True
            test_loader=get_mv_data_mask(is_predict,na_fill_type,feat_lens,feat_arr,label_arr, label_reg_arr,  batch_size,NUM_WORKER)
            # dataloaders={
            #     'test':test_loader
            # }

            seed=int(train_params['seed'])
            import random
            torch.manual_seed(seed) #torch random seed
            random.seed(seed) #python random seed
            np.random.seed(seed) #numpy random seed 

            # :::train params:::
            # stu_num=train_params['stu_num']
         
            
            log_file_name=data_params['log_file_name']+f'{start}-{end}.log'
            out_file_name=data_params['out_file_name']+f'_{start}({start_shift_step})-{end}.csv'
        
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
                model=AdaRNN_noTL(use_bottleneck=True, bottleneck_width=64, n_input=input_dim, n_hiddens=n_hiddens,  n_output=class_num, dropout=dropout, len_seq=seq_size, model_type=model_name).to(device)
            else:
                print(f'This is transformer model, you are predicting with {model_name} model')
                model = Transformer(input_dim, hidden_dim, output_dim, q, v, h, N, attention_size=attention_size,
                dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
            model.load_state_dict(torch.load(checkpoint),strict=False)
            criterion = nn.MSELoss()
            criterion_l1=nn.L1Loss()
            direct_folder=checkpoint.split('/procNum')[0]
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
                # concat_df_n=concat_df.loc[:,~concat_df.columns.duplicated()]
                int_cols=['student_id', 'bktcount','seq_number', 'assessment_duration', 'count','quest_ref_1', 'item_ref_1', 'usmoid_1', 'question_type_1']
                for col in int_cols:

                    concat_df[col]=concat_df[col].astype(int)
                print(f' after concat with input data: {concat_df.shape}')
                display(concat_df.head())
                print(f'..saving the predicted data under {direct_folder}/{pred_file_name}...')
                concat_df.to_csv(os.path.join(direct_folder,pred_file_name),index=False)
                return concat_df
            else:
                epoch_test_loss, epoch_test_loss_r, epoch_test_loss_l1=test_reg_mv(model,test_loader,data_params,train_params,model_params)
                new_df=pd.DataFrame({'source_domain':[domain_src],'target_domain':[domain_tar],'model_name':[model_name],'combo_index':[i],'features':[subset],'shift':[j],
                                    'ckpt':[checkpoint],'MSE':[epoch_test_loss],'RMSE':[epoch_test_loss_r],'L1':[epoch_test_loss_l1]})
                df=df.append(new_df)
                # if j%5==0:
                df.to_csv(os.path.join(direct_folder,out_file_name),index=False)
                display(df.tail())
                if is_continue:
                    start_shift_step=1
                    shift_step_list=range(1,total_time_step)
                else:
                    pass
                print(f'=======we have run total {cnt} feature combos!=====')
                display(df.head(20))
                return df


def get_mv_entire_data(feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER):

    ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    feat=feat.reshape(-1, feat.shape[2])
    scaler=MinMaxScaler()
    feat=scaler.fit_transform(feat)
    feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
    test_set=mv_data_set(feat, label_arr, label_reg_arr)
    test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)

    print(f'test loader length:{len(test_loader)}')
    return test_loader


def load_entire_data_sv_ez_shift(root_path, domain, stu_num,target_var_pos,tag_pos, shift_step,batch_size,seqL,NUM_WORKER,pred_type,phase):
    user_base_path = f'{root_path}/{domain}/processed/1'
    sample_infos=[]
    question_num=QUESTION_NUM[domain]
    for split_type in os.listdir(user_base_path):

        split_type_data_path=f'{user_base_path}/{split_type}/'
        print(split_type_data_path)
        split_type_sample_infos, num_of_split_type_user = get_data_user_sep_sv_shift(split_type_data_path,stu_num,shift_step)
        sample_infos.extend(split_type_sample_infos)
    print(f'sample_infos length {len(sample_infos)}')
    split_type_data =UserSepDataset_amp_sv(split_type, sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type) #the return is a list of dictionaries b/c the get_sequence function is only per index
    
    data_loader = torch.utils.data.DataLoader(dataset=split_type_data, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=NUM_WORKER)
    print(f'data loader for {split_type} len:{len(data_loader)}!')
    return data_loader

