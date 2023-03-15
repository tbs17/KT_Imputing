from constant import *
import sys, os, re, copy, time, random,torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.utils import data
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from kt_utils import *
from torch.nn.utils.rnn import pad_sequence
# QUESTION_NUM={'alg2_bkt_v2': 2256, 'geom_bkt_V2': 2256}

from utils import *
# ===================below is data processing for multi variate=========================

# === data_loader func=====


def scale_create_dataloader(is_predict,na_fill_type,feat_lens,feat,label_arr, label_reg_arr,  batch_size):
    if na_fill_type!='mask':

        ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
        feat=feat.reshape(-1, feat.shape[2])
        scaler=MinMaxScaler()
        feat=scaler.fit_transform(feat)
        feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
        feat_lens=None

    else:
        ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
        feat=feat.reshape(-1, feat.shape[2])
        scaler=MinMaxScaler()
        feat=scaler.fit_transform(feat)
        feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
    if is_predict:
        predict_set=mv_data_set(na_fill_type, feat, label_arr, label_reg_arr,feat_lens)
        predict_loader=torch.utils.data.DataLoader(dataset=predict_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=0)
        return predict_loader
    else:

        data_set=mv_data_set(na_fill_type,feat, label_arr, label_reg_arr,feat_lens)
        data_loader=torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, drop_last=True,num_workers=0)
        return data_loader



def get_train_val_test(save_path,data,ratios):
    
    total_stu_num=data['student_id'].nunique()
    stu_num_list=[int(total_stu_num*x) for x in ratios]
    total_stu=data['student_id'].unique()

    train_stu_num,val_stu_num,test_stu_num=stu_num_list[0],stu_num_list[1],stu_num_list[2]
        
    stu_train=total_stu[:train_stu_num]
    stu_val=total_stu[train_stu_num:train_stu_num+val_stu_num]
    stu_test=total_stu[train_stu_num+val_stu_num:train_stu_num+val_stu_num+test_stu_num]
    train_stu_df=pd.DataFrame({'student_id':stu_train})
    train_stu_df['split']='train'
    val_stu_df=pd.DataFrame({'student_id':stu_val})
    val_stu_df['split']='val'
    test_stu_df=pd.DataFrame({'student_id':stu_test})
    test_stu_df['split']='test'
    stu_list_df=pd.concat([train_stu_df,val_stu_df,test_stu_df])
    print(f'..saving train/val/test student list into {save_path}_train_val_test_stu_list.csv... ')
    stu_list_df.to_csv(f'{save_path}_train_val_test_stu_list.csv',index=False)
    train_df=data[data['student_id'].isin(stu_train)]
    val_df=data[data['student_id'].isin(stu_val)]
    test_df=data[data['student_id'].isin(stu_test)]

    print(f'.. train/val/test data split by students has shape {train_df.shape}/{val_df.shape}/{test_df.shape}')
       

    return train_df,val_df,test_df


def encode_feature_v2(data,feature):
    
    lb=LabelEncoder()
    data[feature]=data[feature].astype(str)
    data[feature+'_1']=lb.fit_transform(data[feature])
    new_label=list(data[feature+'_1'].unique())
    orig_label=lb.inverse_transform(new_label)
    map_dict={}
    for i, nl in enumerate(new_label):
        map_dict[nl]=orig_label[i]
    data.drop(feature,axis=1,inplace=True)
    data=data.rename(columns={f'{feature}_1':feature})
    orig_cols=data.columns
    return orig_cols, map_dict,data



def generate_loader(ratios,batch_size,augment_data_path,save_dir,data,na_fill_type,seq_size,shift_step, test_feature_list,course_name,num_stu,marker,model_transform,save_flag):

    if test_feature_list:
        feat_cols=test_feature_list
    if  augment_data_path:
   
        augment_data=pd.read_csv(augment_data_path)
        print(f'augmented data shape:{augment_data.shape}')
        augment_data_cols=augment_data.columns
        data=data[augment_data_cols]
        total_students=data['student_id'].nunique()
        aug_sub=augment_data[augment_data['student_id'].isin(data['student_id'].unique())]
        aug_stu_match=aug_sub['student_id'].nunique()
        print(f'there are {aug_stu_match} students matched in augmented data out of total {total_students} students')
        aug_orig=pd.concat([data,aug_sub])
        print(f'shape after adding augmented data:{aug_orig.shape}')
        split_stu_ids=pd.read_csv(f'TRANS_DATA/{course_name}_no_outlier_downstream_train_val_test_stu_list.csv')
        aug_orig_merg=pd.merge(aug_orig,split_stu_ids,how='left',on='student_id')
        aug_orig_merg=aug_orig_merg.sort_values(['student_id','event_time','seq_number'])
        feature='event_time'
        orig_cols, map_dict,_=encode_feature_v2(aug_orig_merg,feature)
        data_train=aug_orig_merg[aug_orig_merg['split']=='train']
        data_val=aug_orig_merg[aug_orig_merg['split']=='val']
        data_test=aug_orig_merg[aug_orig_merg['split']=='test']
        print(f'...there are {len(data_train)}/{len(data_val)}/{len(data_test)} rows in train/val/test data after adding augmented data')
    else:
        df=data
        print(f'shape without adding augmented data:{df.shape}')
        df=df.sort_values(['student_id','event_time','seq_number'])
        feature='event_time'
        orig_cols, map_dict,_=encode_feature_v2(df,feature)
        total_stu_num=df['student_id'].nunique()
        print(f'original data shape:{df.shape} with {total_stu_num} total students')
        save_path=f'TRANS_DATA/{course_name}_no_outlier_downstream'
        print(f'..saving the split train/val/test student list to a json file under {save_path}_train_val_test_stu_list.json..')
        data_train,data_val,data_test=get_train_val_test(save_path,df,ratios)
    data_train_stus=data_train['student_id'].nunique()
    data_val_stus=data_val['student_id'].nunique()
    data_test_stus=data_test['student_id'].nunique()
    print(f'...there are {data_train_stus}/{data_val_stus}/{data_test_stus} students in the train/test/val data...')
    # if augment_data_path:
    #     data_train=pd.concat([data_train,augment_data])
    #     data_train=data_train.sort_values(by=['student_id','event_time','seq_number'])

# create npz for train/val/test
    print('\n ...generate train data...')
    _=generate_npz(batch_size,map_dict,orig_cols,save_dir,data_train,na_fill_type,seq_size,shift_step,  feat_cols,course_name,num_stu,marker,model_transform,save_flag)
    train_loader=_[-1]
    print('\n ...generate val data...')
    _=generate_npz(batch_size,map_dict,orig_cols,save_dir,data_val,na_fill_type,seq_size,shift_step,  feat_cols,course_name,num_stu,marker,model_transform,save_flag)
    val_loader=_[-1]
    print('\n ...generate test data...')
    _=generate_npz(batch_size,map_dict,orig_cols,save_dir,data_test,na_fill_type,seq_size,shift_step,  feat_cols,course_name,num_stu,marker,model_transform,save_flag)
    test_loader=_[-1]
    return train_loader,val_loader,test_loader



def generate_npz(batch_size,map_dict,orig_cols,save_dir,df,na_fill_type,seq_size,shift_step,  feat_cols,course_name,num_stu,marker,model_transform,save_flag):
    """
    this func is used to train the observed data for lstm, adaRNN, transformer w/o imputed data
    
    """
    
    time_step_list=[]
    feat_col_len=len(feat_cols)
    feat_arr_all=np.zeros([1,feat_col_len],dtype=float)
    feat_arr_all_mask=[]
    feat_arr_all_full=[]
    # feat_full_df=pd.DataFrame()
    label_arr_all,label_reg_arr_all=[],[]

    if num_stu:
        stu_uni=list(df['student_id'].unique())[:num_stu]
    else:
        stu_uni=list(df['student_id'].unique())
        print(f'there are {len(stu_uni)} unique students')
    cnt=0
    cnt_null=0
    for i, student_id in enumerate(stu_uni):
     
        student_df=df[df['student_id']==student_id]
        time_step=student_df.shape[0]
        # print(f'{i}th student has length: {time_step}')
        time_step_list.append(time_step)
        feat_arr,label_arr,label_reg_arr=[],[],[]
        PAD_INDEX=0
        if i%500==0:print(i, student_id, time_step)
        stop=time_step-1-shift_step
        for step in range(time_step-1,stop,-1):
            # print(f'...data process range:{time_step-1}-{stop}...') # the largest seq length is 624
            data=student_df.iloc[:step+1]
            # print(f'step: {step}')
            
            data_len=len(data)
            if data_len!=0:
                if na_fill_type=='mask':
                    feat=torch.from_numpy(data[feat_cols].values)
                    
                    label=torch.from_numpy(np.array(data['quest_ref_1']).reshape(-1,1))
                    label_reg=torch.from_numpy(np.array(data['score_rate']).reshape(-1,1))
                    # data=label_encode(data,'event_time')

                    feat_full=data.iloc[-1].values
                    feat_arr_all_full.append(feat_full)
                    feat_arr_all_mask.append(feat)
                    label_arr_all.append(label)
                    label_reg_arr_all.append(label_reg)
                    cnt+=1
                else:
                    if data_len<seq_size:
                        pad_counts=seq_size-data_len
                        if na_fill_type=='bfill':
                            # print(data.head())
                            PAD_INDEX=data.iloc[0]['score_rate']#the score_rate is on the 6th column
                            pad_arr=np.array([data.iloc[0][feat_cols]]) # the 9 features are from 7th to 16th column
                            # print(f'we will back fill on the target values of :{PAD_INDEX}'
                        # f' pad array of {pad_arr}')
                        elif na_fill_type=='ffill':
                            # print(f'step/data_len/data shape:{step}/{data_len}/{data.shape}')
                            PAD_INDEX=data.iloc[data_len-1]['score_rate']#the score_rate is on the 6th column
                            pad_arr=np.array([data.iloc[data_len-1][feat_cols]])
                    
                            # print(f'we will forward fill on the target values of :{PAD_INDEX}'
                        # f' pad array of {pad_arr}')
                        else:
                            PAD_INDEX=0#the score_rate is on the 6th column
                            pad_arr=np.zeros([1,feat_col_len],dtype=int)
                            # print(f'we will zero fill on the target values of :{PAD_INDEX}'
                        # f' pad array of {pad_arr}')

                
                    else:
                        
                        data=data[-seq_size:]
                        pad_counts = 0
                        PAD_INDEX=0
                        pad_arr=np.zeros([1,feat_col_len],dtype=int)
                        # print('seq length smaller than the actual sequence')
                        # print(f'pad arr: {pad_arr} ')
                    feat_full=data.iloc[-1].values
                    feat_arr_all_full.append(feat_full)
                    paddings=[PAD_INDEX]*pad_counts
                    padding_matrix=np.repeat(pad_arr,repeats=pad_counts,axis=0)
                    feat=data[feat_cols].values
                    label=data['quest_ref_1'].tolist()
                    label_reg=data['score_rate'].tolist()
                    # print(f'padding matrix shape:{padding_matrix.shape} and feat shape: {feat.shape}')
                    if na_fill_type=='bfill':
                        feat=np.concatenate((padding_matrix, feat),axis=0) 
                        label=np.array([paddings+label]).reshape(-1,1)
                        label_reg=np.array([paddings+label_reg]).reshape(-1,1)
                    else:
                        feat=np.concatenate((feat,padding_matrix),axis=0)
                        label=np.array([label+paddings]).reshape(-1,1)
                        label_reg=np.array([label_reg+paddings]).reshape(-1,1)

                    if step==time_step-1:
                    
                        feat_arr=feat
                        label_arr=label
                        label_reg_arr=label_reg
                        # print(f'step0 feat_arr: {feat_arr.shape}\n'
                        # f'step0 label_arr: {label_arr.shape}\n'
                        # f'step0 label_reg_arr: {label_reg_arr.shape}\n')
                    else:
                
                        feat_arr=np.vstack((feat_arr,feat))
                        
                        label_arr=np.vstack((label_arr,label))
                        
                        label_reg_arr=np.vstack((label_reg_arr,label_reg))
            else:
                cnt_null+=1
                pass
        if na_fill_type!='mask':
            if i==0:

                    feat_arr_all=feat_arr
                    label_arr_all=label_arr
                    label_reg_arr_all=label_reg_arr
            else:
                if len(feat_arr)!=0:
                    
                    label_arr_all=np.vstack((label_arr_all,label_arr))
                    label_reg_arr_all=np.vstack((label_reg_arr_all,label_reg_arr))
                    feat_arr_all=np.vstack((feat_arr_all,feat_arr))

       
    if na_fill_type!='mask':
        feat_lens=None
        
        feat_arr_all=feat_arr_all.reshape(int(feat_arr_all.shape[0]/seq_size),seq_size,feat_col_len)
        if model_transform:
            print(f'when model_transform=={model_transform} you are now converting data for transformer model')
            label_arr_all=label_arr_all.reshape(label_arr_all.shape[0]//seq_size,seq_size,1).squeeze(2) #remove the dimension that only has 1 dimension entry there, e.g. 1*3*1, squeeze on axis=2, meaning remove the dimension on axis=2
            label_reg_arr_all=label_reg_arr_all.reshape(label_reg_arr_all.shape[0]//seq_size,seq_size,1).squeeze(2)
        else:
            print(f'when model_transform=={model_transform} you are not converting data for transformer model')
            label_arr_all=label_arr_all[::seq_size].squeeze(1)
            label_reg_arr_all=label_reg_arr_all[::seq_size].squeeze(1)
    

        print(f'final: after all the students: feat_arr: {feat_arr_all.shape}\n' #(400,16)
            f'final: after all the students:label_arr.shape:  {label_arr_all.shape}\n' #(400,1)
            f'final: after all the students: label_reg_arr shape {label_reg_arr_all.shape}\n')#(400,1)
    else:
        print(f'...we have done {cnt} loops and {cnt_null} failed loops...')
        feat_lens=[len(x) for x in feat_arr_all_mask]
        print(f'...feat_arr_all/label_arr_all/label_reg_arr_all length:{len(feat_arr_all_mask)}/{len(label_arr_all)}/{len(label_reg_arr_all)}/{len(feat_arr_all_full)}...\n')
        
        
        # display(processed_data.head())
        # print(f'..feat_full_df shape: {feat_full_df.shape}..')#(5530011, 11)
        feat_arr_all=pad_sequence(feat_arr_all_mask,batch_first=True)
        
        label_arr_all=pad_sequence(label_arr_all,batch_first=True)
        label_reg_arr_all=pad_sequence(label_reg_arr_all,batch_first=True)
        max_label_lens=feat_arr_all.size()[1]
        label_arr_all=label_arr_all.reshape(-1,1)
        label_reg_arr_all=label_reg_arr_all.reshape(-1,1)
        label_arr_all=label_arr_all[::max_label_lens].squeeze(1)
        label_reg_arr_all=label_reg_arr_all[::max_label_lens].squeeze(1)
        # label_arr_all=(feat_arr_full,label_arr_all)
    feat_arr_all_full=np.array(feat_arr_all_full)
    processed_data=pd.DataFrame(feat_arr_all_full,columns=orig_cols)
    processed_data['event_time']=processed_data['event_time'].replace(map_dict)
    processed_data=processed_data.drop('score_rate',axis=1)
    print(f'...feat_arr_all_full shape: {processed_data.shape}...')
    if save_flag:
        if save_dir:
            save_path=f'{save_dir}/{course_name}_{num_stu}stu_seqL{seq_size}_{feat_col_len}features_{marker}_{na_fill_type}_transformer{model_transform}.npz'
            print(f'....saving to {save_path}.... ')
            np.savez(save_path,feature=feat_arr_all,target_id=label_arr_all, proficiency=label_reg_arr_all,feat_lens=feat_lens)
        else:
            save_path=f'TRANS_DATA/{course_name}_{num_stu}stu_seqL{seq_size}_{feat_col_len}features_{marker}_{na_fill_type}_transformer{model_transform}.npz'
            print(f'....saving to {save_path}.... ')
            np.savez(save_path,feature=feat_arr_all,target_id=label_arr_all, proficiency=label_reg_arr_all,feat_lens=feat_lens)
    print(f'average time step per student:{np.mean(time_step_list)}')

    is_predict=False
    data_loader=scale_create_dataloader(is_predict,na_fill_type,feat_lens,feat_arr_all,label_arr_all, label_reg_arr_all,  batch_size)
    return feat_arr_all,label_arr_all, label_reg_arr_all,feat_lens,processed_data,data_loader



def generate_indiv_loader(batch_size,save_dir,data,na_fill_type,seq_size,shift_step, test_feature_list,course_name,num_stu,marker,model_transform,save_flag):

    if test_feature_list:
        feat_cols=test_feature_list
    feature='event_time'
    orig_cols, map_dict,_=encode_feature_v2(data,feature)
    total_stu_num=data['student_id'].nunique()
    print(f'original data shape:{data.shape} with {total_stu_num} total students')

# create npz for train/val/test
    print('\n ...generate train data...')
    _=generate_npz(batch_size,map_dict,orig_cols,save_dir,data,na_fill_type,seq_size,shift_step,  feat_cols,course_name,num_stu,marker,model_transform,save_flag)
    data_loader=_[-1]
    processed_data=_[-2]
    return data_loader,processed_data



def encode_feature(data,save_dir,feat_name,marker):
    """
    this func is a handy helper to label encode one feature
    """
    le=LabelEncoder()
    # data=data.fillna(0)
    data[feat_name]=le.fit_transform(data[feat_name].astype(str))

    feat_uni=list(data[feat_name].unique())
    old_feat_uni=le.inverse_transform(feat_uni)
    map=pd.DataFrame({f'original_{feat_name}':old_feat_uni,f'new_{feat_name}':feat_uni})
    print(map.shape)
    map.to_csv(f'{save_dir}/{marker}_{feat_name}_enc.csv',index=False)
    map_df=pd.DataFrame({'feat_name':[feat_name],'old_label':[old_feat_uni],'new_label':[feat_uni]})

    display(map.head())
    display(data.head())
    return data,map_df

def combine_feat_covar_npz(reshape_3d,feat_cols,covar_feat_cols,na_fill_type,seq_size,data_file,course_name,num_stu,marker,save_flag):
    """
    the root dir for this func should be under Longitudinal-VAE, this is to process both feature and covariate info and create a npz file
    """
    df=pd.read_csv(data_file)
    df=df.fillna(0)
    print(f'data has total shape:{df.shape}')
    flag=f'{marker}_outcome_labels'
    save_dir='data/k12/pre_train/label_maps_wo_nas'
    os.makedirs(save_dir,exist_ok=True)
    enc_list=['event_time','SPECIAL_ED','GRADE_LEVEL','SCHOOL_ID','FREE_REDUCED_LUNCH','GIFTED_TALENTED', 'student_id']
    map_dat=pd.DataFrame()
    for feat_name in enc_list:
        df,map_df=encode_feature(df,save_dir,feat_name,flag)
        map_dat=map_dat.append(map_df)
    map_dat.to_csv(f'{save_dir}/all_label_coding_keys.csv',index=False)
    print(f'df columns:{df.columns}')
    feat_col_len,covar_feat_len=len(feat_cols),len(covar_feat_cols)
    is_covar=False
    save_path=f'data/k12/split_data/{course_name}_{num_stu}stu_seqL{seq_size}_{na_fill_type}_{marker}'
    feat_arr,mask_arr,fix_len_df=generate_npz_scale_lvae(save_path,reshape_3d,is_covar,feat_cols,df,na_fill_type,seq_size)
    is_covar=True
    covar_arr,covar_mask_arr,fix_len_df=generate_npz_scale_lvae(save_path,reshape_3d,is_covar,covar_feat_cols,df,na_fill_type,seq_size)
    map_dict={}
    for i,name in enumerate(map_dat['feat_name']):
        old_label=map_dat.iloc[i]['old_label']
        new_label=map_dat.iloc[i]['new_label']

        map_dict={new_label[i]:old_label[i] for i in range(len(new_label))}
        fix_len_df[name]=fix_len_df[name].replace(map_dict)
    print(f'..saving the covar data before NPZ to{save_path}_is_Covar{is_covar}_{covar_feat_len}features_beforeNPZ.csv..')
    fix_len_df.to_csv(f'{save_path}_is_Covar{is_covar}_{covar_feat_len}features_beforeNPZ.csv',index=False)
    if save_flag:
        
        save_path=f'data/k12/split_data/{course_name}_{num_stu}stu_seqL{seq_size}_{feat_col_len}features_{covar_feat_len}covar_features_{marker}_{na_fill_type}.npz'
        print(f'....saving to {save_path}.... ')
        
        np.savez(save_path,data_readings=feat_arr,outcome_attrib=covar_arr,data_mask=mask_arr,outcome_mask=covar_mask_arr)
    return feat_arr,covar_arr,mask_arr,covar_mask_arr




def generate_npz_scale_lvae(save_path,reshape_3d,is_covar,feat_cols,df,na_fill_type,seq_size):

    df=df.sort_values(by=['student_id','event_time','seq_number'])
    stu_uni=list(df['student_id'].unique())
    print(f'there are {len(stu_uni)} unique students')
    cnt=0
    cnt_null=0
    time_step_list=[]
    feat_col_len=len(feat_cols)
    feat_arr_all=np.zeros([1,feat_col_len],dtype=float)
    mask_arr_all=np.zeros([1,feat_col_len],dtype=float)
    feat_arr_all_mask=[]
    for i, student_id in enumerate(stu_uni):
        student_df=df[df['student_id']==student_id]
        time_step=student_df.shape[0]
        time_step_list.append(time_step)
        
        if i%500==0:print(i, student_id, time_step)
        data=student_df
        data_len=len(data)
        if data_len!=0:
            if na_fill_type=='mask':
                feat=torch.from_numpy(data[feat_cols].values)
                feat_arr_all_mask.append(feat)
    
                cnt+=1
            else:
                if data_len<seq_size:
                    pad_counts=seq_size-data_len
                    mask_one_counts=data_len
                    if na_fill_type=='bfill':
                        pad_arr=np.array([data.iloc[0][feat_cols]]) # the 9 features are from 7th to 16th column
    
                    elif na_fill_type=='ffill':
                        pad_arr=np.array([data.iloc[data_len-1][feat_cols]])
                
                    else:
                        pad_arr=np.zeros([1,feat_col_len])
                        # ,dtype=int
            
                else:
                    data=data[-seq_size:]
                    pad_counts = 0
                    mask_one_counts=data.shape[0]
                    pad_arr=np.zeros([1,feat_col_len])
    
                
                padding_matrix=np.repeat(pad_arr,repeats=pad_counts,axis=0)
                feat=data[feat_cols].values
                mask_zero=np.zeros([1,feat_col_len])
                mask_padding=np.repeat(mask_zero,repeats=pad_counts,axis=0)
                mask_one=np.ones([1,feat_col_len])
                mask_matrix=np.repeat(mask_one,repeats=mask_one_counts,axis=0)
            
            
                if na_fill_type=='bfill':
                    feat=np.concatenate((padding_matrix, feat),axis=0)
                    mask_matrix=np.concatenate((mask_padding,mask_matrix),axis=0) 
                    
                else:
                    feat=np.concatenate((feat,padding_matrix),axis=0)
                    mask_matrix=np.concatenate((mask_matrix,mask_padding),axis=0)
                # feat_arr=np.vstack((feat_arr,feat))
        else:
            cnt_null+=1
            pass
        if na_fill_type!='mask':
            if i==0:
                feat_arr_all=feat
                mask_arr_all=mask_matrix
                print(f'{i}th student feat_arr_all:{feat_arr_all}\n '
                f'{i}th student mask_arr_all:{mask_arr_all}\n ')
            else:
                if len(feat)!=0:
                    feat_arr_all=np.vstack((feat_arr_all,feat))
                    mask_arr_all=np.vstack((mask_arr_all,mask_matrix))

    if na_fill_type!='mask':
        fix_len_df=pd.DataFrame(feat_arr_all,columns=[feat_cols])
        
        scaler=MinMaxScaler()
        if is_covar:
            feat_arr_all=feat_arr_all
        else:
            fix_len_df.to_csv(f'{save_path}_is_Covar{is_covar}_{feat_col_len}features_beforeNPZ.csv',index=False)
            feat_arr_all=scaler.fit_transform(feat_arr_all)
        feat_lens=None

        if reshape_3d:
            feat_arr_all=feat_arr_all.reshape(int(feat_arr_all.shape[0]/seq_size),seq_size,feat_col_len)
            
            mask_arr_all=mask_arr_all.reshape(int(mask_arr_all.shape[0]/seq_size),seq_size,feat_col_len)
            
        print(f'final: after all the students: feat_arr: {feat_arr_all.shape}\n') #(400,16)
        print(f'final: after all the students: mask_arr: {mask_arr_all.shape}\n') #(400,16)
    else:
        fix_len_df=None
        if is_covar:
            feat_arr_all_mask=feat_arr_all_mask
            
        else:
            feat_arr_all_mask=scaler.fit_transform(feat_arr_all_mask)

        print(f'...we have done {cnt} loops and {cnt_null} failed loops...')
        feat_lens=[len(x) for x in feat_arr_all_mask]
        print(f'...feat_arr_all length:{len(feat_arr_all_mask)}...\n')
        feat_arr_all=pad_sequence(feat_arr_all_mask,batch_first=True)
        mask_arr_all=None
    
    print(f'average time step per student:{np.mean(time_step_list)}')

    
    return feat_arr_all,mask_arr_all,fix_len_df
