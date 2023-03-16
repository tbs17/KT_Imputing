#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os, re, json, csv,time,math,argparse, copy
from datetime import datetime,date
from dateutil.relativedelta import relativedelta
import warnings
from scripts.constant import *
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
# ---below is used to get the attribution score and graph it-----



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

def get_row_num_list_red(data,ratios):
    
    total_stu_num=data['student_id'].nunique()
    stu_num_list=[int(total_stu_num*x) for x in ratios]
    total_stu=data['student_id'].unique()

    row_num_list=[]
    prev=0
    for i in stu_num_list:
    
        # print(step)
        cur=prev+i
        
        stu_perc=total_stu[prev:cur]
        print(f'{prev}:{cur} idx students')
        prev=cur
        stu_df=data[data['student_id'].isin(stu_perc)]
        row_num=stu_df.shape[0]
        print(f'.. has {row_num} rows')
        row_num_list.append(row_num)
    print(row_num_list)
    return row_num_list

def get_perc_data(data_path,ratio):
    data=pd.read_csv(data_path)
    total_stu_num=data['student_id'].nunique()

    stu_num=int(total_stu_num*ratio)
    print(f'total students are {total_stu_num}\n'
    f'taking {ratio*100} % to get {stu_num} students')
    data_part=data[data['student_id'].isin(data['student_id'].unique()[:stu_num])]
    print(data_part.shape)
    print(data_part['student_id'].nunique())
    save_dir='/'.join(data_path.split('/')[:-1])
    save_path=data_path.split('/')[-1].split('.csv')[0]
 
    print(f'..saving to {save_dir}/{save_path}_{stu_num}_ratio{ratio}.csv..')
    data_part.to_csv(f'{save_dir}/{save_path}_{stu_num}_ratio{ratio}.csv',index=False)
    return data_part

def get_mean_seq(data_path):
    data=pd.read_csv(data_path)
    data['seq']=data.index
    print(data.shape)
    display(data.head())

    data_pv=pd.pivot_table(data,index='student_id',values='seq',aggfunc=pd.Series.nunique).reset_index()
    data_pv.columns=['student_id','cnt']
    avg_seq_len=data_pv['cnt'].mean()
    print(f'avg seq length:{avg_seq_len}')
    return avg_seq_len


def combine_process(data_dir,aux_data,marker):
    """
    this func is to combine the train,val,test data and do feature engineering and create auxilary data and masks
    """
    split_type_list=['train','val','test']
    df_full=pd.DataFrame()
    for split_type in split_type_list:
        data_path=f'{data_dir}/{marker}_full_with_nas_{split_type}.csv'
        data=pd.read_csv(data_path)
        print(f'{marker} {split_type} shape:{data.shape} ')
        df_full=df_full.append(data)

    print(df_full.shape)

    df_full.to_csv(f'{data_dir}/{marker}_full_with_nas.csv',index=False)
    df_full=df_full.rename(columns={'event_time_first':'event_time'})
    data=df_full
    keep_vars=[ 'event_time','attempt_time','student_id',  'bktcount','proficiency', 'seq_number',
            'assessment_duration','count','score', 'max_score', 
        'quest_ref', 'item_ref',  'usmo_id','question_type' ]
    marker=f'{marker}_eng_w_nas'
    df_full_eng=clean_featEng(data,keep_vars,marker)
    display(df_full.isna().sum())
    stu_uni=df_full['student_id'].nunique()
    print(f'student number: {stu_uni}')
    df_full_eng=df_full_eng.sort_values(['student_id','event_time','seq_number'])
    df_full_eng_aux=pd.merge(df_full_eng,aux_data,how='left',on='student_id')
    print(df_full_eng_aux.shape)

    df_aux_final=df_full_eng_aux[['event_time', 'LE_SCHOOL_ID','SPECIAL_ED', 'student_id','FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL' ,'correct','score_rate']]
    print(df_aux_final.shape)

    df_aux_final.to_csv(f'data/k12/{marker}_outcome_labels.csv',index=False)
    df_aux_final_mask=np.where(df_aux_final.isna(),0,1)
    df_aux_final_mask_df=pd.DataFrame(df_aux_final_mask,columns=df_aux_final.columns)
    df_aux_final_mask_df.to_csv(f'data/k12/{marker}_aux_final_mask_df.csv',index=False)

    df_data_final=df_full_eng_aux[[ 'seq_number', 'assessment_duration', 'count',
        'quest_difficulty', 'item_difficulty', 'usmo_difficulty',
            'quest_ref_1', 'item_ref_1', 'usmo_id_1',
        'question_type_1']]
    print(df_data_final.shape)

    df_data_final.to_csv(f'data/k12/{marker}_data_readings.csv',index=False)
    df_data_final.isna().sum() #no missing values, we need the missing value version
    df_data_final_mask=np.where(df_data_final.isna(),0,1)
    df_data_final_mask_df=pd.DataFrame(df_data_final_mask,columns=df_data_final.columns)

    df_data_final_mask_df.to_csv(f'data/k12/{marker}_data_final_mask_df.csv',index=False)
    print('data_readings/data_mask/outcome/outcome_mask shape:')
    print(df_data_final.shape,df_data_final_mask_df.shape,df_aux_final.shape,df_aux_final_mask_df.shape)
    data_readings=df_data_final.values
    data_masks=df_data_final_mask_df.values
    labels=df_aux_final.values
    label_masks=df_aux_final_mask_df.values
    np.savez(f'data/k12/{marker}_data.npz',data_readings=data_readings,outcome_attrib=labels,data_mask=data_masks,outcome_mask=label_masks)
    return df_data_final,df_data_final_mask_df,df_aux_final,df_aux_final_mask_df


def clean_featEng(data,keep_vars,marker):
    """
    this func is to create features such as item/quest/usmo difficulties on the total data
    """
    if keep_vars!=None:

        data_nar=data[keep_vars]
    else:
        data_nar=data
    data_nar['correct']=np.where(data_nar['score']==data_nar['max_score'],1,0)
    # ---calculate quest difficulty--
    quest_total_stu=pd.pivot_table(data_nar, index='quest_ref',values='student_id',aggfunc=pd.Series.nunique).reset_index()
    quest_correct_stu=pd.pivot_table(data_nar, index='quest_ref',values='correct',aggfunc=np.sum).reset_index()
    quest_diff_df=pd.merge(quest_total_stu,quest_correct_stu,how='inner',on='quest_ref')
    print(f'quest_diff_df.shape: {quest_diff_df.shape}')
    quest_diff_df['quest_difficulty']=round(quest_diff_df['correct']/quest_diff_df['student_id'],4)
    data_eng=pd.merge(data_nar,quest_diff_df[['quest_ref','quest_difficulty']],how='left',on='quest_ref')
    # ---calculate item difficulty--
    item_total_stu=pd.pivot_table(data_nar, index='item_ref',values='student_id',aggfunc=pd.Series.nunique).reset_index()
    item_correct_stu=pd.pivot_table(data_nar, index='item_ref',values='correct',aggfunc=np.sum).reset_index()
    item_diff_df=pd.merge(item_total_stu,item_correct_stu,how='inner',on='item_ref')
    print(f'item_diff_df.shape: {item_diff_df.shape}')
    item_diff_df['item_difficulty']=round(item_diff_df['correct']/item_diff_df['student_id'],4)
    data_eng=pd.merge(data_eng,item_diff_df[['item_ref','item_difficulty']],how='left',on='item_ref')
    # ---calculate usmo difficulty---
    usmo_total_stu=pd.pivot_table(data_nar, index='usmo_id',values='student_id',aggfunc=pd.Series.nunique).reset_index()
    usmo_correct_stu=pd.pivot_table(data_nar, index='usmo_id',values='correct',aggfunc=np.sum).reset_index()
    usmo_diff_df=pd.merge(usmo_total_stu,usmo_correct_stu,how='inner',on='usmo_id')
    print(f'usmo_diff_df.shape: {usmo_diff_df.shape}')
    usmo_diff_df['usmo_difficulty']=round(usmo_diff_df['correct']/usmo_diff_df['student_id'],4)
    data_eng=pd.merge(data_eng,usmo_diff_df[['usmo_id','usmo_difficulty']],how='left',on='usmo_id')
    data_eng['score_rate']=round(data_eng['score']/data_eng['max_score'],6)
    feature_list=['quest_ref','item_ref','usmo_id','question_type']
    data=data_eng
    data_eng2=enc_features(data,feature_list)

    # data_eng2=data_eng2.drop('quest_ref',axis=1)
    print(f'cleaned data shape {data_eng2.shape}')
    data_eng2=data_eng2[['event_time','attempt_time', 'student_id', 'bktcount', 'proficiency','score', 'max_score',
        'seq_number', 'assessment_duration', 'count',
       'correct', 'quest_difficulty', 'item_difficulty', 'usmo_difficulty',
       'score_rate', 'quest_ref_1', 'item_ref_1', 'usmo_id_1',
       'question_type_1']]
    #    removing 'score', 'max_score', 'attempt_time',
    display(data_eng2.head())
    data_eng2.to_csv(f'../TRANS_DATA/KT_data_2019-01-01_2019-06-30_{marker}.csv',index=False)
    return data_eng2

def enc_features(data,feature_list):

    data0=copy.deepcopy(data)
    for f in feature_list:
        print(f'encoding feature:{f}')
        data0=label_encode(data0,f)
    return data0
def label_encode(data,feature):
#     data=copy.deepcopy(data0)
    lb=LabelEncoder()
    data[feature]=data[feature].astype(str)
    data[feature+'_1']=lb.fit_transform(data[feature])
    data.drop(feature,axis=1,inplace=True)
    return data

def combine_data(folder,marker):
    """
    this function is for combining all the data with missing values
    """
    df=pd.DataFrame()
    len=0
    split_type_list=['train','val','test']
    for split_type in split_type_list:
        data_dir=f'{folder}/processed/1/{split_type}'
        print(f'\n...running for {split_type} data...\n')
        for i, f in enumerate(os.listdir(data_dir)):
            data=pd.read_csv(os.path.join(data_dir,f))
            len+=data.shape[0]
            df=df.append(data)
            
            if i%100==0: 
                print(split_type,i,f,data.shape)
                df.to_csv(f'../TRANS_DATA/{marker}_full_with_nas_{split_type}.csv',index=False)
        stu_num=df['student_id'].nunique()
        avg_len=len/stu_num
        print(f'{split_type} df.shape {df.shape} with avg_len: {avg_len}')
    stu_num=df['student_id'].nunique()
    avg_len=len/stu_num
    print(f'total df.shape {df.shape} with avg_len: {avg_len}')
    display(df.head())
    return df


def create_data_retriever(data,data_name,ref_data_path):
    """
    this function is to creating data for each students with their supposed missing values
    """
    data_dir=data_name+'_retrieved'
    old_data_dir=data_name
    split_type_list=['train','val','test']
    quests_per_event_school_ref=pd.read_csv(ref_data_path)
    empty_path=pd.DataFrame()
    for split_type in split_type_list:

        data_path=f'{data_dir}/processed/1/{split_type}/'
        os.makedirs(data_path,exist_ok=True)
        old_data_path=f'{old_data_dir}/processed/1/{split_type}/'
        

        for i, f in enumerate(os.listdir(old_data_path)):
            if f.endswith('.csv'):
                stu_id=f.split('.csv')[0]
                
                stu_df=data[data['student_id']==int(stu_id)]
                # display(stu_df)
                if stu_df.shape[0]!=0:
                
                    school_id=stu_df.iloc[0]['school_id']
                    school_df=quests_per_event_school_ref[quests_per_event_school_ref['school_id']==school_id]
                    print(f'changing data in {old_data_path}/{f} for school: {school_id}')
                    print(f'student: {stu_id} shape before retrieving the missing: {stu_df.shape}')
                    stu_df_more=pd.merge(school_df,stu_df,how='left',on=['school_id','quest_id'],suffixes=['_first','_perstu'])
                    print(f' student:{stu_id} shape after retrieving the missing: {stu_df_more.shape}\n')
                    
        
                    stu_df_more.loc[~stu_df_more['event_time_perstu'].isnull(),'event_time_first']=stu_df_more['event_time_perstu']
                    stu_df_more_reorder=stu_df_more.sort_values(['event_time_first'])
                    # stu_df_more_reorder.head()
                    if stu_df_more.shape[0]==0:
                        display(stu_df)
                    
                    save_path=os.path.join(data_path,str(stu_id))
                    stu_df_more_reorder.to_csv(f'{save_path}.csv',index=False)
                else:
                    old_path=os.path.join(old_data_path,f)
                    stu_df1=pd.read_csv(old_path)
                    print(f'student {stu_id} in the old data has shape: {stu_df1.shape}')
                    empty_df=pd.DataFrame({'stu_id':[stu_id],'data_path':[old_path],
                                        'old_df_shape':[stu_df.shape],'new_df_shape':[stu_df1.shape]})
                    empty_path=empty_path.append(empty_df)
                    empty_path.to_csv(f'TRANS_DATA/empty_data_from_{data_name}.csv',index=False)
                    pass
            else:
                pass
        # for i, sch in enumerate(data['school_id'].unique()):
        #     if i>=0:
        #         quests_per_event_school=quests_per_event_school[quests_per_event_school['school_id']==sch]
        #         school_df=data[data['school_id']==sch]
        #         for j, stu in enumerate(school_df['student_id'].unique()):
        #             if j>=0:
                        
