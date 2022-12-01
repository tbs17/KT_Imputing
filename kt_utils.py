#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os, re, json, csv,time,math,argparse
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
                        


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_sig_mv(data,marker):
    """
    this func is to generate significance for the trained results
    
    """
    from scipy import stats as st

    df=pd.DataFrame()
    if 'domain' in data.columns:
        
        domain_col='domain' 
    else:
        domain_col='data'
    for domain in data[domain_col].unique():
        domain_df=data[data[domain_col]==domain]
        print(f'>{domain}<')
        base_domain_df=domain_df[domain_df['lag_type']=='specific']
        
        rest_domain_df=domain_df[domain_df['lag_type']!='specific']
        for model in rest_domain_df['model_name'].unique():
            print(f'>>{model}<<')
            model_domain_df=rest_domain_df[rest_domain_df['model_name']==model]
            met1,met2,met3='MSE','RMSE','L1'
            spec_met1=model_domain_df[met1].to_numpy()
            spec_met2=model_domain_df[met2].to_numpy()
            spec_met3=model_domain_df[met3].to_numpy()
            base_met1=base_domain_df[met1].to_numpy()
            base_met2=base_domain_df[met2].to_numpy()
            base_met3=base_domain_df[met3].to_numpy()

            met1_p_value,met1_flag,=bartlett_ttest(spec_met1,base_met1)
            met2_p_value,met2_flag,=bartlett_ttest(spec_met2,base_met2)
            met3_p_value,met3_flag=bartlett_ttest(spec_met3,base_met3)
            dat=pd.DataFrame({'domain':[domain],
                                     'model_name':[model],
                                     f'{met1}_sig':[met1_p_value],
                                      f'{met1}_flag':[met1_flag],
                                     f'{met2}_sig':[met2_p_value],
                              f'{met2}_flag':[met2_flag],
                                     f'{met3}_sig':[met3_p_value],
                             f'{met3}_flag':[met3_flag],})
#             display(dat)
            df=df.append(dat)
    display(df.head())
    df.to_csv(f'TRANS_MODEL/COMBINE/{marker}_sig_result.csv',index=False)
    return df

def check_sig(data,marker):
    from scipy import stats as st

    df=pd.DataFrame()

    for domain in data['domain'].unique():
        domain_df=data[data['domain']==domain]
        print(f'>{domain}<')
        for model in domain_df['model_name'].unique():
            print(f'>>{model}<<')
            model_domain_df=domain_df[domain_df['model_name']==model]
#             display(model_domain_df)
            spec_acc=model_domain_df.loc[model_domain_df['lag_type']=='specific','ACC'].to_numpy()
#             print(spec_acc)
            spec_auc=model_domain_df.loc[model_domain_df['lag_type']=='specific','AUC'].to_numpy()
            spec_f1=model_domain_df.loc[model_domain_df['lag_type']=='specific','F1'].to_numpy()
            full_acc=model_domain_df.loc[model_domain_df['lag_type']=='full','ACC'].to_numpy()
            full_auc=model_domain_df.loc[model_domain_df['lag_type']=='full','AUC'].to_numpy()
            full_f1=model_domain_df.loc[model_domain_df['lag_type']=='full','F1'].to_numpy()
            acc_p_value,acc_flag,=bartlett_ttest(spec_acc,full_acc)
            auc_p_value,auc_flag,=bartlett_ttest(spec_auc,full_auc)
            f1_p_value,f1_flag=bartlett_ttest(spec_f1,full_f1)
            dat=pd.DataFrame({'domain':[domain],
                                     'model_name':[model],
                                     'acc_sig':[acc_p_value],
                                      'acc_flag':[acc_flag],
                                     'auc_sig':[auc_p_value],
                              'auc_flag':[auc_flag],
                                     'f1_sig':[f1_p_value],
                             'f1_flag':[f1_flag],})
#             display(dat)
            df=df.append(dat)
    display(df.head())
    df.to_csv(f'TRANS_MODEL/COMBINE/{marker}_sig_result.csv',index=False)
    return df


def bartlett_ttest(arr1,arr2):
    from scipy import stats as st
    stat,p=st.bartlett(arr1,arr2)
    if p<0.05:
        print(f'bartlett p is {round(p,3)},the two arrays do not have equal variance')
        res=st.ttest_ind(arr1,arr2,equal_var=False)
#         print(res)
    else:
        print(f'bartlett p is {round(p,3)},the two arrays do have equal variance')
        res=st.ttest_ind(arr1,arr2,equal_var=True)
    p_value=res[1]
    if p_value<0.05:
        flag='sig'
    else:
        flag='insig'
#     print(p_value)
    return p_value,flag
    
def get_shift_cat_v2(data,cadence,max_seq):
    num_cat=max_seq//cadence
    if num_cat==3:
#         cadence=total_seq//3
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
               data['shift']>=cadence*2]
        choices=[f'<={cadence}',f'{cadence}_{cadence*2}',f'>={cadence*2}']
        # choices2=
    elif num_cat==4:
#         cadence=total_seq//4
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
             (data['shift']>=cadence*2) &(data['shift']<cadence*3),
               data['shift']>=cadence*3]
        choices=[f'<={cadence}',f'{cadence}_{cadence*2}',f'{cadence*2}_{cadence*3}',f'>={cadence*3}']
    else:
#         cadence=total_seq//5
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
             (data['shift']>=cadence*2) &(data['shift']<cadence*3),
                    (data['shift']>=cadence*3) &(data['shift']<cadence*4),
               data['shift']>=cadence*4]
        choices=[f'<={cadence}',f'{cadence}_{cadence*2}',f'{cadence*2}_{cadence*3}',f'{cadence*3}_{cadence*4}',f'>={cadence*4}']

    data['shift_cat']=np.select(conditions, choices)
    display(data['shift_cat'].value_counts())
    return data


def check_sig(data,marker):
    from scipy import stats as st

    df=pd.DataFrame()

    for domain in data['domain'].unique():
        domain_df=data[data['domain']==domain]
        print(f'>{domain}<')
        for model in domain_df['model_name'].unique():
            print(f'>>{model}<<')
            model_domain_df=domain_df[domain_df['model_name']==model]
#             display(model_domain_df)
            spec_acc=model_domain_df.loc[model_domain_df['lag_type']=='specific','ACC'].to_numpy()
#             print(spec_acc)
            spec_auc=model_domain_df.loc[model_domain_df['lag_type']=='specific','AUC'].to_numpy()
            spec_f1=model_domain_df.loc[model_domain_df['lag_type']=='specific','F1'].to_numpy()
            full_acc=model_domain_df.loc[model_domain_df['lag_type']=='full','ACC'].to_numpy()
            full_auc=model_domain_df.loc[model_domain_df['lag_type']=='full','AUC'].to_numpy()
            full_f1=model_domain_df.loc[model_domain_df['lag_type']=='full','F1'].to_numpy()
            acc_p_value,acc_flag,=bartlett_ttest(spec_acc,full_acc)
            auc_p_value,auc_flag,=bartlett_ttest(spec_auc,full_auc)
            f1_p_value,f1_flag=bartlett_ttest(spec_f1,full_f1)
            dat=pd.DataFrame({'domain':[domain],
                                     'model_name':[model],
                                     'acc_sig':[acc_p_value],
                                      'acc_flag':[acc_flag],
                                     'auc_sig':[auc_p_value],
                              'auc_flag':[auc_flag],
                                     'f1_sig':[f1_p_value],
                             'f1_flag':[f1_flag],})
#             display(dat)
            df=df.append(dat)
    display(df.head())
    df.to_csv(f'TRANS_MODEL/COMBINE/{marker}_sig_result.csv',index=False)
    return df


def bartlett_ttest(arr1,arr2):
    from scipy import stats as st
    stat,p=st.bartlett(arr1,arr2)
    if p<0.05:
        print(f'bartlett p is {round(p,3)},the two arrays do not have equal variance')
        res=st.ttest_ind(arr1,arr2,equal_var=False)
#         print(res)
    else:
        print(f'bartlett p is {round(p,3)},the two arrays do have equal variance')
        res=st.ttest_ind(arr1,arr2,equal_var=True)
    p_value=res[1]
    if p_value<0.05:
        flag='sig'
    else:
        flag='insig'
#     print(p_value)
    return p_value,flag
    

def melt_all_data(folder,marker):
    df=pd.DataFrame()
    for i, f in enumerate(os.listdir(folder)):
        if f.endswith('csv'):
            print(i,f)
            
            dat=pd.read_csv(os.path.join(folder,f))
            print(dat.shape)
            dat_m=pd.melt(dat,id_vars=['model_name','shift'],value_vars=list(dat.columns)[4:7],var_name='Metric')
            data_name=dat.iloc[0,3].split('/')[2].split('_')[0]
            print(data_name)
            dat_m['data']=data_name
            
            display(dat_m.head())
            df=df.append(dat_m)
    
    out_dir='TRANS_MODEL/COMBINE'
    os.makedirs(out_dir,exist_ok=True)
    df['model_name']=df['model_name'].str.replace('DKT_BKT_SV','DKT')
    df=df[~df['Metric'].isin(['MSE','F1'])]
    display(df.sample(10))
#     df=df.drop_duplicates(['model_name','shift'])
    df.to_csv(f'{out_dir}/6_data_dkt_npa_total_{marker}.csv',index=False)
    return df
            

def combin_dkt_npa(params):
    path1=params['path1']
    path2=params['path2']
    marker=params['marker']
    out_dir=params['out_dir']
    check_cols=params['check_cols']
    domain_exist=params['domain_exist']
    need_change=params['need_change']
    p1_data=pd.read_csv(path1)
    p2_data=pd.read_csv(path2)
    total_data=pd.concat([p1_data,p2_data])
    if need_change:
        total_data=total_data.rename(columns={'MSE':'ACC','RMSE':'AUC','L1':'F1'})
    if domain_exist:
        total_data=total_data.drop(['domain'],axis=1)
    
    total_data=total_data.drop_duplicates(check_cols)
    total_data.to_csv(f'{out_dir}/{marker}.csv',index=False)
    print(total_data.shape)
    display(total_data.head())
    return total_data
    
def get_shift_cat(data,num_cat,total_seq):
    if num_cat==3:
        cadence=total_seq//3
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
               data['shift']>=cadence*2]
        choices=[f'shift_le_{cadence}',f'shift_{cadence}_{cadence*2}',f'shift_ge_{cadence*2}']

    elif num_cat==4:
        cadence=total_seq//4
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
             (data['shift']>=cadence*2) &(data['shift']<cadence*3),
               data['shift']>=cadence*3]
        choices=[f'shift_le_{cadence}',f'shift_{cadence}_{cadence*2}',f'shift_{cadence*2}_{cadence*3}',f'shift_ge_{cadence*3}']
    else:
        cadence=total_seq//5
        conditions=[data['shift']<cadence,
               (data['shift']>=cadence) &(data['shift']<cadence*2),
             (data['shift']>=cadence*2) &(data['shift']<cadence*3),
                    (data['shift']>=cadence*3) &(data['shift']<cadence*4),
               data['shift']>=cadence*4]
        choices=[f'shift_le_{cadence}',f'shift_{cadence}_{cadence*2}',f'shift_{cadence*2}_{cadence*3}',f'shift_{cadence*3}_{cadence*4}',f'shift_ge_{cadence*4}']
    data['shift_cat']=np.select(conditions, choices)
    display(data['shift_cat'].value_counts())
    return data

def split_sv_result(file_path):
    dat=pd.read_csv(file_path)
    path=file_path.split('.csv')[0]
    dat_dkt=dat[dat['model_name']=='DKT']
    dat_dkt.to_csv(f'{path}_dkt.csv',index=False)
    print(dat_dkt.shape)
    display(dat_dkt.head())
    
    dat_npa=dat[dat['model_name']=='NPA']
    dat_npa.to_csv(f'{path}_npa.csv',index=False)
    print(dat_npa.shape)
    display(dat_npa.head())
    return dat_dkt,dat_npa



def count_seqL(dataset_name):
    split_type_list=['train','val','test']
    avg_all=[]
    for split_type in split_type_list:
        data_dir=f'../{dataset_name}/processed/1/{split_type}'
        seq_list=[]
        
        for i, f in enumerate(os.listdir(data_dir)):
            dat=pd.read_csv(os.path.join(data_dir,f))
            if i%5000==0:
                print(dat.shape[0])
            seq_list.append(dat.shape[0])
        avg_len=np.mean(seq_list)#82 on test, 77 on train data
        print(f'the avg seq length of {dataset_name} {split_type} data:{avg_len}')
        avg_all.append(avg_len)
    avg_total=np.mean(avg_all)
    print(f'the total avg seq length:{avg_total}')
    
    return avg_total
#     train/val/test:121/122/122


def combine_target_folders(data_dir,target_folder,marker):
    now=datetime.now()
    print(now)
    start=time.time()
    dat=pd.DataFrame()
    for i,folder in enumerate(os.listdir(data_dir)):
        
        if folder!='COMBINE' and target_folder in folder:
            sub_folder=os.path.join(data_dir,folder)
            print(f'...checking folder {sub_folder}')
            for j, f in enumerate(os.listdir(sub_folder)):
                if f.endswith('.csv'):
                    df=pd.read_csv(os.path.join(sub_folder,f))
                    print(i,f,df.shape)
                    print('\n')
                    dat=dat.append(df)     
    display(dat.tail())
    print(dat.shape)
    out_dir=os.path.join(data_dir,'COMBINE')
    os.makedirs(out_dir,exist_ok=True)
    dat.to_csv(os.path.join(out_dir,f'{marker}.csv'),index=False)
    end=time.time()
    m,s =divmod((end-start),60)
    now=datetime.now()
    print(f'it took {m} min and {round(s,2)} sec.\nfinish at: {now}')
    return dat


def combine_all_folders(data_dir,marker):
#     FOR EC2 DATA COMBINE
    now=datetime.now()
    print(now)
    dat=pd.DataFrame()
    for i,folder in enumerate(os.listdir(data_dir)):
        print(f'...checking folder {folder}')
        if folder!='COMBINE':
            sub_folder=os.path.join(data_dir,folder)
            stem=folder.split('_')[0]
            for j, f in enumerate(os.listdir(sub_folder)):
                if f.endswith('.csv'):
                    df=pd.read_csv(os.path.join(sub_folder,f))
                    df=df.rename(columns={'MSE':'ACC','RMSE':'AUC','L1':'F1'})
                    df['data_name']=stem
                    print(i,f,df.shape)
                    dat=dat.append(df)

                
    display(dat.tail())
    print(dat.shape)
    out_dir=os.path.join(data_dir,'COMBINE')
    os.makedirs(out_dir,exist_ok=True)
    dat.to_csv(os.path.join(out_dir,f'{marker}.csv'),index=False)
    return dat

def get_best_model_path(file_path,feat_type):
    dat=pd.read_csv(file_path)
    dat=dat.sort_values('RMSE').reset_index()
    # display(dat.head())
    best_model_path=dat.loc[0,'ckpt']
    if feat_type=='multi':
        best_index=dat.loc[0,'combo_index']
    else:
        best_index=0
    print(f'best model path:\n'
    f'{best_model_path}')
    return best_model_path,best_index

def obtain_op_model(data_dir,check_cols,threshold,marker):
    if data_dir.endswith('.csv'):
        dat=pd.read_csv(data_dir)
        folder=data_dir.split('.csv')[0]
        os.makedirs(folder,exist_ok=True)
        out_dir=f'{folder}/COMBINE'
    else:
        dat=combine_time_feat_data(data_dir,check_cols,marker)
        out_dir=f'{data_dir}/COMBINE'
    if 'AUC' not in dat.columns:
        dat=dat.rename(columns={'MSE':'ACC','RMSE':'AUC','L1':'F1'})
    dat=dat.drop_duplicates(check_cols)
    data=dat
    
    df,dat_valid=get_valid_perf_clf(data,threshold,out_dir,f'{marker}_outperf')
    return df,dat_valid

def get_valid_perf_clf(data,threshold,out_dir,marker):
    from ast import literal_eval
    
    data=data.sort_values(['shift'],ascending=False)
    if threshold==None:
        threshold=data.iloc[0,5]
    print(f'threshold is: {round(threshold,5)}')
    data_valid=data[data['AUC']>threshold]

    print(f'there are {data_valid.shape[0]} outperform the threshold')
    # data_valid['feat_len']=data_valid['features'].apply(lambda x:len(literal_eval(x)))
    data_valid['%more']=round((data_valid['AUC']/threshold-1)*100,2)
    
    conditions=[data_valid['%more']<10,
                (data_valid['%more']>=10)&(data_valid['%more']<15),
                (data_valid['%more']>=15)&(data_valid['%more']<20),
                (data_valid['%more']>=20)&(data_valid['%more']<25),
                data_valid['%more']>=25]
    choices=['perc_lt_10','perc_10-15','perc_15-20','perc_20-25','perc_ge_25']
    data_valid['increase_cat']=np.select(conditions,choices)
    print(data_valid['increase_cat'].value_counts(dropna=False))
    data_name=out_dir.split('/')[2].split('_')[0]
    data_valid['data']=data_name
    data_valid=data_valid.sort_values('%more',ascending=False)
    print('...descending order by %more...')
    display(data_valid.head())
    os.makedirs(out_dir,exist_ok=True)
    data_valid.to_csv(f'{out_dir}/{marker}.csv',index=False)
    return data,data_valid

def get_valid_opm(data_dir,check_cols,marker,threshold):
    data_dir=f'{data_dir}/MODEL_RUN'
    
    df=combine_time_feat_data(data_dir,check_cols,marker)#1982


    data=df

    out_dir=f'{data_dir}/COMBINE'
    marker=f'{marker}_total_dedup'
    df_dedup, df_gt_30,df_lt_30=check_anomaly(data,out_dir,marker)

    data=df_dedup
    out_dir=f'{data_dir}/COMBINE'
    marker=f'{marker}_outperf'

    df_valid=get_valid_perf(data,threshold,out_dir,marker)
    return df_valid
def get_valid_perf(data,threshold,out_dir,marker):
    from ast import literal_eval
    data_valid=data[data['RMSE']<threshold]

    print(f'there are {data_valid.shape[0]} outperform the threshold')
    # data_valid['feat_len']=data_valid['features'].apply(lambda x:len(literal_eval(x)))
    data_valid['%less']=round((1-data_valid['RMSE']/threshold)*100,2)
    
    conditions=[data_valid['%less']<10,
                (data_valid['%less']>=10)&(data_valid['%less']<15),
                (data_valid['%less']>=15)&(data_valid['%less']<20),
                (data_valid['%less']>=20)&(data_valid['%less']<25),
                data_valid['%less']>=25]
    choices=['perc_lt_10','perc_10-15','perc_15-20','perc_20-25','perc_ge_25']
    data_valid['increase_cat']=np.select(conditions,choices)
    print(data_valid['increase_cat'].value_counts(dropna=False))
    data_name=out_dir.split('/')[2].split('_')[0]
    data_valid['data']=data_name
    display(data_valid.head())
    data_valid.to_csv(f'{out_dir}/{marker}.csv',index=False)
    return data_valid

def get_label_map(data,label,marker):
    le=LabelEncoder()
    data[label+'_1']=le.fit_transform(data[label])
    encoded_label_list=data[label+'_1'].unique()
    ori_quest_ref=le.inverse_transform(encoded_label_list)
    label_map=pd.DataFrame({'original_label':ori_quest_ref,
                            'encoded_label':encoded_label_list})
    label_map.to_csv(f'TRANS_DATA/{marker}_original_encoded_label_map.csv',index=False)
    print(f'there are {label_map.shape[0]} unique {label}')
    display(label_map.head())
    return label_map

def get_top_models(data,ori_data,N):
    
    df=pd.DataFrame()
    for m in data['model_name'].unique():

        data_s=ori_data[ori_data['model_name']==m].sort_values(['%less'],ascending=False).head(N)
        df=df.append(data_s)
    display(df.head())
    print(df.shape)
    return df

def find_cf_feat_set(ori_data,N):
    ori_data_num_models=pd.pivot_table(ori_data,index=['features'],values='model_name',aggfunc=pd.Series.nunique).reset_index()
    ori_data_num_models_eq2=ori_data_num_models[ori_data_num_models['model_name']==2]
    print(f'there are {ori_data_num_models_eq2.shape[0]} feature sets that appeared in both NNs')

    ori_data_both_models=ori_data[ori_data['features'].isin(ori_data_num_models_eq2['features'])]
    print(f'there are {ori_data_both_models.shape[0]} models that have the feature sets across NNs')

    feat_freq_by_feat_model_same_data=pd.pivot_table(ori_data_both_models,index=['features','model_name'],values='seq',aggfunc=pd.Series.nunique).reset_index()
    feat_freq_by_feat_model_same_data.sort_values('seq',ascending=False)
    feat_freq_by_feat_model_same_data_gt2=feat_freq_by_feat_model_same_data[feat_freq_by_feat_model_same_data['seq']>=2]

    print(f'there are {feat_freq_by_feat_model_same_data_gt2.shape[0]} feature sets have at least 2 shifts each NN')

    ori_data_both_models_gt2=ori_data[ori_data['features'].isin(feat_freq_by_feat_model_same_data_gt2['features'])]
    print(f'there are {ori_data_both_models_gt2.shape[0]} models that have features sets of at least 2 shifts from original performance data')
    data=ori_data_both_models_gt2
#     N=20
    ori_data_both_models_gt2_s=get_top_models(data,ori_data,N)
    uni_feat_final=ori_data_both_models_gt2_s['features'].nunique()
    uni_feat=ori_data_both_models_gt2_s['features'].unique()
    print(f'we select final {uni_feat_final} unique features from the top {N} models from each NN')
    
    return uni_feat_final,uni_feat

def check_anomaly(data,out_dir,marker):
    unique_combo_index=data['combo_index'].nunique()
    print(f'data has captured {unique_combo_index} unique combo_index!')
    df_gt_30,df_lt_30,df_gt_30_list,df_lt_30_list=get_missed_index(data)
    dedup_df=data.drop_duplicates(['combo_index','shift'])
    _,_,df_lt_30_list_v2,df_lt_30_list=get_missed_index(dedup_df)
    dedup_df=dedup_df.sort_values(['combo_index','shift'])
    dedup_df.to_csv(f'{out_dir}/{marker}.csv',index=False)
    return dedup_df, df_gt_30,df_lt_30

def get_missed_index(data):
    df=pd.DataFrame(data['combo_index'].value_counts(dropna=False)).reset_index()
    df.columns=['combo_index','cnts']
    display(df)
    df_gt_30=df.loc[df['cnts']>30]
    df_lt_30=df[df['cnts']<30]
    df_gt_30_list=np.sort(df_gt_30['combo_index'].unique())
    df_lt_30_list=np.sort(df_lt_30['combo_index'].unique())
    print(f'greater than 30 has {df_gt_30.shape[0]}\n'
    f'they are {df_gt_30_list}\n'
    f'less than 30 has {df_lt_30.shape[0]}\n'
    f'they are {df_lt_30_list}')
    return df_gt_30,df_lt_30,df_gt_30_list,df_lt_30_list

def add_combo_index(data_dir,out_dir):
    
    for i,f in enumerate(os.listdir(data_dir)):
        
        if f.endswith('.csv'):
            print(i, f)
            combo_range=f.split('_')[2].split('.')[0]
            combo_start=int(combo_range.split('-')[0])
            combo_end=int(combo_range.split('-')[1])
            print(f'combo start/end: {combo_start}/{combo_end}')
            df=pd.read_csv(os.path.join(data_dir,f))
            combo_index_list=[]
            for s in df['shift']:

                combo_index_list.append(combo_start)
                if s==30:
                    combo_start+=1
                    display(df.tail())

            df['combo_index']=combo_index_list
            df=df[['model_name','combo_index','features','shift','ckpt','MSE','RMSE','L1']]
            display(df.tail())
            file_name=f.split('.csv')[0]
            df.to_csv(f'{out_dir}/{file_name}_w_combo.csv',index=False)
            if combo_start>combo_end:
                continue
        
        
    
            
def combine_time_feat_data(data_dir,check_cols,marker):
    now=datetime.now()
    print(now)
    dat=pd.DataFrame()
    for i,f in enumerate(os.listdir(data_dir)):
        if f.endswith('.csv'):
            df=pd.read_csv(os.path.join(data_dir,f))
            if 'Unnamed: 0' in df.columns:
                df=df.drop('Unnamed: 0',axis=1)
            dat=dat.append(df)
            print(i,f,df.shape)
    if check_cols:
        dat=dat.drop_duplicates(check_cols)
    display(dat.head())
    print(dat.shape)
    out_dir=os.path.join(data_dir,'COMBINE')
    os.makedirs(out_dir,exist_ok=True)
    dat.to_csv(os.path.join(out_dir,f'{marker}_total.csv'),index=False)
    return dat

def get_hist(data_path,data_type):
    import math
    timeStep_total=pd.read_csv(data_path)
    if 'alg2' in data_type:
        threshold=0.4054
        avg_step_per_day=9
    else:
        threshold=0.4211
        avg_step_per_day=8
    timeStep_total_op=timeStep_total[timeStep_total['RMSE']<threshold]
    print(f'{data_type} timeStep_total_op.shape{timeStep_total_op.shape}')

    timeStep_total_op['which_day']=timeStep_total_op['shift']//avg_step_per_day
    timeStep_total_op['which_day']=timeStep_total_op['which_day'].astype(str)
    display(timeStep_total_op.head())
    g=sns.histplot(data=timeStep_total_op,x='which_day')
    g.set_title(f'Num of days shift back to outperform full sequence for {data_type}')
    plt.savefig(f'TRANS_DATA/{data_type}_shift_days_histogram_outperform_models.jpg')
    return timeStep_total_op,g


def get_algo_attr(model,X_test,y_test,X_train,num_var,input_weight_layer):
    
    X_test= torch.tensor(X_test).float()

    y_test = torch.tensor(y_test).view(-1, 1).float()
    X_train=torch.tensor(X_train).float()
    print(X_test.shape, y_test.shape,X_train.shape)
    # ig = IntegratedGradients(model)
    # ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)

    # ig_attr_test = ig.attribute(X_test, n_steps=2)
    # ig_nt_attr_test = ig_nt.attribute(X_test)
    start=time.time()
    print('...start attributing...')
    dl_attr_test = dl.attribute(X_test)
    gs_attr_test = gs.attribute(X_test, X_train)
    fa_attr_test = fa.attribute(X_test)
    end=time.time()
    m,s=divmod((end-start),60)
    print(f'it took {m} min and {round(s,2)} seconds to finish attribution!')
    print(dl_attr_test.shape,gs_attr_test.shape,fa_attr_test.shape)

    dl_attr_test_sum = dl_attr_test.reshape(-1,num_var).detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    gs_attr_test_sum = gs_attr_test.reshape(-1,num_var).detach().numpy().sum(0)
    gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

    fa_attr_test_sum = fa_attr_test.reshape(-1,num_var).detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

    lstm_weight = model.input_weight_layer.detach().numpy().sum(0)
    y_axis_lstm_weight = lstm_weight / np.linalg.norm(lstm_weight, ord=1)
    attr_dict={
        'dl_attr_test_norm_sum':dl_attr_test_norm_sum,
        'gs_attr_test_norm_sum':gs_attr_test_norm_sum,
        'fa_attr_test_norm_sum':fa_attr_test_norm_sum,
        'y_axis_lstm_weight':y_axis_lstm_weight
    }
    print(f'dl_attr_test_norm_sum shape: {dl_attr_test_norm_sum.shape}\n'
         f'gs_attr_test_norm_sum shape: {gs_attr_test_norm_sum.shape}\n'
         f'fa_attr_test_norm_sum shape: {fa_attr_test_norm_sum.shape}\n'
         f'y_axis_lstm_weight shape: {y_axis_lstm_weight.shape}\n'    
    )
    return attr_dict
    
    
def graph_attr(attr_dict,test_feature_list,marker):
    dl_attr_test_norm_sum=attr_dict['dl_attr_test_norm_sum']
    gs_attr_test_norm_sum=attr_dict['gs_attr_test_norm_sum']
    fa_attr_test_norm_sum=attr_dict['fa_attr_test_norm_sum']
    y_axis_lstm_weight=attr_dict['y_axis_lstm_weight']
    
    feature_names=test_feature_list
    num_var=len(feature_names)
    x_axis_data = np.arange(X_test.shape[2])
    x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))
    width = 0.14
    legends = ['DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']
    # 'Int Grads', 'Int Grads w/SmoothGrad'
    plt.figure(figsize=(20, 10))

    ax = plt.subplot()
    ax.set_title(f'Comparing input feature importances across multiple algorithms and learned weights for {num_var} variables')
    ax.set_ylabel('Attributions')

    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
    plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

    # ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
    # ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
    ax.bar(x_axis_data , dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
    ax.bar(x_axis_data + 1 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
    ax.bar(x_axis_data + 2 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
    ax.bar(x_axis_data + 3 * width, y_axis_lstm_weight, width, align='center', alpha=1.0, color='grey')
    ax.autoscale_view()
    plt.tight_layout()

    ax.set_xticks(x_axis_data + 0.5)
    ax.set_xticklabels(x_axis_data_labels)

    plt.legend(legends, loc=3)
    plt.savefig(f'TRANS_MODEL/BKT_MV/3_interp_algo_vs_weights_{marker}.jpg')
    plt.show()
    

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


def counter_feature(data_path,noProf_path,full_flag,threshold,marker):
    
    import warnings
    from ast import literal_eval
    warnings.filterwarnings("ignore")
    result=pd.read_csv(data_path)
    print(f'read in file shape: {result.shape}')
    # result['features'].nunique()
    if full_flag:
        result_noprof=result[~result['features'].str.contains('proficiency')] 
    else:
        result_noprof=result
    print(f'data has no proficiency shape: {result_noprof.shape}')
    # print(result_noprof['features'].nunique())
    result_noprof['features']=result_noprof['features'].apply(lambda x:literal_eval(x))
    cur_arr=result_noprof['features'].unique()#70
    dat_noProf=pd.read_csv(noProf_path)
    dat_noProf['combo']=dat_noProf['combo'].apply(lambda x:literal_eval(x))
    total_arr=dat_noProf['combo'].unique()#186
    arr_diff=np.setdiff1d(total_arr,cur_arr)
    print(f'there are {len(arr_diff)} features in the non proficiency combos but not in the current file')
    # ---check top 20 contain prof--
    
    # check models better than the baseline dkt--
    if threshold:
        
        result_noprof_valid=result_noprof[result_noprof['RMSE']<threshold]
        result_noprof_valid.to_csv(f'TRANS_MODEL/BKT_MV/{marker}_outperf_baseline.csv',index=False)
        noprof_uni=result_noprof_valid['features'].unique()
        print(f'when less than threshold, there are {result_noprof_valid.shape[0]} models\n'
        f'there are {len(noprof_uni)} sets of features that have better performance!')
        return_df= result_noprof_valid
    else:
        result_noprof_top20=result_noprof.sort_values('RMSE').head(20)
        noprof_uni=result_noprof_top20['features'].unique()
        return_df= result_noprof_top20
    print(f'they are feature sets:\n{noprof_uni}')
    # ---counter the frequency of the features w/o prof---
    # aset=('question_type_1', 'bktcount', 'quest_difficulty', 'usmo_difficulty')
    from collections import Counter
    alist=[]
    for i, subset in enumerate(noprof_uni):
        for j in subset:

            alist.append(j)
    uni_alist=Counter(alist)
    print(uni_alist)
    return uni_alist,return_df

# =======process imputed data================

    
def clean_featEng_v2(data_path,save_dir):
    """
    this func is to create features such as item/quest/usmo difficulties on the individual users
    """
    data_nar=pd.read_csv(data_path)
    user_id=int(data_path.split('/')[-1].split('.csv')[0])

    data_nar['student_id']=user_id
    data_nar['correct']=np.where(data_nar['score']==data_nar['max_score'],1,0)
    if data_nar['correct'].sum()!=0:
    # ---calculate quest difficulty--
        quest_total_stu=pd.pivot_table(data_nar, index='quest_ref',values='student_id',aggfunc=pd.Series.nunique).reset_index()
        quest_correct_stu=pd.pivot_table(data_nar, index='quest_ref',values='correct',aggfunc=np.sum).reset_index()
        quest_diff_df=pd.merge(quest_total_stu,quest_correct_stu,how='inner',on='quest_ref')
        # print(f'quest_diff_df.shape: {quest_diff_df.shape}')
        quest_diff_df['quest_difficulty']=round(quest_diff_df['correct']/quest_diff_df['student_id'],4)
        data_eng=pd.merge(data_nar,quest_diff_df[['quest_ref','quest_difficulty']],how='left',on='quest_ref')
        # ---calculate item difficulty--
        item_total_stu=pd.pivot_table(data_nar, index='item_ref',values='student_id',aggfunc=pd.Series.nunique).reset_index()
        item_correct_stu=pd.pivot_table(data_nar, index='item_ref',values='cor rect',aggfunc=np.sum).reset_index()
        item_diff_df=pd.merge(item_total_stu,item_correct_stu,how='inner',on='item_ref')
        # print(f'item_diff_df.shape: {item_diff_df.shape}')
        item_diff_df['item_difficulty']=round(item_diff_df['correct']/item_diff_df['student_id'],4)
        data_eng=pd.merge(data_eng,item_diff_df[['item_ref','item_difficulty']],how='left',on='item_ref')
        # ---calculate usmo difficulty---
        usmo_total_stu=pd.pivot_table(data_nar, index='usmoid',values='student_id',aggfunc=pd.Series.nunique).reset_index()
        usmo_correct_stu=pd.pivot_table(data_nar, index='usmoid',values='correct',aggfunc=np.sum).reset_index()
        usmo_diff_df=pd.merge(usmo_total_stu,usmo_correct_stu,how='inner',on='usmoid')
        # print(f'usmo_diff_df.shape: {usmo_diff_df.shape}')
        usmo_diff_df['usmo_difficulty']=round(usmo_diff_df['correct']/usmo_diff_df['student_id'],4)
        data_eng=pd.merge(data_eng,usmo_diff_df[['usmoid','usmo_difficulty']],how='left',on='usmoid')
        data_eng['score_rate']=round(data_eng['score']/data_eng['max_score'],6)
        feature_list=['item_ref','usmoid','question_type','quest_ref']
        data=data_eng
        data_eng2=enc_features(data,feature_list)

        # data_eng2=data_eng2.drop('quest_ref',axis=1)
        
        data_eng2=data_eng2.rename(columns={'event_time_first':'event_time'})
        data_eng2['seq']=data_eng2.index
        data_eng2=data_eng2[['event_time', 'student_id', 'quest_ref_1', 'item_ref_1', 'usmoid_1',
        'correct', 'score_rate', 'question_type_1', 'seq_number', 'bktcount',
        'proficiency', 'assessment_duration', 'count', 'quest_difficulty',
        'item_difficulty', 'usmo_difficulty','seq']]
        print(f'cleaned data shape {data_eng2.shape}')

        data_eng2.to_csv(f'{save_dir}/{user_id}.csv',index=False)
    else:
        pass
        data_eng2=None
    return data_eng2


def min_max_scale(data,stat_df,feature_list,is_reverse):
    data=copy.deepcopy(data)

    for col in feature_list:
        # print(col)
        min=stat_df[stat_df['stat']=='min'][col].values[0]
        max=stat_df[stat_df['stat']=='max'][col].values[0]
        if is_reverse:
            if col in [ 'bktcount','seq_number', 'assessment_duration', 'count', 'quest_ref_1', 'item_ref_1', 'usmoid_1','question_type_1']:
                data[col]=data[col].apply(lambda x:round(x*(max-min)+min,0))
            else:
                data[col]=data[col].apply(lambda x:round(x*(max-min)+min,4))
        else:
            data[col]=data[col].apply(lambda x:round((x-min)/(max-min),4))
        # display(data.head())
    return data


def get_null_pairs(data):
    
    null_index = data[data['student_id'].isna()].index
    print(f'there are {len(null_index)} null values')

    start_list=[null_index[0]]

    for i,value in enumerate(null_index):
        if i<len(null_index)-1 and null_index[i+1]-null_index[i]>1:
            start_list.append(null_index[i])
            start_list.append(null_index[i+1])
    print(f'the null indices are : {start_list}')
    null_pairs=[]
    j=0
    while j<len(start_list)-1:
        start=start_list[j]
        end=start_list[j+1]
        null_pairs.append((start,end))
        j+=2
    return null_pairs



# =======process imputed data================

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



def truncate_data(data_path,truncate_size):
    data=pd.read_csv(data_path)
    quest_list=data['quest_ref'].unique()
    data_n=data[data['quest_ref'].isin(quest_list[:truncate_size])]
    quest_uni_len=data_n['quest_ref'].nunique()
    print(f'truncated data size {data_n.shape} with unique {quest_uni_len} questions!')
    return data_n
    

def split_user_seq(data_path,cadence,out_dir):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    le=LabelEncoder()
    truncate_size=int(QUESTION_NUM[out_dir])
    data=truncate_data(data_path,truncate_size)
    data['quest_ref_index']=le.fit_transform(data['quest_ref'])
    data['quest_ref_index']=data['quest_ref_index']+1
    data['correctness']=np.where(data['score']==data['maxscore'],1,0)
#     print(f'there are {data['quest_ref_index'].nunique()} questions!')
    kfd=KFold(n_splits=5)
    data_n=data[['studentid','endedattime','assessmenttype','seqnumber','quest_ref_index','correctness']]
    for i, (train_index, test_index) in enumerate(kfd.split(data_n['studentid'].unique())):
        print('====')
#         print(f'Fold {i}:\n  train user index {train_index} with length {len(train_index)}\n  test user index {test_index} with length {len(test_index)}')
        train_stu_list=data_n['studentid'].unique()[train_index]
        train_stu, val_stu=train_test_split(train_stu_list, test_size=0.1,random_state=88)
        print(f'  Fold {i} has {len(train_stu)} train users,{len(val_stu)} val users')
        train_by_user=data_n.loc[data_n['studentid'].isin(train_stu)]
        print(f'  train_by_user shape {train_by_user.shape}')
        val_by_user=data_n.loc[data_n['studentid'].isin(val_stu)]
        print(f'  val_by_user shape {val_by_user.shape}')
        train_by_user=train_by_user.sort_values(['studentid','endedattime','assessmenttype','seqnumber'])
        val_by_user=val_by_user.sort_values(['studentid','endedattime','assessmenttype','seqnumber'])
        total_steps=0
        train_out_dir=f'{out_dir}/processed/{i+1}/train'
        val_out_dir=f'{out_dir}/processed/{i+1}/val'
        test_out_dir=f'{out_dir}/processed/{i+1}/test'
        
        os.makedirs(train_out_dir,exist_ok=True)
        os.makedirs(val_out_dir,exist_ok=True)
        os.makedirs(test_out_dir,exist_ok=True)
        for m, u in enumerate(list(train_by_user['studentid'].unique())):
            stu=train_by_user[train_by_user['studentid']==u]
            time_step=stu['quest_ref_index'].nunique()
            total_steps+=time_step
            if m%cadence==0:
                print(f'  train_by_user data split: {m}th User with ID {u} has {time_step}-step sequence')
                
            stu[['quest_ref_index','correctness']].to_csv(f'{train_out_dir}/{u}.csv',index=False)
        print(f'>>fold {i} train data has average time steps of {round(total_steps/len(train_stu),2)}<<')
        total_steps=0
        for n, u in enumerate(list(val_by_user['studentid'].unique())):
    #         print(f'  val_by_user data split by user: {n},{u}')
            stu=val_by_user[val_by_user['studentid']==u]
    #         display(stu)
            time_step=stu['quest_ref_index'].nunique()
            total_steps+=time_step
            if n%cadence==0:
                print(f'  val_by_user data split: {n}th User with ID {u} has {time_step}-step sequence')
            stu[['quest_ref_index','correctness']].to_csv(f'{val_out_dir}/{u}.csv',index=False)
        print(f'>>fold {i} val data has average time steps of {round(total_steps/len(val_stu),2)}<<')
        test_stu_list=data_n['studentid'].unique()[test_index]
        print(f'  Fold {i} has {len(test_stu_list)} test users')
        test_by_user=data_n.loc[data_n['studentid'].isin(test_stu_list)]
        print(f'  test_by_user shape {test_by_user.shape}')
        test_by_user=test_by_user.sort_values(['studentid','endedattime','assessmenttype','seqnumber'])
    #     test_by_user=test_by_user[['studentid','quest_ref_index','correctness']]
        total_steps=0
        for k, u in enumerate(list(test_by_user['studentid'].unique())):
    #         print(f'  test_by_user data split by user: {k},{u}')
            stu=test_by_user[test_by_user['studentid']==u]
            time_step=stu['quest_ref_index'].nunique()
            total_steps+=time_step
            if k%cadence==0:
                print(f'  test_by_user data split: {k}th User with ID {u} has {time_step}-step sequence')
    #         display(stu)
            stu[['quest_ref_index','correctness']].to_csv(f'{test_out_dir}/{u}.csv',index=False)
        print(f'>>fold {i} test data has average time steps of {round(total_steps/len(test_stu_list),2)}<<')

        
QUESTION_NUM = {
    'Geom_user':501,
    # /497,
    'Alg2_user':501,
    'alg2_bkt':1154,
    'geom_bkt':1430,
     'alg2_bkt_v2':2566,
    'geom_bkt_v2':2566,
    'modified_AAAI20': 16175,
    'ASSISTments2009': 110,#SKILL NAME
    'ASSISTments2012': 0, # TODO: fix
    'ASSISTments2015': 100,
    'ASSISTmentsChall': 102,
    'KDDCup': 0, # TODO: fix
    'Junyi': 722,
    'STATICS': 1223,
    'EdNet-KT1': 18143,
    'K12':1167,
    'K12_USMO':2688,
    'assess_USMO':2688,
    'practice_USMO':2688,
    'exercise_USMO':2688,
    'review_USMO':2688,
    'quiz_USMO':2688,
    'eval_USMO':2688,
    'teacher_USMO':2688,
    'self_USMO':2688,
    'assessment':4065,
    'assess4practice':2181,
    'assess4exercise':3181,
    'assess4eval':638,
    'evaluation':638,
    'review':5903,
    'review4assess':4065,
    'review4exercise':3181,
    'review4practice':2181,
    'review4eval':638,
    'exercise':3181,
    'exercise4practice':2181,
    'exercise4eval':638,
    'practice':2181,
    'practice4eval':638,
    'quiz':7082,
    'quiz4assess':4065,
    'quiz4practice':2181,
    'quiz4exercise':3181,
    'quiz4review':5903,
    'quiz4eval':638,
    'k12_new_all':26459,
    'algbr_diag':2004,
   'algbr_practice':5632,
    'teacher':11754,
    'teacher4self':10721,
    'teacher4assess':4065,
    'teacher4practice':2181,
    'teacher4exercise':3181,
    'teacher4quiz':7082,
    'teacher4review':5903,
    'teacher4eval':638,
    'self':10721,
    'self4assess':4065,
    'self4practice':2181,
    'self4exercise':3181,
    'self4review':5903,
    'self4quiz':7082,
    'self4eval':638,
}
# with open ('scripts/question_num.json','w') as f:
#     json.dump(QUESTION_NUM,f)
