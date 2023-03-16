from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import os,random
import pandas as pd
import torch,copy
import numpy as np
import itertools
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt




def combine_model_run_results(data_dir):
    df=pd.DataFrame()
    datt=pd.DataFrame()
    for i,f in enumerate(os.listdir(data_dir)):
        print(f'\n{i}th file: {f}\n')
        if f.endswith('.csv'):
            data=pd.read_csv(f'{data_dir}/{f}')
            na_fill_type=data.iloc[0]['na_fill_type']
            data_name=data.iloc[0]['data']
            model_name=data.iloc[0]['model_name']
            marker=data.iloc[0]['marker']
            avg_mse=data['MSE'].mean()
            avg_rmse=data['RMSE'].mean()
            avg_l1=data['L1'].mean()
            if '100.0' in f:
                parts=f.split('TRAIN_')[1].split('_')
                vae_flag=parts[0]
                loss=parts[1]
                perc_flag=parts[2]
            
            else:
                
                vae_flag=marker.split('_')[3]
                loss=marker.split('_')[4]
                if 'ALIGNED' in marker:
                    print(marker.split('_'))
                    perc_flag=marker.split('_')[6]
                else:
                    
                    perc_flag=marker.split('_')[5]
            

            data['vae_flag']=vae_flag
            data['loss']=loss
            data['perc_flag']=perc_flag
            df=df.append(data)
            dat=pd.DataFrame({'data_name':[data_name],'na_fill_type':[na_fill_type],
                              'model_name':[model_name],
                             'vae_flag':[vae_flag],'loss':[loss],
                             'perc_flag':[perc_flag],'avg_mse':[avg_mse],
                             'avg_rmse':[avg_rmse],'avg_l1':[avg_l1]})
            datt=datt.append(dat)
    df=df.sort_values(['data','model_name','vae_flag','perc_flag'])
    datt=datt.sort_values(['data_name','model_name','vae_flag','perc_flag'])
    save_dir='/'.join(data_dir.split('/')[:3])+'/FINAL'
    os.makedirs(save_dir,exist_ok=True)
    print(f'..saving combined overall data under {save_dir}/PERC_DATA_COMBINED.csv')
    df.to_csv(f'{save_dir}/PERC_DATA_COMBINED.csv',index=False)
    print(f'..saving combined avg data under {save_dir}/PERC_DATA_AVG_COMBINED.csv')
    datt.to_csv(f'{save_dir}/PERC_DATA_AVG_COMBINED.csv',index=False)
    return df, datt

def get_perc_aug_data(augment_data_path,ratio,data_name):
    """
    this func runs in the root dir of KT/
    """
    data=pd.read_csv(augment_data_path)

    split_stu_ids=pd.read_csv(f'TRANS_DATA/{data_name}_no_outlier_downstream_train_val_test_stu_list.csv')
    aug_split_marker=pd.merge(data,split_stu_ids,how='left',on='student_id')
    print(f' after merging with the split student list shape: {aug_split_marker.shape}')
    aug_train_stus=list(aug_split_marker[aug_split_marker['split']=='train']['student_id'].unique())
    aug_val_stus=list(aug_split_marker[aug_split_marker['split']=='val']['student_id'].unique())
    aug_test_stus=list(aug_split_marker[aug_split_marker['split']=='test']['student_id'].unique())
    print(f' there are {len(aug_train_stus)}/{len(aug_val_stus)}/{len(aug_test_stus)} unique train/val/test students in augmented data')
    
    train_num=int(len(aug_train_stus)*ratio)
    train_aug=aug_split_marker[aug_split_marker['student_id'].isin(aug_train_stus[:train_num])]
    val_num=int(len(aug_val_stus)*ratio)
    val_aug=aug_split_marker[aug_split_marker['student_id'].isin(aug_val_stus[:val_num])]
    test_num=int(len(aug_test_stus)*ratio)
    test_aug=aug_split_marker[aug_split_marker['student_id'].isin(aug_test_stus[:test_num])]
    ratio_perc=int(ratio*100)
    print(f'we extract the first {ratio_perc}% of the train/val/test:{train_num}/{val_num}/{test_num} students from data with')
    total_sub_data=pd.concat([train_aug,val_aug,test_aug])
    print(f'...that is {len(train_aug)}/{len(val_aug)}/{len(test_aug)} rows from train/val/test students in the augmented data with total {len(total_sub_data)} rows ')
    
    save_path=augment_data_path.split('.csv')[0]+f'_{ratio_perc}_PERC_DATA.csv'
    
    total_sub_data_n=total_sub_data.drop('split',axis=1)
    total_sub_data_n.to_csv(save_path,index=False)


    return total_sub_data_n


def align_data_size(vae_path,lvae_path):
    """
    this func is to plot the distribution match between the original imputeeration data and the imputed data
    
    """
    
    VAE_data=pd.read_csv(vae_path)
    LVAE_data=pd.read_csv(lvae_path)
    print(f'LAVE/VAE data shape before aligning:{VAE_data.shape}/{LVAE_data.shape}')
    VAE_data=VAE_data.iloc[:LVAE_data.shape[0]]
    print(f'LAVE/VAE data shape after aligning:{VAE_data.shape}/{LVAE_data.shape}')
    vae_path_stem=vae_path.split('.csv')[0]
    save_path=f'{vae_path_stem}_ALIGNED.csv'
    print(f'..saving truncated VAE_data data to path:{save_path}..')
    VAE_data.to_csv(save_path,index=False)
    return VAE_data
    
      

def plot_dist_match(orig_gen_path,orig_gen_label_path,vae_gen_path,lvae_gen_path,marker):
    """
    this func is to plot the distribution match between the original generation data and the imputed data
    
    """
    orig_gen_data_readings=pd.read_csv(orig_gen_path)
    orig_gen_data_label=pd.read_csv(orig_gen_label_path)
    orig_gen_data_comb=pd.concat([orig_gen_data_readings,orig_gen_data_label[['event_time', 'student_id',
       'score_rate']]],axis=1)
    orig_gen_VAE=pd.read_csv(vae_gen_path)
    cut_len=np.min([len(orig_gen_data_comb),len(orig_gen_VAE)])

    if lvae_gen_path:
        orig_gen_LVAE=pd.read_csv(lvae_gen_path)
        cut_len=min(cut_len,len(orig_gen_LVAE))
        orig_gen_LVAE=orig_gen_LVAE.iloc[:cut_len]

    orig_gen_VAE=orig_gen_VAE.iloc[:cut_len]
    
    orig_gen_data_comb=orig_gen_data_comb.iloc[:cut_len]
    print(f'..saving the truncated VAE data to path')
    print(f'orig_gen_data_readings/orig_gen_VAE/:{orig_gen_data_comb.shape}/{orig_gen_VAE.shape}')
    if lvae_gen_path:print(f'orig_gen_LVAE.shape:{orig_gen_LVAE.shape}')
    print(f'\n>>>below is the nunique stats comparison for {marker}<<<\n')
    print(f'\n..{marker} original GEN data W/O reconstruction..\n')
    print(orig_gen_data_comb.nunique())
    print(f'\n..{marker} VAE generation on GEN data...\n')
    print(orig_gen_VAE.nunique())
    if lvae_gen_path:
        print(f'\n..{marker} LVAE generation on GEN data...\n')
        print(orig_gen_LVAE.nunique())
        
        
    print(f'\n>>>below is the description stats comparison for {marker}<<<\n')
    print(f'\n..{marker} original GEN data W/O reconstruction..\n')
    display(orig_gen_data_comb.describe())
    print(f'\n..{marker} VAE generation on GEN data...\n')
    display(orig_gen_VAE.describe())
    if lvae_gen_path:
        
        print(f'\n..{marker} LVAE generation on GEN data...\n')
        display(orig_gen_LVAE.describe())



    check_cols=['seq_number', 'assessment_duration', 'bktcount', 'count',
           'quest_difficulty', 'item_difficulty', 'usmo_difficulty', 'quest_ref_1',
           'item_ref_1', 'usmoid_1', 'question_type_1']


    for i, col in enumerate(check_cols):
        sns.kdeplot(data=orig_gen_VAE,x=col,shade=True,color='purple',label='generated-VAE')
        if lvae_gen_path:
            sns.kdeplot(data=orig_gen_LVAE,x=col,shade=True,color='blue',label='generated-LVAE')
        
        sns.kdeplot(data=orig_gen_data_comb,x=col,shade=True,color='green',label='original data')



        plt.legend()
        plt.show()
        # print('====')
        save_dir='results/k12/plots'
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'{save_dir}/dist_compare_original_imputed_{marker}.jpg')



def combine_data_aux(df_path,aux_data_path):
    """
    combine the data_readings and auxilary data into one file
    """
    df_eng=pd.read_csv(df_path)
    print(f'df_eng shape:{df_eng.shape}')
    display(df_eng.head())
    aux_data=pd.read_csv(aux_data_path)
    print(f'aux_data shape:{aux_data.shape}')
    aux_data.head()
    df_eng_aux=pd.concat([df_eng,aux_data],axis=1)
    print(f'total data has shape: {df_eng_aux.shape}')
    display(df_eng_aux.head())
    

    save_path=df_path.split('.csv')[0]+'_'+aux_data_path.split('.csv')[0].split('/')[-1]+'_combined'
    print(f'..saving combined aux data under {save_path}.csv...')
    if 'correct' not in df_eng_aux.columns:
        df_eng_aux['correct']=np.nan
        df_eng_aux['score_rate']=np.nan
    print(f'combined data has shape:{df_eng_aux.shape}')
    display(df_eng_aux.head())
    df_eng_aux.to_csv(f'{save_path}.csv',index=False)
    return df_eng_aux


def get_train_val_test_vae(df_path,aux_data_path,ratios):
    """
    this func is to separate data for LVAE into five parts: train/val/test/pred/gen
    """
    df_eng=pd.read_csv(df_path)
    aux_data=pd.read_csv(aux_data_path)
    df_eng=df_eng.sort_values(['student_id','event_time','seq_number'])
    aux_data['student_id']=aux_data['student_id'].astype(str)
    df_eng['student_id']=df_eng['student_id'].astype(str)
    df_eng_aux=pd.merge(df_eng,aux_data,how='left',on='student_id')
    print(f'total data has shape: {df_eng_aux.shape}')
    
    save_path=df_path.split('.csv')[0]+'_'+aux_data_path.split('.csv')[0].split('/')[-1]+'_combined'
    print(f'..saving combined aux data under {save_path}.csv...')
    if 'correct' not in df_eng_aux.columns:
        df_eng_aux['correct']=np.nan
        df_eng_aux['score_rate']=np.nan
    display(df_eng_aux.head())
    df_eng_aux.to_csv(f'{save_path}.csv',index=False)
    if ratios:
        total_stu_num=df_eng_aux['student_id'].nunique()
        stu_num_list=[int(total_stu_num*x) for x in ratios]
        total_stu=df_eng_aux['student_id'].unique()
        total_stu_num=df_eng_aux['student_id'].nunique()
    

        train_stu_num,val_stu_num,test_stu_num,pred_stu_num,gen_stu_num=stu_num_list[0],stu_num_list[1],stu_num_list[2],stu_num_list[3],stu_num_list[4]
            
        stu_train=total_stu[:train_stu_num]
        stu_val=total_stu[train_stu_num:train_stu_num+val_stu_num]
        stu_test=total_stu[train_stu_num+val_stu_num:train_stu_num+val_stu_num+test_stu_num]
        stu_pred=total_stu[train_stu_num+val_stu_num+test_stu_num:train_stu_num+val_stu_num+test_stu_num+pred_stu_num]
        stu_gen=total_stu[train_stu_num+val_stu_num+test_stu_num+pred_stu_num:train_stu_num+val_stu_num+test_stu_num+pred_stu_num+gen_stu_num]
        print(f'.. train/val/test/val/pred/gen student number: {len(stu_train)}/{len(stu_val)}/{len(stu_test)}/{len(stu_pred)}/{len(stu_gen)}')
        train_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_train)]
        val_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_val)]
        test_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_test)]
        pred_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_pred)]
        gen_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_gen)]
        print(f'.. train/val/test/val/pred/gen data split by students has shape {train_df.shape}/{val_df.shape}/{test_df.shape}/{pred_df.shape}/{gen_df.shape}')
        print(f'..saving train aux data under {save_path}_train.csv...')
        train_df.to_csv(f'{save_path}_train.csv',index=False)
        print(f'..saving val aux data under {save_path}_val.csv...')
        val_df.to_csv(f'{save_path}_val.csv',index=False)
        print(f'..saving test aux data under {save_path}_test.csv...')
        test_df.to_csv(f'{save_path}_test.csv',index=False)
        print(f'..saving pred aux data under {save_path}_pred.csv...')
        pred_df.to_csv(f'{save_path}_pred.csv',index=False)
        print(f'..saving gen aux data under {save_path}_gen.csv...')
        gen_df.to_csv(f'{save_path}_gen.csv',index=False)
    else:
        train_df,val_df,test_df,pred_df,gen_df=None,None,None,None,None
    return train_df,val_df,test_df,pred_df,gen_df,df_eng_aux


def create_npz(want_csv,data,enc_list,csv_save_marker,npz_save_marker):
    """
    this func is to create the npz data from the engineered data and auxilary data and  masks as well
    """
    # from sklearn.preprocessing import LabelEncoder

    # df_eng=df_eng.sort_values(['student_id','event_time','seq_number'])
    # data=pd.merge(df_eng,aux_data,how='left',on='student_id')
    # print(data.shape)

    if 'correct' in data.columns:

        df_aux_final=data[['event_time', 'SCHOOL_ID','SPECIAL_ED', 'student_id','FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL' ,'correct','score_rate']]
    else:
        df_aux_final=data[['event_time', 'SCHOOL_ID','SPECIAL_ED', 'student_id','FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL' ]]
        df_aux_final['correct']=np.nan
        df_aux_final['score_rate']=np.nan
    print(df_aux_final.shape)
    df_aux_final=df_aux_final.fillna(0)
    display(df_aux_final.isna().sum())
    if want_csv:
        print(f'....saving data to data/k12/split_data/{csv_save_marker}_outcome_labels.csv.... ')
        df_aux_final.to_csv(f'data/k12/split_data/{csv_save_marker}_outcome_labels.csv',index=False)
    df_aux_final_mask=np.where(df_aux_final.isna(),0,1)
    df_aux_final_mask_df=pd.DataFrame(df_aux_final_mask,columns=df_aux_final.columns)
    if want_csv:
        print(f'....saving data to data/k12/split_data/{csv_save_marker}_aux_final_mask_df.csv.... ')
        df_aux_final_mask_df.to_csv(f'data/k12/split_data/{csv_save_marker}_aux_final_mask_df.csv',index=False)

    df_data_final=data[[ 'seq_number', 'assessment_duration', 'bktcount','count',
        'quest_difficulty', 'item_difficulty', 'usmo_difficulty',
            'quest_ref_1', 'item_ref_1', 'usmoid_1',
        'question_type_1']]
    print(df_data_final.shape)
    display(df_data_final.isna().sum())
    if want_csv:
        print(f'....saving data to data/k12/split_data/{csv_save_marker}_data_readings.csv.... ')
        df_data_final.to_csv(f'data/k12/split_data/{csv_save_marker}_data_readings.csv',index=False)
    df_data_final.isna().sum() #no missing values, we need the missing value version
    df_data_final_mask=np.where(df_data_final.isna(),0,1)
    df_data_final_mask_df=pd.DataFrame(df_data_final_mask,columns=df_data_final.columns)
    if want_csv:
        print(f'....saving data to data/k12/split_data/{csv_save_marker}_data_final_mask_df.csv.... ')
        df_data_final_mask_df.to_csv(f'data/k12/split_data/{csv_save_marker}_data_final_mask_df.csv',index=False)
    print('data_readings/data_mask/outcome/outcome_mask shape:')
    print(df_data_final.shape,df_data_final_mask_df.shape,df_aux_final.shape,df_aux_final_mask_df.shape)
    df_data_final=df_data_final.fillna(0)
    data_readings=df_data_final.values
    scaler=MinMaxScaler()
    data_readings=scaler.fit_transform(data_readings)
    data_masks=df_data_final_mask_df.values
    
    flag=f'{npz_save_marker}_outcome_labels'
    save_dir='data/k12/pre_train/label_maps_wo_nas'
    os.makedirs(save_dir,exist_ok=True)
    for feat_name in enc_list:
        df_aux_final=encode_feature(df_aux_final,save_dir,feat_name,flag)
    labels=df_aux_final.fillna(0).values
    label_masks=df_aux_final_mask_df.values
    print(f'..saving npz data into data/k12/split_data/{npz_save_marker}_data.npz...\n')
    np.savez(f'data/k12/split_data/{npz_save_marker}_data.npz',data_readings=data_readings,outcome_attrib=labels,data_mask=data_masks,outcome_mask=label_masks)
    return df_data_final,df_data_final_mask_df,df_aux_final,df_aux_final_mask_df


def combine_na_data(root_dir,data_name,cols_keep,marker):
    sub_dir=data_name+'_bkt_v2_retrieved'
    dat=pd.DataFrame()
    split_type_list=['train','val','test']
    for split_type in split_type_list:

        data_dir=f'{root_dir}/{sub_dir}/processed/1/{split_type}'
        save_dir=f'{root_dir}/{sub_dir}/processed/1'
        # os.makedirs(data_path,exist_ok=True)
        split_df=pd.DataFrame()
        for i, f in enumerate(os.listdir(data_dir)):
            # print(i,f)
            if f.endswith('.csv'):
                stu_id=f.split('.csv')[0].split('/')[-1]
                df=pd.read_csv(os.path.join(data_dir,f))
                na_df=df.loc[df['as_of_date'].isna()]
                na_df['student_id']=stu_id
                na_df['event_time']=na_df['event_time_first']
                na_df['seq_number']=na_df['seq_first']
                na_df=na_df.filter(cols_keep)
                exist_cols=na_df.columns
                non_exist_cols=list(set(cols_keep)-set(exist_cols))
                for col in non_exist_cols:
                    na_df[col]=np.nan
                na_df=na_df[cols_keep]  
                split_df=split_df.append(na_df)
                dat=dat.append(na_df)
                if i%100==0:
                    print(f'we are able to catch {len(exist_cols)} columns')
                    print(f'{i}th file {f}:na df has shape:{na_df.shape}')
                    split_df.to_csv(f'{save_dir}/{split_type}_na_data_com.csv',index=False)
        uni_stu0=split_df['student_id'].nunique()
        print(f'...{split_type} data has combined na data shape:{split_df.shape}\n'
        f'{split_type} has {uni_stu0} unique students')
    uni_stu=dat['student_id'].nunique()
    print(f'>>>total data has combined na data shape:{dat.shape}\n'
        f'total data has {uni_stu} unique students')
    display(dat.head())
    dat.to_csv(f'{root_dir}/TRANS_DATA/{data_name}_na_data_com_{marker}.csv',index=False)
    return dat


def min_max_scaler(target_data,orig_df,is_reverse):
    """
    this func can be used to generate scaled data as well as reversing the scaling
    """
    # data=pd.read_csv(target_data_path)
    data=copy.deepcopy(target_data)
    # orig_df=pd.read_csv(orig_df_path)
    stat_df=orig_df.describe().reset_index()
    stat_df=stat_df.rename(columns={'index':'stat'})
    feature_list=list(orig_df.columns)
    for col in feature_list:
        # print(col)
        min=stat_df[stat_df['stat']=='min'][col].values[0]
        max=stat_df[stat_df['stat']=='max'][col].values[0]
        if is_reverse:
            if col in ['seq_number', 'assessment_duration','bktcount', 'count','quest_ref_1', 'item_ref_1',
       'usmoid_1', 'question_type_1']:
                data[col]=data[col].apply(lambda x:round(x*(max-min)+min,0)).astype(int)
            else:
                data[col]=data[col].apply(lambda x:round(x*(max-min)+min,4))
        else:
            data[col]=data[col].apply(lambda x:round((x-min)/(max-min),4))
    # display(data.head())
    # data.to_csv(f'results/k12/Generate/{marker}.csv',index=False)
    return data


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")
# features = 1 #The features=16 is used in the output features for the encoder and the input features of the decoder.


def get_row_num_list(df_path,aux_data_path,ratios):
    df_eng=pd.read_csv(df_path)
    aux_data=pd.read_csv(aux_data_path)
    df_eng=df_eng.sort_values(['student_id','event_time','seq_number'])
    aux_data['student_id']=aux_data['student_id'].astype(str)
    df_eng['student_id']=df_eng['student_id'].astype(str)
    df_eng_aux=pd.merge(df_eng,aux_data,how='left',on='student_id')
    print(f'total data has shape: {df_eng_aux.shape}')
    total_stu_num=df_eng_aux['student_id'].nunique()
    stu_num_list=[int(total_stu_num*x) for x in ratios]
    total_stu=df_eng_aux['student_id'].unique()

    row_num_list=[]
    prev=0
    for i in stu_num_list:
    
        # print(step)
        cur=prev+i
        
        stu_perc=total_stu[prev:cur]
        print(f'{prev}:{cur} idx students')
        prev=cur
        stu_df=df_eng_aux[df_eng_aux['student_id'].isin(stu_perc)]
        row_num=stu_df.shape[0]
        print(f'.. has {row_num} rows')
        row_num_list.append(row_num)
    print(row_num_list)
    return row_num_list






def clean_aux_data_new(data_path,marker):
    """
    this func is to clean and dedup the auxilary data for k12 data w/o missing data
    """
    aux_vars=pd.read_csv(data_path)
    aux_vars=aux_vars.drop(['SAM_STU_ID', 'IDENTITYID', 'STUDENT_ID.1'],axis=1)
    print(aux_vars.shape)#(3682, 7)
    aux_vars_dedup=aux_vars.drop_duplicates('STUDENT_ID')
    print(f'aux_vars_dedup.shape:{aux_vars_dedup.shape}')#(3290, 7)
    
    aux_vars_dedup=aux_vars_dedup.rename(columns={'STUDENT_ID':'student_id'})
    display(aux_vars_dedup.head())
    aux_vars_dedup_n=aux_vars_dedup[['SCHOOL_ID','SPECIAL_ED', 'student_id','FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL']]
    aux_vars_dedup_n.to_csv(f'data/k12/split_data/{marker}_aux_vars_dedup_new.csv',index=False)
    return aux_vars_dedup_n


def clean_aux_data(data_path,marker):
    """
    this func is to clean and dedup the auxilary data for k12 data
    """
    aux_vars=pd.read_csv(data_path)

    print(aux_vars.shape)#(3682, 7)
    aux_vars_dedup=aux_vars.drop_duplicates('STUDENT_ID')
    print(f'aux_vars_dedup.shape:{aux_vars_dedup.shape}')#(3290, 7)
    


    aux_vars_dedup=aux_vars_dedup.rename(columns={'STUDENT_ID':'student_id'})
    display(aux_vars_dedup.head())
    aux_vars_dedup_n=aux_vars_dedup[['LE_SCHOOL_ID','SPECIAL_ED', 'student_id','FREE_REDUCED_LUNCH', 'GIFTED_TALENTED', 'GRADE_LEVEL']]
    aux_vars_dedup_n.to_csv(f'data/k12/{marker}_aux_vars_dedup.csv',index=False)
    return aux_vars_dedup_n

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


def reformat_data(data_dir,save_dir,marker):
    aux_cols=['RecordID','Age','Gender','Height', 'ICUType','Weight']
    time_vars=['Albumin', 'ALP', 'ALT', 'AST','BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS','Glucose', 'HCO3',
        'HCT', 'HR', 'K','Lactate',
        'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
        'NISysABP', 'Na','PaCO2', 'PaO2', 'Platelets',  'pH','RespRate', 'SysABP', 'Temp', 'TroponinI','TroponinT', 'Temp', 'Urine',
        'WBC','Weight'] #there are all 37 variables
    outcome_var=['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival','In-hospital_death']
    df=pd.DataFrame()
    label_df=pd.DataFrame()
    masks=pd.DataFrame()
    for i, f in enumerate(os.listdir(data_dir)):
        save_name=f.split('.txt')[0]
        if i>=0:
            
            file=pd.read_csv(os.path.join(data_dir,f),sep=',')
            file_pv=pd.pivot_table(file,index='Time',columns='Parameter',values='Value',aggfunc=np.sum).reset_index()
            if i%100==0: print(f'gen {i}th file: {f} with shape:{file_pv.shape}')
            
            labels=file_pv[aux_cols]
            labels=labels[~labels['RecordID'].isna()]
            label_df=label_df.append(labels)
            remainder=set(file_pv.columns)-set(aux_cols)
            data=file_pv[list(remainder)+['RecordID']]            
            data.to_csv(f'{save_dir}/physionet/{save_name}_masked_data.csv',index=False)
            df=pd.concat([df,data])
    
          
    print(f'df shape:{df.shape}')
    display(df.head())
    df['RecordID']=df['RecordID'].ffill()
    df.to_csv(f'{save_dir}/{marker}_masked_data.csv',index=False)
    mask=np.where(df.isna(),0,1)
    mask_df=pd.DataFrame(mask,columns=df.columns)
    print(f'mask df shape:{mask_df.shape}')
    mask_df.to_csv(f'{save_dir}/{marker}_data_masks.csv',index=False)
    
    print(f'label_df shape:{label_df.shape}')
    display(label_df.head())
    label_df.to_csv(f'{save_dir}/{marker}_labels.csv',index=False)
    return mask_df,df,label_df



# =========================ORIGINAL FUNCTIONS=====================
class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class HensmanDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Dataloader when using minibatching with Stochastic Variational Inference.

    """
    def __init__(self, dataset, batch_sampler, num_workers):
        super().__init__(dataset, batch_sampler=_RepeatSampler(batch_sampler), num_workers=num_workers)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class SubjectSampler(Sampler):
    """
    Perform individual-wise sampling
    find T segments for each P based on the P index
    """
    def __init__(self, data_source, P, T):
        super(SubjectSampler, self).__init__(data_source)
        self.data_source = data_source
        self.P = P
        self.T = T

    def __iter__(self):
        r = np.arange(self.P)
        set_seed(999)
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [i for i in range(self.T*x, self.T*(x+1))], r))
        res = list(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return len(self.data_source)

class VaryingLengthSubjectSampler(Sampler):
    """
    Perform individual-wise sampling when individuals have varying number of temporal samples.
    outputs a list of random indices of the sujects that will iterate through the subject id sequence
    
    """
    def __init__(self, data_source, id_covariate):
        super(VaryingLengthSubjectSampler, self).__init__(data_source)
        self.data_source = data_source
        self.id_covariate = id_covariate

        def f(x):
            return int(x['label'][id_covariate].item()) #turn the subject ids into a list

        l = list(map(f, data_source))
        self.P = len(set(l))
        self.start_indices = [l.index(x) for x in list(OrderedDict.fromkeys(l))]
        self.end_indices = self.start_indices[1:] + [len(data_source)]

    def __iter__(self):
        set_seed(999)
        r = np.arange(self.P) 
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [(i, x) for i in range(self.start_indices[x], self.end_indices[x])], r))
        res = iter(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return self.P

class VaryingLengthBatchSampler(BatchSampler):
    """
    Perform batch sampling when individuals have varying number of temporal samples.
    Further specifying how the data should be fed into NN in what specific order
    BatchSample is a subclass to yield a list of batch indices order at a time to feed into NN
    input is a sample and a batch_size to fulfill
    
    """
    def __init__(self, sampler, batch_size):
        super(VaryingLengthBatchSampler, self).__init__(sampler, batch_size, False)
        assert isinstance(sampler, VaryingLengthSubjectSampler)#this is to make sure we use varyinglenthsubjectsampler as a samplying method

        self.sampler = sampler
        self.batch_size = batch_size

    #__len__ defined by the superclass

    def __iter__(self):
        batch = []
        batch_subjects = set()
        for idx, subj in self.sampler:
            if subj not in batch_subjects:
                if len(batch_subjects) == self.batch_size:
                    yield batch
                    batch = []
                    batch_subjects.clear()
                batch_subjects.add(subj)
            batch.append(idx)
        yield batch 
        # this function is to generate a batch of indices 
        # the method goes: if the subject not in the batch_subjects, then add, if it's in the subject pool, then we will add that index to the batch list
        # the mechanism to add subject is from telling the len(batch_subjects) equal to the batch size, meaning, if we sample enough and made up to the batch size, we don't have to sample anymore, otherwise, add more subjects into it untill batch_size is fulfilled

def batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, 
                            test_x, mu, zt_list, id_covariate, eps):
    """
    Perform batch predictions when individuals have varying number of temporal samples.
    
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]

    I_M = torch.eye(M, dtype=torch.double).to(device) 

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, prediction_x.shape[0], M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        # print(f'test_x, zt_list dimensions:{test_x.shape}/{zt_list.shape}')
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    
    K0zx = K0xz.transpose(-1, -2)
    # print(f'K0xz/K0zx shape:{K0xz.shape}/{K0zx.shape}') #K0xz/K0zx shape:torch.Size([32, 600, 60])/torch.Size([32, 60, 600])

    iB_st_list = []
    H = K0zz
    subjects = torch.unique(prediction_x[:, id_covariate]).tolist()
    iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for s in subjects:
        indices = prediction_x[:, id_covariate] == s
        x_st = prediction_x[indices]
        T = x_st.shape[0]
        I_T = torch.eye(T, dtype=torch.double).to(device)

        if isinstance(covar_module0, list):
            B_st = torch.zeros(latent_dim, T, T, dtype=torch.double).to(device)
            for i in range(latent_dim):
                B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
        else:
            stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=0)
            B_st = covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(dim=2)

        LB_st = torch.linalg.cholesky(B_st)
        iB_st = torch.cholesky_solve(I_T, LB_st)
        K0xz_st = K0xz[:, indices]
        K0zx_st = K0xz_st.transpose(-1, -2)
        iB_K0xz = torch.matmul(iB_st, K0xz_st)
        K0zx_iB_K0xz = torch.matmul(K0zx_st, iB_K0xz)
        H = H + K0zx_iB_K0xz
        iB_mu[:, indices] = torch.matmul(iB_st, mu[indices].T.unsqueeze(dim=2))
        iB_st_list.append(iB_st)

    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.linalg.solve(H,torch.matmul(K0zx, iB_mu))[0])
    iB_K0xz_iH_K0zx_iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for i, s in enumerate(subjects):
        indices = prediction_x[:, id_covariate] == s
        iB_K0xz_iH_K0zx_iB_mu[:, indices] = torch.matmul(iB_st_list[i], K0xz_iH_K0zx_iB_mu_st[:, indices])
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    # print(f'K0zx/mu_tilde shape:{K0zx.shape}/{mu_tilde.shape}') #K0zx/mu_tilde shape:torch.Size([32, 60, 600])/torch.Size([32, 600, 1])
    # print(f'K0zz and torch.matmul(K0zx, mu_tilde) shape:{K0zz.shape}/{torch.matmul(K0zx, mu_tilde).shape} ')#K0zz and torch.matmul(K0zx, mu_tilde) shape:torch.Size([32, 60, 60])/torch.Size([32, 60, 1]) 
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve(K0zz,torch.matmul(K0zx, mu_tilde) )[0])

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()
        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])
    # med_step=K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde
    # print(f'(K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde) shape:{med_step.shape}')
    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred



def batch_predict(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, mu, 
                 zt_list, P, T, id_covariate, eps):
    """
    Perform batch-wise predictions
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(prediction_x, [P, T, Q])

    mu = mu.T
    mu_st = torch.reshape(mu, [latent_dim, P, T, 1])

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, P*T, M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        B_st = torch.zeros(latent_dim, P, T, T).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(dim=2)).transpose(0, 1)
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    LB_st = torch.linalg.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [latent_dim, P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)
    K0zx = K0xz.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0zx, torch.reshape(iB_K0xz, [latent_dim, P*T, M]))
    H = K0zz + K0zx_iB_K0xz
    iB_mu = torch.matmul(iB_st, mu_st).view(latent_dim, -1, 1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.linalg.solve(H,torch.matmul(K0zx, iB_mu) )[0]).reshape(latent_dim, P, T, -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(latent_dim, -1, 1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve(K0zz,torch.matmul(K0zx, mu_tilde))[0])

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()

        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])
    
    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred

def predict(covar_module0, covar_module1, likelihood, train_xt, test_x, mu, z, P, T, id_covariate, eps):
    """
    Helper function to perform predictions.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = train_xt.shape[1]
    M = z.shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(train_xt, [P, T, Q])
    mu_st = torch.reshape(mu, [P, T, 1])

    K0xz = covar_module0(train_xt, z).evaluate()
    K0zz = covar_module0(z, z).evaluate() + eps * I_M
    K1_st = covar_module1(x_st, x_st).evaluate()
    K0Xz = covar_module0(test_x, z).evaluate()

    B_st = K1_st + I_T * likelihood.noise_covar.noise
    LB_st = torch.linalg.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0xz.T, torch.reshape(iB_K0xz, [P*T, M]))
    H = K0zz + K0zx_iB_K0xz

    iB_mu = torch.matmul(iB_st, mu_st).view(-1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.linalg.solve(H,torch.matmul(K0xz.T, iB_mu).unsqueeze(dim=1) )[0]).reshape(P, T, -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(-1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.linalg.solve( K0zz,torch.matmul(K0xz.T, mu_tilde).unsqueeze(dim=1))[0]).squeeze()

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(train_xt[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(test_x.shape[0], dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s
        K1Xx = covar_module1(test_x[indices], train_xt[mask]).evaluate()
        K1Xx_mu_tilde[indices] = torch.matmul(K1Xx, mu_tilde[mask])

    Z_pred = K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde

    return Z_pred

