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

def get_data_loader(src_path,tar_path,batch_size,NUM_WORKER):

    src_data=np.load(src_path)
    feat_arr_src=src_data['feature']
    label_arr_src=src_data['target_id']
    label_reg_arr_src=src_data['proficiency']
    feat=feat_arr_src
    label=label_arr_src
    label_reg=label_reg_arr_src

    print(feat_arr_src.shape)
    print(label_reg_arr_src.shape)
    train_loader_src,val_loader_src,test_loader_src=get_mv_data(feat,label, label_reg,  batch_size,NUM_WORKER)

    tar_data=np.load(tar_path)
    feat_arr_tar=tar_data['feature']
    label_arr_tar=tar_data['target_id']
    label_reg_arr_tar=tar_data['proficiency']
    print(feat_arr_tar.shape)
    print(label_reg_arr_tar.shape)
    feat=feat_arr_tar
    label=label_arr_tar
    label_reg=label_reg_arr_tar
    train_loader_tar,val_loader_tar,test_loader_tar=get_mv_data(feat,label, label_reg,  batch_size,NUM_WORKER)


    data_loader={
            'train_src':train_loader_src,
        'val_src':val_loader_src,
        'test_src':test_loader_src,
        'train_tar':train_loader_tar,
        'val_tar':val_loader_tar,
        'test_tar':test_loader_tar,

    }
    return data_loader



# ===================below is data processing for single variate=========================

#---- below is for desired lags----
class UserSepDataset_amp_sv_mask0(Dataset): #the old UserSepDataset_reg_sv
# this function is from https://github.com/seewoo5/KT
# this is a class of map-style dataset, you can access the data values using the __get_item__ via its indices or key and has a __len()__ attribute
    def __init__(self, name, na_fill_type,sample_infos,seq_size,target_var_pos,tag_pos, question_num, dataset_name,pred_type):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # list of (user_path, target_index)
        self._dataset_name = dataset_name
        self._seq_size=seq_size
        self._question_num=question_num
        self._target_var_pos=target_var_pos
        self._tag_pos=tag_pos
        self._pred_type=pred_type
        self._na_fill_type=na_fill_type

    def get_sequence(self, sample):
        PAD_INDEX = 0
        user_path, target_index = sample
        users_2018=['4768879', '1498688', '4785433', '4036445', '5025764', '4957214', '4869968','4747751','4834648', '5013917', '4784249', '3615463', '4745466'] 
        #the first 8 is geom 2018 users and last 5 is alg2 2018 users
        user_id=user_path.split('/')[-1].split('.')[0]
        # print(user_id)
        if user_id not in users_2018:
 
            with open(user_path, 'r') as f:
                # print(user_path)
                data = f.readlines()[1:] # header exists
                data = data[:target_index+1]
                user_data_length = len(data)
            if self._na_fill_type=='mask':
                pad_counts=0
                PAD_INDEX=0
            else:
                if user_data_length >= self._seq_size :
                    data = data[-(self._seq_size):]
                    pad_counts = 0
                else:
                    pad_counts = self._seq_size - user_data_length
                
                    if self._na_fill_type=='bfill':
                        first_line=data[0].rstrip().split(',')
                        PAD_INDEX=int(first_line[0])
                        
                    elif self._na_fill_type=='ffill':
                        last_line=data[-1].rstrip().split(',')
                        PAD_INDEX=int(last_line[0])
                    else:
                        PAD_INDEX=0  #the score_rate is on the 6th column
                    
            input_list=[]
            for idx, line in enumerate(data):
                line = line.rstrip().split(',')
                tag_id = int(line[self._tag_pos])#in reg, self._tag_pos= 2, in clf, self._tag_pos=0
               
                if self._pred_type=='reg':
                    label = float(line[self._target_var_pos])
                else:
                    label=int(line[self._target_var_pos])
              
                if idx == len(data) - 1:
                    last_label = label
                    target_id = tag_id
                else:
                    if label:
                        input_list.append(tag_id)
                    else:
                        input_list.append(tag_id + self._question_num)
            paddings = [PAD_INDEX] * pad_counts
            if len(input_list)>0 and last_label!=None and target_id!=None:
               
                assert len(input_list)>0, 'input is empty!'
                assert last_label!=None, 'label is empty!'
                assert target_id!=None, 'target_id is empty!'
                if self._na_fill_type=='mask':
                    input_list = input_list
                
                else:
                    assert len(input_list) == self._seq_size, "sequence size error"
                    if user_data_length >= self._seq_size :
                        input_list=paddings+input_list
                    else:
                        if self._na_fill_type!='ffill':
                            input_list = paddings + input_list
                        else:
                            input_list = input_list+paddings
                    
                        
            else:
                pass
                
            input_list=torch.Tensor(input_list).long()
            final_label=[last_label]
            if self._pred_type=='reg':
        
                final_target=[target_id] 
            else:
        
                final_target=[target_id-1]
            if self._na_fill_type!='mask':
                return {
                'label': torch.tensor(final_label),
                'input': input_list,
                'target_id': torch.tensor(final_target)
                }
            else:
        
                return (input_list,final_label,final_target)
    

    # def __repr__(self):
    #     return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])

class UserSepDataset_amp_sv(Dataset): #the old UserSepDataset_reg_sv
# this function is from https://github.com/seewoo5/KT
    def __init__(self, name, na_fill_type,sample_infos,seq_size,target_var_pos,tag_pos, question_num, dataset_name,pred_type):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # list of (user_path, target_index)
        self._dataset_name = dataset_name
        self._seq_size=seq_size
        self._question_num=question_num
        self._target_var_pos=target_var_pos
        self._tag_pos=tag_pos
        self._pred_type=pred_type
        self._na_fill_type=na_fill_type
        
    def get_sequence(self, sample):
        PAD_INDEX = 0
        user_path, target_index = sample
        users_2018=['4768879', '1498688', '4785433', '4036445', '5025764', '4957214', '4869968','4747751','4834648', '5013917', '4784249', '3615463', '4745466'] 
        #the first 8 is geom 2018 users and last 5 is alg2 2018 users
        user_id=user_path.split('/')[-1].split('.')[0]
        # print(user_id)
        if user_id not in users_2018:
 
            with open(user_path, 'r') as f:
                data = f.readlines()[1:] # header exists
                data = data[:target_index+1]
                user_data_length = len(data)

            if user_data_length > self._seq_size + 1:
                data = data[-(self._seq_size + 1):]
                pad_counts = 0
            else:
                pad_counts = self._seq_size + 1 - user_data_length
            
            input_list = []
            
            for idx, line in enumerate(data):
                line = line.rstrip().split(',')
                tag_id = int(line[self._tag_pos])#in reg, self._tag_pos= 2, in clf, self._tag_pos=0
                
                if self._pred_type=='reg':
                    label = float(line[self._target_var_pos])
                else:
                    label=int(line[self._target_var_pos])
              
                if idx == len(data) - 1:
                    last_label = label
                    target_id = tag_id
                else:
                    if label:
                        input_list.append(tag_id)
                    else:
                        input_list.append(tag_id + self._question_num)

            if self._na_fill_type=='bfill':
                first_line=data[0].rstrip().split(',')
                PAD_INDEX=int(first_line[0])
                
                paddings = [PAD_INDEX] * pad_counts
                input_list = paddings + input_list
            elif self._na_fill_type=='ffill':
                last_line=data[-1].rstrip().split(',')
                PAD_INDEX=int(last_line[0])
                paddings = [PAD_INDEX] * pad_counts
                input_list = input_list+paddings
            else:
                PAD_INDEX=0#the score_rate is on the 6th column
                paddings = [PAD_INDEX] * pad_counts
                input_list = paddings + input_list
            assert len(input_list) == self._seq_size, "sequence size error"
  
            if self._pred_type=='reg':
                final_label=torch.Tensor([last_label]).float()
                final_target=torch.Tensor([target_id ]).long()
            else:
                final_label=torch.Tensor([last_label]).long()
                final_target=torch.Tensor([target_id-1]).long()
        return {
            'label': final_label,
            'input': torch.Tensor(input_list).long(),
            'target_id': final_target
        }
    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])
# below version is the slightly improved  version

class UserSepDataset_amp_sv_mask(Dataset): #the old UserSepDataset_reg_sv
# this function is from https://github.com/seewoo5/KT
# this is a class of map-style dataset, you can access the data values using the __get_item__ via its indices or key and has a __len()__ attribute
    def __init__(self, name, na_fill_type,sample_infos,seq_size,target_var_pos,tag_pos, question_num, dataset_name,pred_type):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # list of (user_path, target_index)
        self._dataset_name = dataset_name
        self._seq_size=seq_size
        self._question_num=question_num
        self._target_var_pos=target_var_pos
        self._tag_pos=tag_pos
        self._pred_type=pred_type
        self._na_fill_type=na_fill_type

    def get_sequence(self, sample):
        PAD_INDEX = 0
        user_path, target_index = sample
        users_2018=['4768879', '1498688', '4785433', '4036445', '5025764', '4957214', '4869968','4747751','4834648', '5013917', '4784249', '3615463', '4745466'] 
        #the first 8 is geom 2018 users and last 5 is alg2 2018 users
        user_id=user_path.split('/')[-1].split('.')[0]
        # print(user_id)
        if user_id not in users_2018:
 
            with open(user_path, 'r') as f:
                # print(user_path)
                data = f.readlines()[1:] # header exists
                data = data[:target_index+1]
                user_data_length = len(data)
     
            if user_data_length >= self._seq_size:
                data = data[-(self._seq_size ):]
                pad_counts = 0
            else:
                if self._na_fill_type=='mask':
                    pad_counts=0
                    PAD_INDEX=0
                else:

                    pad_counts = self._seq_size - user_data_length
                    if self._na_fill_type=='bfill':
                        first_line=data[0].rstrip().split(',')
                        PAD_INDEX=int(first_line[0])
                        
                    elif self._na_fill_type=='ffill':
                        last_line=data[-1].rstrip().split(',')
                        PAD_INDEX=int(last_line[0])
                    else:
                        PAD_INDEX=0  #the score_rate is on the 6th column
                        
            input_list=[]
            for idx, line in enumerate(data):
                line = line.rstrip().split(',')
                tag_id = int(line[self._tag_pos])#in reg, self._tag_pos= 2, in clf, self._tag_pos=0, the tag_id-1 is because it's 1-indexed instead of 0-indexed
               
                if self._pred_type=='reg':
                    label = float(line[self._target_var_pos])
                else:
                    label=int(line[self._target_var_pos])
                input_list.append(tag_id)
                if idx == len(data) - 1:
                    last_label = label
                    target_id = tag_id
            paddings = [PAD_INDEX] * pad_counts
            # if len(input_list)>0 and last_label!=None and target_id!=None:
               
            #     assert len(input_list)>0, 'input is empty!'
            #     assert last_label!=None, 'label is empty!'
            #     assert target_id!=None, 'target_id is empty!'
        
                    
            if user_data_length >= self._seq_size:
                    input_list=paddings+input_list
            else:
                if self._na_fill_type=='mask':
                    input_list = input_list
                else:
                    if self._na_fill_type!='ffill':
                        input_list = paddings + input_list
                    else:
                        input_list = input_list+paddings
                
                    # assert len(input_list) == self._seq_size, "sequence size error"
        
                
            input_list=torch.Tensor(input_list).long()
            final_label=[last_label]
            final_target=[target_id]
     
            if self._na_fill_type!='mask':
                return {
                'label': torch.tensor(final_label),
                'input': input_list,
                'target_id': torch.tensor(final_target)
                }
            else:
        
                return (input_list,final_label,final_target)
    

    # def __repr__(self):
    #     return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])


def get_data_user_sep_sv_no_shift(data_path,stu_num,shift_step):
    # this function is from https://github.com/seewoo5/KT
    # almost same as get_sample_info
    # for user separated format data
    sample_infos = []
    # get list of all files
    user_path_list = os.listdir(data_path)
    num_of_users = len(user_path_list)

    for i, user_path in enumerate(user_path_list):
        user_data=[]
        # print(i,user_path)
        if stu_num!=None:
            
            if i<=stu_num:
                with open(data_path + user_path, 'rb') as f:
                    lines = f.readlines()
                    
                    lines = lines[1:]
                
                    if lines:
                        num_of_interactions = len(lines)
                        print(f'has {num_of_interactions} rows')
                        for end_index in range(num_of_interactions):
                            user_data.append((data_path + user_path, end_index))
                        # user_data=user_data[-shift_step:]
                        # print(f"user data last element: {user_data[-1]}")
                        sample_infos.extend(user_data)
        else:
            with open(data_path + user_path, 'rb') as f:
                    lines = f.readlines()
                    
                    lines = lines[1:]
                    if lines:
                        num_of_interactions = len(lines)
                        for end_index in range(num_of_interactions):
                            user_data.append((data_path + user_path, end_index))
                        # user_data=user_data[-shift_step:]
                        sample_infos.extend(user_data)
        # print(f'last sample of sample_infos: {sample_infos[-1]}')
    return sample_infos, num_of_users



def get_data_user_sep_sv_shift(data_path,stu_num,shift_step):
    # this function is from https://github.com/seewoo5/KT
    # almost same as get_sample_info
    # for user separated format data
    sample_infos = []
    # get list of all files
    user_path_list = os.listdir(data_path)
    num_of_users = len(user_path_list)

    for i, user_path in enumerate(user_path_list):
        user_data=[]
        if stu_num!=None:
            
            if i<=stu_num:
                with open(data_path + user_path, 'rb') as f:
                    lines = f.readlines()
                    
                    lines = lines[1:]
                
                    if lines:
                        num_of_interactions = len(lines)
                        for end_index in range(num_of_interactions):
                            user_data.append((data_path + user_path, end_index))
                        user_data=user_data[-shift_step:]
                        sample_infos.extend(user_data)
        else:
            with open(data_path + user_path, 'rb') as f:
                    lines = f.readlines()
                    
                    lines = lines[1:]
                    if lines:
                        num_of_interactions = len(lines)
                        for end_index in range(num_of_interactions):
                            user_data.append((data_path + user_path, end_index))
                        user_data=user_data[-shift_step:]
                        sample_infos.extend(user_data)
        # print(f'last sample of sample_infos: {sample_infos[-1]}')
    return sample_infos, num_of_users

# pytorch's way of padding a sequence with a certain value
def pad_collate(batch): #collate: collect and combine a sequences, feed into collate_fn which usually prefers a mapstyle dataset
    input_list,labels,target_ids=[],[],[]
    for data in batch:
        if data!=None:
            input, label,target_id = data #one star before the variable means collects all the positional arguments , equal to (1,2,3,4)
            input_list.append(input)
            labels.extend(label)
            target_ids.extend(target_id)
        #   (input_list,labels,target_ids)=zip(*batch)
    input_lens = [len(x) for x in input_list]
    # print(f'input before converting to tensor :{input_list.shape}') #should return tensor
    # print(f'label before converting to tensor :{labels}') #should return a list in tensor
    # print(f'target_id before converting to tensor :{target_ids}')

    labels=np.array(labels)
    target_ids=np.array(target_ids)
    # #   print(f'label after reshaping  to tensor :{labels}')
    # #   print(f'target_id after reshaping  to tensor :{target_ids}')
    labels=torch.from_numpy(labels).view(-1,1)
    target_ids=torch.from_numpy(target_ids).view(-1,1)

#   input=torch.from_numpy(input)
#   print(f'input after reshaping to tensor :{input}')
    input_pad = pad_sequence(input_list, batch_first=True, padding_value=0) #pad sequence needs to have same length on dim1
    return input_pad, labels,target_ids,input_lens


def load_data_sv_amp_shift_mask(root_path,na_fill_type, domain, question_num, split_type, stu_num,target_var_pos,tag_pos, shift_step,batch_size,seqL,NUM_WORKER,pred_type,phase):
    user_base_path = f'{root_path}/{domain}/processed/1'
    domain_split_type=[]
    # question_num=QUESTION_NUM[domain]
    split_type_data_path=f'{user_base_path}/{split_type}/'
    print(split_type_data_path)
    if shift_step!=None:
        split_type_sample_infos, num_of_split_type_user = get_data_user_sep_sv_shift(split_type_data_path,stu_num,shift_step)
    else:
        split_type_sample_infos, num_of_split_type_user = get_data_user_sep_sv_no_shift(split_type_data_path,stu_num,shift_step)
    # split_type_data =UserSepDataset_amp_sv(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type) #the return is a list of dictionaries b/c the get_sequence function is only per index
    if na_fill_type!='mask':
        split_type_data =UserSepDataset_amp_sv(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type)
        data_loader = torch.utils.data.DataLoader(dataset=split_type_data, batch_size=batch_size, shuffle=phase=='src', drop_last=True, num_workers=NUM_WORKER)
    else:
        split_type_data =UserSepDataset_amp_sv_mask(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type)
        data_loader = torch.utils.data.DataLoader(dataset=split_type_data, batch_size=batch_size, shuffle=phase=='src', drop_last=True, 
                                                  num_workers=NUM_WORKER, collate_fn=pad_collate)
    print(f'data loader for {split_type} len:{len(data_loader)}!')
    return data_loader


def load_data_sv_amp_shift(root_path,na_fill_type, domain, question_num, split_type, stu_num,target_var_pos,tag_pos, shift_step,batch_size,seqL,NUM_WORKER,pred_type,phase):
    user_base_path = f'{root_path}/{domain}/processed/1'
    domain_split_type=[]
    # question_num=QUESTION_NUM[domain]
    split_type_data_path=f'{user_base_path}/{split_type}/'
    print(split_type_data_path)
    split_type_sample_infos, num_of_split_type_user = get_data_user_sep_sv_shift(split_type_data_path,stu_num,shift_step)
    split_type_data =UserSepDataset_amp_sv(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type) #the return is a list of dictionaries b/c the get_sequence function is only per index
    
    data_loader = torch.utils.data.DataLoader(dataset=split_type_data, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=NUM_WORKER)
    print(f'data loader for {split_type} len:{len(data_loader)}!')
    return data_loader


class UserSepDataset_amp_sv_vae(Dataset): #the old UserSepDataset_reg_sv
# this function is from https://github.com/seewoo5/KT
# this is a class of map-style dataset, you can access the data values using the __get_item__ via its indices or key and has a __len()__ attribute
    def __init__(self, name, na_fill_type,sample_infos,seq_size,target_var_pos,tag_pos, question_num, dataset_name,pred_type):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # list of (user_path, target_index)
        self._dataset_name = dataset_name
        self._seq_size=seq_size
        self._question_num=question_num 
        self._target_var_pos=target_var_pos
        self._tag_pos=tag_pos
        self._pred_type=pred_type
        self._na_fill_type=na_fill_type

    def get_sequence(self, sample):
        PAD_INDEX = 0
        user_path, target_index = sample
        users_2018=['4768879', '1498688', '4785433', '4036445', '5025764', '4957214', '4869968','4747751','4834648', '5013917', '4784249', '3615463', '4745466'] 
        #the first 8 is geom 2018 users and last 5 is alg2 2018 users
        user_id=user_path.split('/')[-1].split('.')[0]
        # print(user_id)
        if user_id not in users_2018:
 
            with open(user_path, 'r') as f:
                # print(user_path)
                data = f.readlines()[1:] # header exists
                data = data[:target_index+1]
                user_data_length = len(data)
     
            input_list=[]
            for idx, line in enumerate(data):
                line = line.rstrip().split(',')
                tag_id = int(line[self._tag_pos])#in reg, self._tag_pos= 2, in clf, self._tag_pos=0
               
                if self._pred_type=='reg':
                    label = float(line[self._target_var_pos])
                else:
                    label=int(line[self._target_var_pos])
                if tag_id:
                # if label:
                    input_list.append(tag_id)
                # else:
                    # input_list.append(tag_id + self._question_num)
                # input_list.append(tag_id)
                if idx == len(data) - 1:
                    last_label = label
                    target_id = tag_id

            start_end={'sos':self._question_num,'eos':self._question_num+1}#<---
            source=[start_end['sos']]+input_list#<---
            # print(source)
            source=torch.tensor(source)
            target=input_list+[start_end['eos']]#<---
            # print(target)
            target=torch.tensor(target) #if using torch.tensor, yes it will infer the dtype, but it will also put it on the cpu
            # source_len=len(source)#<---

                
            input_list=torch.Tensor(input_list).long()
            final_label=[last_label]
            final_target=[target_id]
            # output={'clf_output':(input_list,final_label,final_target),
            #         'vae_output':(source,target)}
        # (input_list,final_label,final_target),
            return  (source,target)##<---
    

    # def __repr__(self):
    #     return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])



def pad_collate_vae(batch): #collate: collect and combine a sequences, feed into collate_fn which usually prefers a mapstyle dataset
    source_list,target_list=[],[]
    for data in batch:
        if data!=None:
            source,target = data #one star before the variable means collects all the positional arguments , equal to (1,2,3,4)
            source_list.append(source)
            target_list.append(target)

    input_lens = [len(x) for x in source_list]

    source_pad = pad_sequence(source_list, batch_first=True, padding_value=0) #pad sequence needs to have same length on dim1
    target_pad = pad_sequence(target_list, batch_first=True, padding_value=0)
    return source_pad, target_pad,input_lens


def load_data_sv_amp_shift_vae(root_path,na_fill_type, domain, question_num, split_type, stu_num,target_var_pos,tag_pos, shift_step,batch_size,seqL,NUM_WORKER,pred_type,phase):
    user_base_path = f'{root_path}/{domain}/processed/1'
    domain_split_type=[]
    # question_num=QUESTION_NUM[domain]
    split_type_data_path=f'{user_base_path}/{split_type}/'
    print(split_type_data_path)
    split_type_sample_infos, num_of_split_type_user = get_data_user_sep_sv_shift(split_type_data_path,stu_num,shift_step)
    
    if na_fill_type!='mask':
        split_type_data =UserSepDataset_amp_sv(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type)
        data_loader = torch.utils.data.DataLoader(dataset=split_type_data, batch_size=batch_size, shuffle=phase=='src', drop_last=True, num_workers=NUM_WORKER)
    else:
        vae_data=UserSepDataset_amp_sv_vae(split_type, na_fill_type,split_type_sample_infos,seqL,target_var_pos,tag_pos,question_num, domain,pred_type)

        vae_data_loader=torch.utils.data.DataLoader(dataset=vae_data, batch_size=batch_size, shuffle=True, drop_last=True, 
                                                  num_workers=NUM_WORKER, collate_fn=pad_collate_vae)

    print(f'vae data loader for {split_type} len:{len(vae_data_loader)}!')
    return vae_data_loader



# ----------------------SPLIT SEQ BY USER------------

def split_user_seq_kt_v2(data, cadence, out_dir):

    kfd=KFold(n_splits=5)
    drop_cols=['attempt_time', 'score', 'max_score']
    data_n=data.drop(drop_cols, axis=1)
    col_reord=['event_time',  'student_id','quest_ref_1',
       'item_ref_1', 'usmoid_1','correct','score_rate', 'question_type_1','seq_number', 'bktcount', 'proficiency','assessment_duration', 'count','quest_difficulty', 'item_difficulty', 'usmo_difficulty']
    data_n=data_n[col_reord]
    for i, (train_index, test_index) in enumerate(kfd.split(data_n['student_id'].unique())):
        if i>=0:
            print('====')
            train_stu_list=data_n['student_id'].unique()[train_index]
            train_stu, val_stu=train_test_split(train_stu_list, test_size=0.1,random_state=88)
            print(f'  Fold {i+1} has {len(train_stu)} train users,{len(val_stu)} val users')
            train_by_user=data_n.loc[data_n['student_id'].isin(train_stu)]
            print(f'  train_by_user shape {train_by_user.shape}')
            val_by_user=data_n.loc[data_n['student_id'].isin(val_stu)]
            print(f'  val_by_user shape {val_by_user.shape}')
            train_by_user=train_by_user.sort_values(['student_id','event_time'])
            val_by_user=val_by_user.sort_values(['student_id','event_time'])
            total_steps=0
            train_out_dir=f'{out_dir}/processed/{i+1}/train'
            val_out_dir=f'{out_dir}/processed/{i+1}/val'
            test_out_dir=f'{out_dir}/processed/{i+1}/test'
            
            os.makedirs(train_out_dir,exist_ok=True)
            os.makedirs(val_out_dir,exist_ok=True)
            os.makedirs(test_out_dir,exist_ok=True)
            for m, u in enumerate(list(train_by_user['student_id'].unique())):
                stu=train_by_user[train_by_user['student_id']==u]
                time_step=stu.shape[0]
                total_steps+=time_step
                if m%cadence==0:
                    print(f'  train_by_user data_n split: {m}th User with ID {u} has {time_step}-step sequence')
                    
                stu.to_csv(f'{train_out_dir}/{u}.csv',index=False)
            print(f'>>fold {i+1} train data_n has average time steps of {round(total_steps/len(train_stu),2)}<<')
            total_steps=0
            for n, u in enumerate(list(val_by_user['student_id'].unique())):
        #         print(f'  val_by_user data_n split by user: {n},{u}')
                stu=val_by_user[val_by_user['student_id']==u]
        #         display(stu)
                time_step=stu.shape[0]
                total_steps+=time_step
                if n%cadence==0:
                    print(f'  val_by_user data_n split: {n}th User with ID {u} has {time_step}-step sequence')
                stu.to_csv(f'{val_out_dir}/{u}.csv',index=False)
            print(f'>>fold {i+1} val data_n has average time steps of {round(total_steps/len(val_stu),2)}<<')
            test_stu_list=data_n['student_id'].unique()[test_index]
            print(f'  fold {i+1} has {len(test_stu_list)} test users')
            test_by_user=data_n.loc[data_n['student_id'].isin(test_stu_list)]
            print(f'  test_by_user shape {test_by_user.shape}')
            test_by_user=test_by_user.sort_values(['student_id','event_time'])
        #     test_by_user=test_by_user[['student_id','quest_ref_index','correctness']]
            total_steps=0
            for k, u in enumerate(list(test_by_user['student_id'].unique())):
        #         print(f'  test_by_user data_n split by user: {k},{u}')
                stu=test_by_user[test_by_user['student_id']==u]
                time_step=stu.shape[0]
                total_steps+=time_step
                if k%cadence==0:
                    print(f'  test_by_user data_n split: {k}th User with ID {u} has {time_step}-step sequence')
        #         display(stu)
                stu.to_csv(f'{test_out_dir}/{u}.csv',index=False)
            print(f'>>fold {i+1} test data_n has average time steps of {round(total_steps/len(test_stu_list),2)}<<')


# ===================below is data processing for multi variate=========================



class mv_data_set(Dataset):
    def __init__(self, na_fill_type, df_feature, df_label, df_label_reg,feat_lens, t=None):

        assert len(df_feature) == len(df_label)
        assert len(df_feature) == len(df_label_reg)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature=df_feature
        self.df_label=df_label
        self.df_label_reg = df_label_reg
        self._feat_lens=feat_lens
        self._na_fill_type=na_fill_type

        self.T=t
        self.df_feature=torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label=torch.tensor(
            self.df_label, dtype=torch.float32)
        self.df_label_reg=torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        if self._na_fill_type=='mask':
            sample, target, label_reg,feat_lens =self.df_feature[index], self.df_label[index], self.df_label_reg[index],self._feat_lens[index]
            return sample, target, label_reg,feat_lens
        else:
            sample, target, label_reg =self.df_feature[index], self.df_label[index], self.df_label_reg[index]
            return sample, target, label_reg
        # if self.T:
        #     return self.T(sample), target
        # else:
        #     return sample, target, label_reg,feat_lens

    def __len__(self):
        return len(self.df_feature)


def get_mv_data(is_predict,feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER):

    ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    feat=feat.reshape(-1, feat.shape[2])
    scaler=MinMaxScaler()
    feat=scaler.fit_transform(feat)
    feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
    if is_predict:
        predict_set=mv_data_set(feat, label_arr, label_reg_arr)
        predict_loader=torch.utils.data.DataLoader(dataset=predict_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        return predict_loader
    else:
        feat_train,feat_test=train_test_split(feat,test_size=0.2,random_state=999)
        feat_train,feat_val=train_test_split(feat_train,test_size=0.1,random_state=999)
        label_train,label_test=train_test_split(label_arr,test_size=0.2,random_state=999)
        label_train,label_val=train_test_split(label_train,test_size=0.1,random_state=999)
        label_reg_train,label_reg_test=train_test_split(label_reg_arr,test_size=0.2,random_state=999)
        label_reg_train,label_reg_val=train_test_split(label_reg_train,test_size=0.1,random_state=999)
        train_set=mv_data_set(feat_train, label_train, label_reg_train)
        test_set=mv_data_set(feat_test, label_test, label_reg_test)
        val_set=mv_data_set(feat_val, label_val, label_reg_val)

        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=NUM_WORKER)
        test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        val_loader=torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        print(f'train/val/test loader length:{len(train_loader)}/{len(val_loader)}/{len(test_loader)}')
        return train_loader,val_loader,test_loader
 
def get_mv_data_mask(is_predict,na_fill_type,feat_lens,feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER):
    if na_fill_type!='mask':

        ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
        feat=feat.reshape(-1, feat.shape[2])
        scaler=MinMaxScaler()
        feat=scaler.fit_transform(feat)
        feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
        feat_lens_train,feat_lens_test,feat_lens_val=None,None,None
    else:
        
        feat_lens_train,feat_lens_test=train_test_split(feat_lens,test_size=0.2,random_state=999)
        feat_lens_train,feat_lens_val=train_test_split(feat_lens_train,test_size=0.1,random_state=999)
    if is_predict:
        predict_set=mv_data_set(na_fill_type, feat, label_arr, label_reg_arr,feat_lens)
        predict_loader=torch.utils.data.DataLoader(dataset=predict_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        return predict_loader
    else:

        
        feat_train,feat_test=train_test_split(feat,test_size=0.2,random_state=999)
        feat_train,feat_val=train_test_split(feat_train,test_size=0.1,random_state=999)

        label_train,label_test=train_test_split(label_arr,test_size=0.2,random_state=999)
        label_train,label_val=train_test_split(label_train,test_size=0.1,random_state=999)
        label_reg_train,label_reg_test=train_test_split(label_reg_arr,test_size=0.2,random_state=999)
        label_reg_train,label_reg_val=train_test_split(label_reg_train,test_size=0.1,random_state=999)

        train_set=mv_data_set(na_fill_type,feat_train, label_train, label_reg_train,feat_lens_train)
        test_set=mv_data_set(na_fill_type,feat_test, label_test, label_reg_test,feat_lens_test)
        val_set=mv_data_set(na_fill_type,feat_val, label_val, label_reg_val,feat_lens_val)
    
        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=NUM_WORKER)
        test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        val_loader=torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=NUM_WORKER)
        
        print(f'train/val/test loader length:{len(train_loader)}/{len(val_loader)}/{len(test_loader)}')
        return train_loader,val_loader,test_loader
 

def get_mv_data_int(feat,label_arr, label_reg_arr,  batch_size,NUM_WORKER):

    ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    feat=feat.reshape(-1, feat.shape[2])
    scaler=MinMaxScaler()
    feat=scaler.fit_transform(feat)
    feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    feat_train,feat_test=train_test_split(feat,test_size=0.2,random_state=999)
    feat_train,feat_val=train_test_split(feat_train,test_size=0.1,random_state=999)
    label_train,label_test=train_test_split(label_arr,test_size=0.2,random_state=999)
    label_train,label_val=train_test_split(label_train,test_size=0.1,random_state=999)
    label_reg_train,label_reg_test=train_test_split(label_reg_arr,test_size=0.2,random_state=999)
    label_reg_train,label_reg_val=train_test_split(label_reg_train,test_size=0.1,random_state=999)
    train_set=mv_data_set(feat_train, label_train, label_reg_train)
    test_set=mv_data_set(feat_test, label_test, label_reg_test)
    val_set=mv_data_set(feat_val, label_val, label_reg_val)

    train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKER)
    test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)
    val_loader=torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)
    print(f'train/val/test loader length:{len(train_loader)}/{len(val_loader)}/{len(test_loader)}')
    dataloaders={
        'train':train_loader,
        'test':test_loader,
        'val':val_loader,
    }
    return feat_train,  label_reg_train,feat_test,  label_reg_test,dataloaders
# --------convert users to npz data: not from per user data-------



def convert_npz_lagged_bkt_shift_mask(save_dir,data,na_fill_type,seq_size,shift_step, remove_feature_list, test_feature_list,course_name,num_stu,marker,model_transform,save_flag):

    # drop_cols=['attempt_time', 'score', 'max_score']
    # data_n=data.drop(drop_cols, axis=1)
    # col_reord=['event_time',  'student_id','quest_ref_1',
    #    'item_ref_1', 'usmoid_1','correct','score_rate', 'question_type_1','seq_number', 'bktcount', 'proficiency','assessment_duration', 'count','quest_difficulty', 'item_difficulty', 'usmo_difficulty']
    
    data=data.sort_values(by=['student_id','event_time','seq_number'])
    lb=LabelEncoder()
    feature='event_time'
    data[feature]=data[feature].astype(str)
    data[feature+'_1']=lb.fit_transform(data[feature])
    new_label=list(data[feature+'_1'].unique())
    orig_label=lb.inverse_transform(new_label)

    map_dict={}
    for i, nl in enumerate(new_label):
        map_dict[nl]=orig_label[i]
    # display(map.head())
    data.drop(feature,axis=1,inplace=True)
    data=data.rename(columns={f'{feature}_1':feature})
    orig_cols=data.columns
    feat_cols=[ 'question_type_1','seq_number', 'bktcount','assessment_duration', 'count','quest_difficulty', 'item_difficulty', 'usmo_difficulty']
    if remove_feature_list:
        for f in remove_feature_list:
            feat_cols.remove(f)

    if test_feature_list:
        feat_cols=test_feature_list
    #     data_n=data[['event_time','student_id','quest_ref_1','score_rate','seq_number']+feat_cols]
    #     data_n=data_n.loc[:,~data_n.columns.duplicated()]
    #     # display(data_n.head())

    # else:
    data_n=data
    print(f'data shape: {data_n.shape}')
    time_step_list=[]
    feat_col_len=len(feat_cols)
    feat_arr_all=np.zeros([1,feat_col_len],dtype=float)
    feat_arr_all_mask=[]
    feat_arr_all_full=[]
    # feat_full_df=pd.DataFrame()
    label_arr_all,label_reg_arr_all=[],[]


    
    if num_stu:
        stu_uni=list(data_n['student_id'].unique())[:num_stu]
    else:
        stu_uni=list(data_n['student_id'].unique())
        print(f'there are {len(stu_uni)} unique students')
    cnt=0
    cnt_null=0
    for i, student_id in enumerate(stu_uni):
     
        student_df=data_n[data_n['student_id']==student_id]
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
        
        # map_dict=None
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
    return feat_arr_all,label_arr_all, label_reg_arr_all,feat_lens,processed_data





def convert_npz_flat(data,seq_size, len_win, course_name,num_stu,marker):

    drop_cols=['mth_course', 'score_obtained',
        'score_possible', 'school_id','has_ipr_mth_data_four','has_ipr_data_four','has_d2l_data_four','has_d2l_mth_data_four','seq']
    data['student_id']=data['student_id'].astype(int)
    data['usmoid']=data['usmoid'].fillna(0).astype(int)
    data_n=data.drop(drop_cols, axis=1)
    for col in data_n.columns[4:20]:
        data_n[col]=data_n[col].round(3)
    data_n.iloc[:,4:20]=data_n.iloc[:,4:20].fillna(0)
    data_n.iloc[:,2]=data_n.iloc[:,2].fillna(0)
    print(data_n.shape)
    display(data_n.head())
    time_step_list=[]
    feat_arr,label_arr,label_reg_arr=[],[],[]
    for i, student_id in enumerate(data_n['student_id'].unique()):
        if i<num_stu:
            student_df=data_n[data_n['student_id']==student_id]
            time_step=student_df.shape[0]
            time_step_list.append(time_step)
            if i%500==0:print(i, student_id, time_step)
            if time_step<seq_size:
                pad_counts=seq_size-time_step
                if i%500==0:print(f'need to pad {pad_counts} counts')
            else:
                student_df=student_df[-seq_size:]
                pad_counts = 0
                if i%500==0:print(f'no need to pad !')
            paddings=[0]*pad_counts
            padding_matrix=np.zeros([pad_counts,16],dtype=int)
            feat=student_df[student_df.columns[4:20]].values
            feat=np.concatenate((padding_matrix, feat),axis=0)
       
            if i%500==0:print(f'feat.shape after padding: {feat.shape}')
            label=student_df['usmoid'].tolist()
            label=np.array([paddings+label]).reshape(-1,1)
     
            label_reg=student_df['proficiency'].tolist()
            label_reg=np.array([paddings+label_reg]).reshape(-1,1)  
            if i==0:
            
                feat_arr=feat
                label_arr=label
                label_reg_arr=label_reg
            else:
        
                feat_arr=np.vstack((feat_arr,feat))
                
                label_arr=np.vstack((label_arr,label))
                
                label_reg_arr=np.vstack((label_reg_arr,label_reg))

    feat_arr=feat_arr.reshape(int(feat_arr.shape[0]/len_win),len_win,16)
    label_arr=label_arr[::len_win].squeeze(1)
    label_reg_arr=label_reg_arr[::len_win].squeeze(1)


    np.savez(f'TRANS_DATA/{course_name}_{num_stu}stu_seqL{seq_size}_lenWin{len_win}_{marker}.npz',feature=feat_arr,target_id=label_arr, proficiency=label_reg_arr)
    print(f'average time step per student:{np.mean(time_step_list)}')    
    return feat_arr,label_arr, label_reg_arr


# ===rewrite data_loader func=====


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


def combine_feat_covar_npz_old(feat_cols,covar_feat_cols,na_fill_type,seq_size,data_file,course_name,num_stu,marker,save_flag):
    """
    the root dir for this func should be under Longitudinal-VAE
    """
    df=pd.read_csv(data_file)
    df=df.fillna(0)
    print(f'data has total shape:{df.shape}')
    flag=f'{marker}_outcome_labels'
    save_dir='data/k12/pre_train/label_maps_wo_nas'
    os.makedirs(save_dir,exist_ok=True)
    enc_list=['event_time','SPECIAL_ED','GRADE_LEVEL','LE_SCHOOL_ID','FREE_REDUCED_LUNCH','GIFTED_TALENTED', 'student_id']
    map_dat=pd.DataFrame()
    for feat_name in enc_list:
        df,map_df=encode_feature(df,save_dir,feat_name,flag)
        map_dat=map_dat.append(map_df)
    map_dat.to_csv(f'{save_dir}/all_label_coding_keys.csv',index=False)
    print(f'df columns:{df.columns}')
    feat_col_len,covar_feat_len=len(feat_cols),len(covar_feat_cols)
    is_covar=False
    feat_arr,mask_arr=generate_npz_scale_lvae(is_covar,feat_cols,df,na_fill_type,seq_size)
    is_covar=True
    covar_arr,covar_mask_arr=generate_npz_scale_lvae(is_covar,covar_feat_cols,df,na_fill_type,seq_size)
    if save_flag:
        
        save_path=f'data/k12/split_data/{course_name}_{num_stu}stu_seqL{seq_size}_{feat_col_len}features_{covar_feat_len}covar_features_{marker}_{na_fill_type}.npz'
        print(f'....saving to {save_path}.... ')
        np.savez(save_path,data_readings=feat_arr,outcome_attrib=covar_arr,data_mask=mask_arr,outcome_mask=covar_mask_arr)
    return feat_arr,covar_arr,mask_arr,covar_mask_arr

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
