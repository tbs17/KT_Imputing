from multiprocessing import allow_connection_pickling
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np

from functools import partial
# class PhysionetDataset(Dataset):
#     """
#     Dataset definition for the Physionet Challenge 2012 dataset.
#     generating a map-style dataset
#     """

#     def __init__(self, data_file, root_dir, transform=None):
#         data = np.load(os.path.join(root_dir, data_file),allow_pickle=True)
#         self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
#         self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
#         self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
#         self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_source)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         patient_data = self.data_source[idx, :]
#         patient_data = torch.from_numpy(np.array(patient_data,dtype=np.float32))

#         mask = self.mask_source[idx, :]
#         mask = torch.from_numpy(np.array(mask, dtype='uint8'))
# # original:mask = np.array(mask, dtype='uint8')
#         label = self.label_source[idx, :]
#         # label[8] = label[8] - 24
#         label_mask = self.label_mask_source[idx, :]
#         label = torch.Tensor(label)
#         # label = torch.Tensor(np.concatenate((label, label_mask)))

#         if self.transform:
#             patient_data = self.transform(patient_data)

#         sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
#         return sample

class PhysionetDataset_old(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    generating a map-style dataset
    """

    def __init__(self, data_file, root_dir,T,  transform=None):
        data = np.load(os.path.join(root_dir, data_file),allow_pickle=True)
        if T:
            dim0=data['data_readings'].shape[0]//T #// is called integer divide or integer divide
            new_data=data['data_readings'].reshape(-1)[:dim0*T*data['data_readings'].shape[-1]]
            self.data_source = new_data.reshape(dim0,T,data['data_readings'].shape[-1])
            # new_label=data['outcome_attrib'].reshape(-1)[:dim0*T*data['outcome_attrib'].shape[-1]]
            # self.label_source = new_label.reshape(dim0,T,data['outcome_attrib'].shape[-1])
            new_mask=data['data_mask'].reshape(-1)[:dim0*T*data['data_mask'].shape[-1]]
            self.mask_source = new_mask.reshape(dim0,T,data['data_mask'].shape[-1])
            # new_o_mask=data['outcome_mask'].reshape(-1)[:dim0*T*data['outcome_mask'].shape[-1]]
            # self.label_mask_source = new_o_mask.reshape(dim0,T,data['outcome_mask'].shape[-1])
     
        else:
            self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
            self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
        self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
        self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx]
        patient_data = torch.from_numpy(np.array(patient_data,dtype=np.float32))

        mask = self.mask_source[idx]
        mask = torch.from_numpy(np.array(mask, dtype='uint8'))
# original:mask = np.array(mask, dtype='uint8')
        label = self.label_source[idx]
        # label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx]
        label = torch.Tensor(label)
        # label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample

class PhysionetDataset(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    generating a map-style dataset
    data_format==3d, is for training with VAE
    data_format=2d, is for LVAE training
    """

    def __init__(self, data_file, root_dir,data_format,  transform=None):
        data = np.load(os.path.join(root_dir, data_file),allow_pickle=True)
        self.data_format=data_format
        if self.data_format=='3d':

            self.data_source = data['data_readings']
            self.mask_source = data['data_mask']
            self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
            self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
     
        else:
            self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
            self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
            self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
            self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx]
        patient_data = torch.from_numpy(np.array(patient_data,dtype=np.float32))

        mask = self.mask_source[idx]
        mask = torch.from_numpy(np.array(mask, dtype='uint8'))
# original:mask = np.array(mask, dtype='uint8')
        label = self.label_source[idx]
        # label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx]
        label = torch.Tensor(label)
        # label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample



class PhysionetDataset_old1(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    generating a map-style dataset
    """

    def __init__(self, data_file, root_dir,  transform=None):
        data = np.load(os.path.join(root_dir, data_file),allow_pickle=True)
 
        self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
        self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
        self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
        self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx]
        patient_data = torch.from_numpy(np.array(patient_data,dtype=np.float32))

        mask = self.mask_source[idx]
        mask = torch.from_numpy(np.array(mask, dtype='uint8'))
# original:mask = np.array(mask, dtype='uint8')
        label = self.label_source[idx]
        # label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx]
        label = torch.Tensor(label)
        # label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDataset(Dataset):
    """
    Dataset definition for the rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 784.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array([digit])

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the rotated MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 28 x 28.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array(digit)
        digit = digit.reshape(28, 28)
        digit = digit[..., np.newaxis]

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class HealthMNISTDataset(Dataset):
    """
    Dataset definition for the Health MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array([digit], dtype='uint8')

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        # changed
        # time_age 6,  disease_time4,  subject0,  gender5,  disease3,  location7
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)] 
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(36, 36)
        digit = digit[..., np.newaxis]

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        # CHANGED
        # time_age,  disease_time,  subject,  gender,  disease,  location(digit and angle is ignored)
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample
