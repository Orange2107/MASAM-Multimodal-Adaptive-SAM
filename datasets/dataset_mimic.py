import os
import pickle
from pathlib import Path
import ast
import re

import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def set_seed(_seed):
    global seed
    seed = _seed
    print(f"set the seed of dataloader as :{seed}")
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MultiModalMIMIC(Dataset):
    def __init__(self, seed,data_root, fold, partition, cxr_img_root, task,
                 time_limit=48, normalization='robust_scale', ehr_time_step=1,
                 matched_subset=True, imagenet_normalization=True,
                 preload_images=False, pkl_dir=None, attribution_cols=None,index = None,one_hot = None,
                 resized_base_path='/research/mimic_cxr_resized',
                 image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"):
        self.seed = seed
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.get_rng_state()
        
        
        self.task = task
        self.normalization = normalization
        self.ehr_time_step = ehr_time_step
        self.time_limit = time_limit
        self.matched_subset = matched_subset
        self.index = index
        self.one_hot = one_hot
        self.preload_images = {}
        self.resized_base_path = resized_base_path

        print(f"In our data time_limit is {time_limit}")
        print(f"Checking pkl_dir: {pkl_dir}")
        print(f"pkl_dir exists: {os.path.exists(pkl_dir)}")

        if attribution_cols is None:
            print(f"attribution_cols is None")
            self.attribution_cols = ['first_careunit', 'age', 'gender', 'admission_type', 'admission_location',
                                     'insurance', 'marital_status', 'race']

        self.data_root = Path(data_root)
        pkl_dir = '/home/cszjchen/MASAM/data_pkls'
        if pkl_dir is not None:
            if task == 'mortality2' or task == 'mortality':
                ehr_pkl_fpath = Path(pkl_dir) / f'mortality_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}_ts.pkl'
            else:
                ehr_pkl_fpath = Path(pkl_dir) / f'{task}_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}_ts.pkl'
                print(f"ehr_pkl_fpath is {ehr_pkl_fpath}")
            cxr_pkl_fpath = None
        else:
            ehr_pkl_fpath = None
            cxr_pkl_fpath = None


        meta_files_root = self.data_root/'splits'/f'fold{fold}'
        self.ehr_meta = pd.read_csv(meta_files_root/f'stays_{partition}.csv') 
        with open(meta_files_root/'train_stats.yaml', 'r') as f:
            self.train_stats = yaml.safe_load(f)


        _stay_sample = pd.read_csv(self.data_root/'time_series'/f'{self.ehr_meta.stay_id.loc[0]}.csv')
        self.features = [x for x in _stay_sample.columns if x in self.train_stats]
        self.mask_name = [x for x in _stay_sample.columns if 'mask' in x]
        print(f"now self.features is {self.features}")
        print(f"now self.mask_name is {self.mask_name}")

        self.features_stats = {
            stat: np.array([self.train_stats[feat][stat] for feat in self.features]).astype(float)
            for stat in ['iqr','max','mean','median','min','std']
        }
        self.features_no_normalization = [feat for feat in self.features if not self.train_stats[feat]['normalize']]
        self.default_imputation = {feat: self.train_stats[feat]['median'] for feat in self.features}

        print(f"In our datase, match is {self.matched_subset}")
        

        print(f"before matched data length of data is {len(self.ehr_meta['stay_id'].tolist())}")
        if self.matched_subset :
            self.ehr_meta = self.ehr_meta[(self.ehr_meta['valid_cxrs'] != '[]') & (self.ehr_meta['valid_cxrs'].notna())]
            self.stay_ids = self.ehr_meta['stay_id'].tolist()
            print(f"after matched length is {len(self.stay_ids)}")

        self.stay_ids = self.ehr_meta['stay_id'].tolist()

    
        print(f"ehr_pkl_fpath is {ehr_pkl_fpath}")
        print(f"ehr_pkl_fpath exists: {ehr_pkl_fpath.exists() if ehr_pkl_fpath else 'N/A'}")
        
        if pkl_dir and os.path.exists(pkl_dir):
            pkl_files = os.listdir(pkl_dir)
            print(f"Files in pkl_dir: {pkl_files}")
            matching_files = [f for f in pkl_files if f'fold{fold}' in f and partition in f]
            print(f"Matching files (fold{fold}, {partition}): {matching_files}")
        
        if ehr_pkl_fpath and ehr_pkl_fpath.exists():
            with open(ehr_pkl_fpath, 'rb') as f:
                self.normalized_data, self.missing_masks = pickle.load(f)
            print('Time series data loaded from pkl file.')
        else:
            self.normalized_data, self.missing_masks = self.load_and_normalize_time_series()
            if ehr_pkl_fpath:
                with open(ehr_pkl_fpath, 'wb') as f:
                    pickle.dump([self.normalized_data, self.missing_masks], f)


        cxr_transform = [transforms.Resize(256)]
        if partition == 'train':
            cxr_transform += [
                transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),
            ]
        cxr_transform += [
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        if imagenet_normalization:
            cxr_transform += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.cxr_transform = transforms.Compose(cxr_transform)

        if task == 'mortality':
            self.CLASSES = ['Mortality']
            self.targets = self.ehr_meta['icu_mortality'].values
        elif task == 'mortality2':
            self.CLASSES = ['Survival', 'Mortality'] 
            mortality_labels = self.ehr_meta['icu_mortality'].values
            targets_2class = np.zeros((len(mortality_labels), 2))
            targets_2class[np.arange(len(mortality_labels)), mortality_labels.astype(int)] = 1
            self.targets = targets_2class
        elif task == 'phenotype':
            self.CLASSES = self.ehr_meta.columns[-26:-1].tolist()
            self.targets = self.ehr_meta[self.CLASSES].values
        else:
            raise ValueError(f'Unknown task `{task}`')

        
        self.meta_attr = self.ehr_meta.set_index('stay_id')
        self.meta_attr = self.meta_attr[self.attribution_cols]

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        data = torch.FloatTensor(self.normalized_data[stay_id][:self.time_limit])
        masks = torch.FloatTensor(self.missing_masks[stay_id][:self.time_limit])
        labels = torch.FloatTensor(np.atleast_1d(self.targets[idx]))

        if self.index is not None:
            index = self.index
            labels = labels[index].unsqueeze(0)

        cxr_img = self._get_last_cxr_image_by_stay_id(stay_id, self.resized_base_path)
        has_cxr = False if cxr_img == None else True
        meta_attrs = self.meta_attr.loc[stay_id]

        if self.one_hot and self.task == "mortality":
            num_classes = 2 
            labels_one_hot = torch.zeros(num_classes)  
            labels_one_hot.scatter_(0, labels.long(), 1)  
            return stay_id, data, masks, cxr_img, has_cxr, labels_one_hot, meta_attrs, torch.LongTensor([idx])
        
        if self.task == "mortality2":
            return stay_id, data, masks, cxr_img, has_cxr, labels, meta_attrs, torch.LongTensor([idx])
        
        return stay_id, data, masks, cxr_img, has_cxr, labels, meta_attrs, torch.LongTensor([idx])

 
        

    def __len__(self):
        return len(self.stay_ids)

    def __load_time_series_by_stay_id(self, stay_id):
        stay_data_origin = pd.read_csv(self.data_root/'time_series'/f'{stay_id}.csv').sort_values(by='time_step')
        stay_data =  stay_data_origin[['time_step'] + self.features]
        stay_data_mask = stay_data_origin[self.mask_name]

        missing_mask = stay_data[self.features].isna().astype(float).values 
        data_imputed = stay_data[self.features].ffill().fillna(self.default_imputation)
        data_normalized = (data_imputed - self.features_stats['median']) / self.features_stats['iqr']
        data_normalized[self.features_no_normalization] = data_imputed[self.features_no_normalization]

        concatenated_data = np.concatenate((data_normalized, stay_data_mask), axis=1)
        


        return stay_id, concatenated_data, missing_mask

    def load_and_normalize_time_series(self):
        normalized_data = {}
        missing_masks = {}

        for stay_id in tqdm(self.stay_ids, desc='Loading and pre-processing raw time series'):
            _, data, masks = self.__load_time_series_by_stay_id(stay_id)
            normalized_data[stay_id] = data
            missing_masks[stay_id] = masks

        return normalized_data, missing_masks

    def _get_last_cxr_image_by_stay_id(self, stay_id, resized_base_path='/research/mimic_cxr_resized'):
        valid_cxrs = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'valid_cxrs'].values[0]
        
        if pd.isna(valid_cxrs) or valid_cxrs == "[]":
            return None

        valid_cxrs_clean = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", valid_cxrs)
        valid_cxrs_clean_parse = ast.literal_eval(valid_cxrs_clean)
        dicom_id = valid_cxrs_clean_parse[-1][0]

        subject_id = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'subject_id'].values[0]
        img_path = self.get_image_path(dicom_id, subject_id, resized_base_path=resized_base_path)
        if img_path is None:
            return None

        try:
            cxr_img = Image.open(img_path).convert('RGB')
            return self.cxr_transform(cxr_img)
        except FileNotFoundError:
            print(f"{img_path} not exists!!!!")
            return None


    def get_image_path(self, dicom_id, subject_id, resized_base_path='/research/mimic_cxr_resized'):

        image_path = f"{resized_base_path}/{dicom_id}.jpg"

        return image_path


def pad_temporal_data(batch):
    stay_ids, data, masks, cxr_imgs, has_cxr, labels, meta_attrs,idx = zip(*batch)
    seq_len = [x.shape[0] for x in data]
    max_len = max(seq_len)
    data_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                               for x in data], dim=0)
    masks_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                               for x in masks], dim=0)

    processed_cxr_imgs = []
    for x in cxr_imgs:
        if x is None:
            processed_cxr_imgs.append(torch.zeros(3, 224, 224))
        else:
            if isinstance(x, tuple):
                processed_cxr_imgs.append(torch.tensor(x))
            else:
                processed_cxr_imgs.append(x)

    cxr_imgs = torch.stack(processed_cxr_imgs)


    has_cxr = torch.FloatTensor(has_cxr)
    labels = torch.stack(labels, dim=0)

    idx = torch.stack(idx, dim=0)
    meta_attrs = pd.DataFrame(meta_attrs)
    batch_data = {
        'stay_ids': list(stay_ids),
        'seq_len': seq_len,
        'ehr_ts': data_padded,
        'ehr_masks': masks_padded,
        'cxr_imgs': cxr_imgs,
        'has_cxr': has_cxr,
        'labels': labels,
        'meta_attrs': meta_attrs,
        'idx':idx
    }
    return batch_data

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def create_data_loaders(ehr_data_dir, cxr_data_dir, task, replication, batch_size,
                        num_workers, time_limit=None,matched_subset = True,index = None,seed = None, one_hot=False,
                        resized_base_path='/research/mimic_cxr_resized',
                        image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"):
    set_seed(seed)
    time_limit = 48
    print(f"time limited is {time_limit}")

    data_loaders = []
    print(f"Now matched_subset is {matched_subset}")
    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')
        ds = MultiModalMIMIC(seed,ehr_data_dir, replication, split,
                             cxr_data_dir, task, time_limit=time_limit,matched_subset = matched_subset,index = index, one_hot = one_hot,
                             pkl_dir='/home/cszjchen/MASAM/data_pkls', resized_base_path=resized_base_path,
                             image_meta_path=image_meta_path)
        dl = DataLoader(ds, pin_memory=True,
                        shuffle=is_train, drop_last=is_train,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=pad_temporal_data,
                        worker_init_fn=seed_worker)
        data_loaders.append(dl)
        print(f"data_loaders is {len(data_loaders)}")
    return data_loaders
