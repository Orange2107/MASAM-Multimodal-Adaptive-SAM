import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def set_seed(_seed: int):
    global seed
    seed = _seed
    import random
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class URFunnyDataset(Dataset):

    def __init__(self, args, mode: str = 'train'):
        data_root = '/hdd/datasets/UR-Funny/URFUNNY_AGM_format.pkl'

        assert os.path.exists(data_root), f"UR-Funny pkl not found: {data_root}"
        assert mode in ['train', 'val', 'test']

        self.aligned = getattr(args, 'aligned', True)
        self.z_norm = getattr(args, 'z_norm', True)
        self.flatten = getattr(args, 'flatten', False)
        self.max_pad = getattr(args, 'max_pad', False)
        self.max_pad_num = getattr(args, 'max_pad_num', 128)

        with open(data_root, 'rb') as f:
            all_data = pickle.load(f)

        split_key = 'train' if mode == 'train' else ('valid' if mode == 'val' else 'test')
        dataset = self._drop_entry(all_data[split_key])

        self.vision = dataset['vision']
        self.audio = dataset['audio']
        self.text = dataset['text']
        self.labels = dataset['labels']
        self.CLASSES = ['NotFunny', 'Funny']
        self.v_dim = int(self.vision.shape[-1]) if self.vision.ndim == 3 else 371
        self.a_dim = int(self.audio.shape[-1]) if self.audio.ndim == 3 else 81
        self.t_dim = int(self.text.shape[-1]) if self.text.ndim == 3 else 300

    @staticmethod
    def _drop_entry(dataset):
        drop = []
        for idx, k in enumerate(dataset['text']):
            if k.sum() == 0:
                drop.append(idx)
        for modality in list(dataset.keys()):
            dataset[modality] = np.delete(dataset[modality], drop, 0)
        return dataset

    def __len__(self):
        return self.vision.shape[0]

    def __getitem__(self, index: int):
        vision = torch.tensor(self.vision[index])
        audio = torch.tensor(self.audio[index])
        text = torch.tensor(self.text[index])

        audio[audio == -np.inf] = 0.0

        def _first_valid_idx(x: torch.Tensor):
            if x.ndim != 2 or x.shape[0] == 0:
                return None
            finite = torch.isfinite(x).all(dim=1)
            nonzero = (x.abs().sum(dim=1) > 0)
            mask = finite & nonzero
            idx = torch.nonzero(mask)
            if idx.numel() == 0:
                return None
            return int(idx[0].item())

        if self.aligned:
            start_candidates = []
            idx_t = _first_valid_idx(text)
            if idx_t is not None:
                start = idx_t
            else:
                for m in (vision, audio, text):
                    idx = _first_valid_idx(m)
                    if idx is not None:
                        start_candidates.append(idx)
                start = min(start_candidates) if start_candidates else 0
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        if self.z_norm:
            vision = torch.nan_to_num((vision - vision.mean(0)) / (vision.std() + 1e-8))
            audio = torch.nan_to_num((audio - audio.mean(0)) / (audio.std() + 1e-8))
            text = torch.nan_to_num((text - text.mean(0)) / (text.std() + 1e-8))

        has_v = float(vision.shape[0] > 0 and torch.isfinite(vision).any() and vision.abs().sum() > 0)
        has_a = float(audio.shape[0] > 0 and torch.isfinite(audio).any() and audio.abs().sum() > 0)
        has_t = float(text.shape[0] > 0 and torch.isfinite(text).any() and text.abs().sum() > 0)

        if vision.shape[0] == 0:
            vision = torch.zeros(1, self.v_dim)
        if audio.shape[0] == 0:
            audio = torch.zeros(1, self.a_dim)
        if text.shape[0] == 0:
            text = torch.zeros(1, self.t_dim)

        lbl = self.labels[index]
        if isinstance(lbl, (list, tuple, np.ndarray)):
            try:
                lbl = int(np.array(lbl).reshape(-1)[0] >= 1)
            except Exception:
                lbl = int(np.array(lbl).mean() >= 1)
        else:
            lbl = int(lbl >= 1)

        label = torch.tensor(lbl, dtype=torch.long)

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), index, label]
        elif self.max_pad:
            v = vision[: self.max_pad_num]
            a = audio[: self.max_pad_num]
            t = text[: self.max_pad_num]
            pad_v = torch.nn.functional.pad(v, (0, 0, 0, self.max_pad_num - v.shape[0]))
            pad_a = torch.nn.functional.pad(a, (0, 0, 0, self.max_pad_num - a.shape[0]))
            pad_t = torch.nn.functional.pad(t, (0, 0, 0, self.max_pad_num - t.shape[0]))
            return [pad_v, pad_a, pad_t, label, has_v, has_a, has_t]
        else:
            return [vision, audio, text, index, label, has_v, has_a, has_t]

def _collate_variable_len(batch: List[List[torch.Tensor]]):
    visions = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    indices = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    has_v = [item[5] for item in batch]
    has_a = [item[6] for item in batch]
    has_t = [item[7] for item in batch]

    vis_lens = torch.as_tensor([v.size(0) for v in visions])
    aud_lens = torch.as_tensor([a.size(0) for a in audios])
    txt_lens = torch.as_tensor([t.size(0) for t in texts])

    vis_pad = pad_sequence(visions, batch_first=True)
    aud_pad = pad_sequence(audios, batch_first=True)
    txt_pad = pad_sequence(texts, batch_first=True)

    return {
        'vision': vis_pad.float(),
        'audio': aud_pad.float(),
        'text': txt_pad.float(),
        'vision_lengths': vis_lens,
        'audio_lengths': aud_lens,
        'text_lengths': txt_lens,
        'labels': torch.as_tensor(labels, dtype=torch.long),
        'idx': torch.as_tensor(indices, dtype=torch.long).view(len(batch), 1),
        'has_vision': torch.as_tensor(has_v, dtype=torch.float32).view(len(batch), 1),
        'has_audio': torch.as_tensor(has_a, dtype=torch.float32).view(len(batch), 1),
        'has_text': torch.as_tensor(has_t, dtype=torch.float32).view(len(batch), 1),
    }

def _collate_fixed_len(batch: List[List[torch.Tensor]]):
    visions = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    has_v = [item[4] for item in batch]
    has_a = [item[5] for item in batch]
    has_t = [item[6] for item in batch]

    return {
        'vision': torch.stack(visions).float(),
        'audio': torch.stack(audios).float(),
        'text': torch.stack(texts).float(),
        'labels': torch.as_tensor(labels, dtype=torch.long),
        'has_vision': torch.as_tensor(has_v, dtype=torch.float32).view(len(batch), 1),
        'has_audio': torch.as_tensor(has_a, dtype=torch.float32).view(len(batch), 1),
        'has_text': torch.as_tensor(has_t, dtype=torch.float32).view(len(batch), 1),
    }

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def create_urfunny_data_loaders(args, batch_size: int, num_workers: int,
                                seed: int = 42):
    set_seed(seed)

    loaders = []
    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')
        ds = URFunnyDataset(args, mode=split)
        collate_fn = _collate_fixed_len if getattr(args, 'max_pad', False) else _collate_variable_len
        dl = DataLoader(
            ds,
            pin_memory=True,
            shuffle=is_train,
            drop_last=is_train,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )
        loaders.append(dl)
    return loaders
