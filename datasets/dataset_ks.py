import copy
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from .label_noise_generator import LabelNoiseGenerator, create_noise_generator_from_args

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

class KineticsDataset(Dataset):

    def __init__(self, args, mode='train'):
        data = []
        data2class = {}
        self.mode = mode
        self.args = args

        self.noise_generator = None
        self.original_labels = None
        self.noisy_labels = None

        if hasattr(args, 'label_noise') and args.label_noise and mode == 'train':
            print(f"Creating label noise generator for {mode} set")
            self.noise_generator = create_noise_generator_from_args(args)

        self.CLASSES = [str(i) for i in range(31)]

        self.data_root = '/hdd/datasets/kinetics-dataset/'
        self.visual_feature_path = os.path.join(self.data_root, "image/processed_frames")
        self.audio_feature_path = os.path.join(self.data_root, f"audio/wav_npy")

        self.train_txt = os.path.join(self.data_root, "k400/data_txt/self_train.txt")
        self.val_txt = os.path.join(self.data_root, "k400/data_txt/self_val.txt")
        self.test_txt = os.path.join(self.data_root, "k400/data_txt/self_test.txt")

        class_file = os.path.join(self.data_root, "k400/data_txt/class_list.txt")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                classes = [line.strip() for line in f.readlines()]
                self.CLASSES = classes

        if mode == 'train':
            csv_file = self.train_txt
        elif mode == 'val':
            csv_file = self.val_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            for single_line in f2:
                line_parts = single_line.strip().split(",")
                if len(line_parts) >= 2:
                    video_id = line_parts[0]
                    class_id = line_parts[-1]

                    audio_path = os.path.join(self.audio_feature_path, mode, video_id + '.npy')
                    visual_path = os.path.join(self.visual_feature_path, video_id)

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        data.append(video_id)
                        data2class[video_id] = class_id
                    else:
                        print(f"Missing files for {video_id}: audio={os.path.exists(audio_path)}, visual={os.path.exists(visual_path)}")
                        continue

        self.classes = sorted(self.CLASSES, key=int)
        self.data2class = data2class
        self.av_files = data

        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

        self._prepare_labels()

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        audio_path = os.path.join(self.audio_feature_path, self.mode, av_file + '.npy')
        spectrogram = np.load(audio_path)

        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = sorted(os.listdir(visual_path))
        file_num = len(allimages)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 2

        images = torch.zeros((pick_num, 3, 224, 224))

        if file_num >= pick_num:
            select_index = np.random.choice(file_num, size=pick_num, replace=False)
        else:
            select_index = np.random.choice(file_num, size=file_num, replace=False)
        select_index.sort()

        loaded_count = 0
        max_retries = min(file_num, 10)

        for i, idx in enumerate(select_index):
            success = False
            retry_count = 0

            while not success and retry_count < max_retries:
                try:
                    current_idx = (idx + retry_count) % file_num
                    image_path = os.path.join(visual_path, allimages[current_idx])

                    image = Image.open(image_path)

                    image.verify()

                    image = Image.open(image_path).convert('RGB')

                    image = transform(image)
                    images[i] = image
                    success = True

                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Warning: Failed to load image for {av_file} after {max_retries} attempts. Using default image.")

                        default_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                        image = transform(default_image)
                        images[i] = image
                        success = True
                    else:
                        continue

            loaded_count += 1

        if loaded_count < pick_num:
            print(f"Warning: Only loaded {loaded_count}/{pick_num} images for {av_file}")
            for i in range(loaded_count, pick_num):
                default_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                image = transform(default_image)
                images[i] = image

        image_n = images
        label = self.classes.index(self.data2class[av_file])

        if self.noise_generator is not None and self.noisy_labels is not None:
            file_idx = self.av_files.index(av_file)
            label = self.noisy_labels[file_idx]

            if hasattr(self.args, 'debug_noise') and self.args.debug_noise:
                original_label = self.original_labels[file_idx]
                if original_label != label:
                    print(f"Noise corruption: {self.classes[original_label]} -> {self.classes[label]}")

        return spectrogram, image_n, label, torch.LongTensor([idx])

    def _prepare_labels(self):
        self.original_labels = []
        for av_file in self.av_files:
            label = self.classes.index(self.data2class[av_file])
            self.original_labels.append(label)

        self.original_labels = np.array(self.original_labels)

        if self.noise_generator is not None:
            print(f"Applying {self.noise_generator.noise_type} noise with ratio {self.noise_generator.noise_ratio}")

            save_path = f"./noise_maps/kinetics_{self.mode}_seed{self.noise_generator.noise_seed}_ratio{self.noise_generator.noise_ratio}.json"

            self.noisy_labels = self.noise_generator.add_noise(
                self.original_labels,
                class_names=self.classes,
                save_noise_map=True,
                save_path=save_path
            )

            self.noise_generator.print_noise_summary()
        else:
            self.noisy_labels = self.original_labels

    def get_noise_stats(self):
        if self.noise_generator:
            return self.noise_generator.get_noise_stats()
        return None

    def get_original_labels(self):
        return self.original_labels.copy() if self.original_labels is not None else None

def pad_av_temporal_data(batch):
    audio, vedio, labels, idx = zip(*batch)

    labels = [label.clone().detach() if torch.is_tensor(label) else torch.tensor(label) for label in labels]
    labels = torch.stack(labels, dim=0)

    idx = [i.clone().detach() if torch.is_tensor(i) else torch.tensor(i) for i in idx]
    idx = torch.stack(idx, dim=0)

    audio = [a.clone().detach().unsqueeze(0) if torch.is_tensor(a) else torch.tensor(a).unsqueeze(0) for a in audio]
    audio = torch.stack(audio, dim=0)

    vedio = [v.clone().detach() if torch.is_tensor(v) else torch.tensor(v) for v in vedio]
    vedio = torch.stack(vedio, dim=0)

    batch_data = {
        'audio': audio,
        'vedio': vedio,
        'labels': labels,
        'idx': idx
    }
    return batch_data

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_kinetics_data_loaders(args, batch_size, num_workers, seed=None):
    set_seed(seed)

    data_loaders = []

    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')

        ds = KineticsDataset(args, mode=split)

        dl = DataLoader(ds,
                       pin_memory=True,
                       shuffle=is_train,
                       drop_last=is_train,
                       batch_size=batch_size,
                       num_workers=num_workers,
                       collate_fn=pad_av_temporal_data,
                       worker_init_fn=seed_worker)
        data_loaders.append(dl)

    return data_loaders