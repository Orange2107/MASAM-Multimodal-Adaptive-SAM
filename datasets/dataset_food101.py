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
import json
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

class Food101Dataset(Dataset):

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

        self.CLASSES = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
            'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad',
            'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
            'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
            'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
            'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
            'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
            'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
            'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
            'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
        ]

        self.data_root = "your_path_to_dataset"
        self.image_path = self.data_root
        self.bert_feature_path = os.path.join(self.data_root, "bert_features")
        self.json_path = os.path.join(self.data_root, "cggm_json_with_bert")

        self.train_json = os.path.join(self.json_path, "train.json")
        self.val_json = os.path.join(self.json_path, "val.json")
        self.test_json = os.path.join(self.json_path, "test.json")

        if mode == 'train':
            json_file = self.train_json
        elif mode == 'val':
            json_file = self.val_json
        else:
            json_file = self.test_json

        with open(json_file, "r") as f:
            json_data = json.load(f)

        for sample in json_data:
            sample_id = sample['id']
            image_file = sample['img_path']
            text_tokens = sample['text_tokens']
            label_name = sample['label']

            if 'train/' in image_file:
                class_name = image_file.split('train/')[1].split('/')[0]
            elif 'test/' in image_file:
                class_name = image_file.split('test/')[1].split('/')[0]
            else:
                class_name = label_name

            full_image_path = os.path.join(self.data_root, image_file)

            bert_feature_file = sample.get('bert_feature_path', '')
            if bert_feature_file:
                full_bert_path = os.path.join(self.bert_feature_path, bert_feature_file)
            else:
                full_bert_path = os.path.join(self.bert_feature_path, mode, f"{sample_id}.npy")

            if os.path.exists(full_image_path) and os.path.exists(full_bert_path):
                data.append({
                    'id': sample_id,
                    'image_path': full_image_path,
                    'bert_path': full_bert_path,
                    'text_tokens': text_tokens,
                    'class_name': class_name
                })
                data2class[sample_id] = class_name
            else:
                print(full_image_path)
                print(f"Missing files for {sample_id}: image={os.path.exists(full_image_path)}, bert={os.path.exists(full_bert_path)}")
                continue

        self.classes = sorted(self.CLASSES)
        self.data2class = data2class
        self.samples = data

        print('# of samples = %d ' % len(self.samples))
        print('# of classes = %d' % len(self.classes))

        self._prepare_labels()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']

        bert_features = np.load(sample['bert_path'])

        image_path = sample['image_path']

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

        max_retries = 3
        for retry in range(max_retries):
            try:
                image = Image.open(image_path)
                image.verify()
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                break
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Warning: Failed to load image {image_path} after {max_retries} attempts. Using default image.")
                    default_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    image = transform(default_image)
                else:
                    continue

        label = self.classes.index(self.data2class[sample_id])

        if self.noise_generator is not None and self.noisy_labels is not None:
            sample_idx = idx
            label = self.noisy_labels[sample_idx]

            if hasattr(self.args, 'debug_noise') and self.args.debug_noise:
                original_label = self.original_labels[sample_idx]
                if original_label != label:
                    print(f"Noise corruption: {self.classes[original_label]} -> {self.classes[label]}")

        return bert_features, image, label, torch.LongTensor([idx])

    def _prepare_labels(self):
        self.original_labels = []
        for sample in self.samples:
            sample_id = sample['id']
            label = self.classes.index(self.data2class[sample_id])
            self.original_labels.append(label)

        self.original_labels = np.array(self.original_labels)

        if self.noise_generator is not None:
            print(f"Applying {self.noise_generator.noise_type} noise with ratio {self.noise_generator.noise_ratio}")

            save_path = f"./noise_maps/food101_{self.mode}_seed{self.noise_generator.noise_seed}_ratio{self.noise_generator.noise_ratio}.json"

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

def pad_food101_data(batch):
    bert_features, images, labels, idx = zip(*batch)

    labels = [label.clone().detach() if torch.is_tensor(label) else torch.tensor(label) for label in labels]
    labels = torch.stack(labels, dim=0)

    idx = [i.clone().detach() if torch.is_tensor(i) else torch.tensor(i) for i in idx]
    idx = torch.stack(idx, dim=0)

    bert_features = [b.clone().detach() if torch.is_tensor(b) else torch.tensor(b) for b in bert_features]
    bert_features = torch.stack(bert_features, dim=0)

    images = [img.clone().detach() if torch.is_tensor(img) else torch.tensor(img) for img in images]
    images = torch.stack(images, dim=0)

    batch_data = {
        'bert_features': bert_features,
        'images': images,
        'labels': labels,
        'idx': idx
    }
    return batch_data

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_food101_data_loaders(args, batch_size, num_workers, seed=None):
    set_seed(seed)

    data_loaders = []

    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')

        ds = Food101Dataset(args, mode=split)

        dl = DataLoader(ds,
                       pin_memory=True,
                       shuffle=is_train,
                       drop_last=is_train,
                       batch_size=batch_size,
                       num_workers=num_workers,
                       collate_fn=pad_food101_data,
                       worker_init_fn=seed_worker)
        data_loaders.append(dl)

    return data_loaders
