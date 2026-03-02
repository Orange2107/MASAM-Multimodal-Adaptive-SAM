#!/usr/bin/env python3

import numpy as np
import random
import torch
from typing import Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path


class LabelNoiseGenerator:

    def __init__(self, 
                 noise_type: str = 'symmetric',
                 noise_ratio: float = 0.0,
                 noise_seed: int = 42,
                 dataset_type: str = 'CREMAD',
                 debug_noise: bool = False):
        
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.noise_seed = noise_seed
        self.dataset_type = dataset_type
        self.debug_noise = debug_noise
        self._set_seed(noise_seed)
        self.asymmetric_patterns = self._get_asymmetric_patterns()
        self.noise_stats = {
            'total_samples': 0,
            'noisy_samples': 0,
            'noise_matrix': None,
            'corruption_indices': []
        }
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _get_asymmetric_patterns(self) -> Dict[str, Dict]:
        patterns = {
            'CREMAD': {
                0: {1: 0.6, 2: 0.3, 5: 0.1},
                1: {0: 0.7, 2: 0.2, 5: 0.1},
                2: {0: 0.4, 1: 0.4, 5: 0.2},
                3: {4: 0.8, 5: 0.2},
                4: {3: 0.6, 5: 0.4},
                5: {2: 0.5, 4: 0.5},
            },
            'FOOD101': {
                'pizza': {'pasta': 0.4, 'bread': 0.3, 'sandwich': 0.3},
                'pasta': {'pizza': 0.5, 'rice': 0.3, 'noodles': 0.2},
                'bread': {'pizza': 0.4, 'sandwich': 0.4, 'cake': 0.2},
            },
            'KINETICS': {
                'playing basketball': {'playing soccer': 0.4, 'playing tennis': 0.3, 'catching ball': 0.3},
                'playing soccer': {'playing basketball': 0.4, 'playing tennis': 0.3, 'kicking ball': 0.3},
                'playing tennis': {'playing basketball': 0.3, 'playing soccer': 0.3, 'catching ball': 0.4},
                'catching ball': {'playing basketball': 0.3, 'playing soccer': 0.3, 'throwing ball': 0.4},
                'throwing ball': {'catching ball': 0.5, 'kicking ball': 0.3, 'playing soccer': 0.2},
                'kicking ball': {'playing soccer': 0.6, 'throwing ball': 0.2, 'playing basketball': 0.2},
                'running': {'jogging': 0.5, 'walking': 0.3, 'riding bike': 0.2},
                'jogging': {'running': 0.6, 'walking': 0.3, 'riding bike': 0.1},
                'walking': {'jogging': 0.4, 'running': 0.3, 'standing': 0.3},
                'standing': {'walking': 0.5, 'sitting': 0.3, 'lying down': 0.2},
                'sitting': {'standing': 0.4, 'lying down': 0.4, 'sitting on floor': 0.2},
                'lying down': {'sitting': 0.5, 'standing': 0.3, 'sleeping': 0.2},
                'riding bike': {'running': 0.4, 'jogging': 0.3, 'walking': 0.3},
                'doing exercise': {'riding bike': 0.3, 'running': 0.3, 'walking': 0.4},
                'push up': {'doing exercise': 0.5, 'sit up': 0.3, 'pull up': 0.2},
                'sit up': {'push up': 0.4, 'doing exercise': 0.4, 'lying down': 0.2},
                'pull up': {'push up': 0.5, 'doing exercise': 0.3, 'climbing': 0.2},
                'playing guitar': {'playing violin': 0.4, 'playing piano': 0.3, 'playing drums': 0.3},
                'playing violin': {'playing guitar': 0.4, 'playing piano': 0.3, 'playing cello': 0.3},
                'playing piano': {'playing guitar': 0.3, 'playing violin': 0.3, 'playing drums': 0.4},
                'playing drums': {'playing guitar': 0.3, 'playing piano': 0.4, 'playing violin': 0.3},
                'playing cello': {'playing violin': 0.5, 'playing guitar': 0.3, 'playing piano': 0.2},
                'brushing hair': {'washing hands': 0.4, 'combing hair': 0.3, 'showering': 0.3},
                'washing hands': {'brushing hair': 0.3, 'showering': 0.4, 'cleaning windows': 0.3},
                'showering': {'washing hands': 0.4, 'brushing hair': 0.3, 'bathing': 0.3},
                'cleaning windows': {'washing hands': 0.3, 'cleaning floor': 0.4, 'mopping floor': 0.3},
                'cleaning floor': {'cleaning windows': 0.4, 'mopping floor': 0.4, 'sweeping floor': 0.2},
                'mopping floor': {'cleaning floor': 0.5, 'cleaning windows': 0.3, 'washing hands': 0.2},
                'hugging': {'kissing': 0.4, 'shaking hands': 0.3, 'waving hands': 0.3},
                'kissing': {'hugging': 0.5, 'shaking hands': 0.2, 'waving hands': 0.3},
                'shaking hands': {'hugging': 0.3, 'kissing': 0.2, 'waving hands': 0.5},
                'waving hands': {'shaking hands': 0.4, 'hugging': 0.3, 'clapping': 0.3},
                'clapping': {'waving hands': 0.4, 'shaking hands': 0.3, 'applauding': 0.3},
                'eating': {'drinking': 0.4, 'chewing': 0.3, 'biting': 0.3},
                'drinking': {'eating': 0.4, 'chewing': 0.3, 'sipping': 0.3},
                'chewing': {'eating': 0.5, 'drinking': 0.3, 'biting': 0.2},
                'biting': {'chewing': 0.5, 'eating': 0.3, 'drinking': 0.2},
                'talking on phone': {'texting': 0.4, 'typing': 0.3, 'writing': 0.3},
                'texting': {'talking on phone': 0.4, 'typing': 0.4, 'writing': 0.2},
                'typing': {'texting': 0.4, 'talking on phone': 0.3, 'writing': 0.3},
                'writing': {'typing': 0.4, 'texting': 0.3, 'talking on phone': 0.3},
                'opening door': {'closing door': 0.5, 'opening window': 0.3, 'closing window': 0.2},
                'closing door': {'opening door': 0.5, 'closing window': 0.3, 'opening window': 0.2},
                'opening window': {'opening door': 0.3, 'closing window': 0.4, 'opening door': 0.3},
                'closing window': {'opening window': 0.5, 'closing door': 0.3, 'opening door': 0.2},
                'climbing': {'descending': 0.4, 'riding bike': 0.3, 'walking': 0.3},
                'descending': {'climbing': 0.4, 'walking': 0.4, 'running': 0.2},
                'jumping': {'running': 0.4, 'hopping': 0.3, 'skipping': 0.3},
                'hopping': {'jumping': 0.5, 'skipping': 0.3, 'running': 0.2},
                'skipping': {'jumping': 0.4, 'hopping': 0.4, 'running': 0.2},
                'sleeping': {'lying down': 0.6, 'sitting': 0.3, 'standing': 0.1},
                'dancing': {'running': 0.3, 'jogging': 0.3, 'doing exercise': 0.4},
                'singing': {'playing guitar': 0.3, 'playing piano': 0.3, 'talking': 0.4},
                'drawing': {'writing': 0.4, 'painting': 0.3, 'typing': 0.3},
                'painting': {'drawing': 0.5, 'writing': 0.3, 'coloring': 0.2},
                'swimming': {'running': 0.3, 'doing exercise': 0.4, 'riding bike': 0.3},
                'applauding': {'clapping': 0.6, 'waving hands': 0.2, 'shaking hands': 0.2},
                'combing hair': {'brushing hair': 0.6, 'washing hands': 0.2, 'styling hair': 0.2},
                'sweeping floor': {'cleaning floor': 0.5, 'mopping floor': 0.3, 'cleaning windows': 0.2},
                'bathing': {'showering': 0.6, 'washing hands': 0.2, 'swimming': 0.2},
                'sipping': {'drinking': 0.6, 'eating': 0.2, 'chewing': 0.2},
                'coloring': {'drawing': 0.5, 'painting': 0.3, 'writing': 0.2},
                'styling hair': {'combing hair': 0.5, 'brushing hair': 0.3, 'washing hands': 0.2},
            },
            'Clinical': {
                0: {1: 0.8, 2: 0.2},
                1: {0: 0.3, 2: 0.7},
                2: {1: 0.9, 0: 0.1},
            }
        }
        return patterns
    
    def add_noise(self, 
                  labels: Union[List, np.ndarray, torch.Tensor],
                  class_names: Optional[List[str]] = None,
                  save_noise_map: bool = False,
                  save_path: Optional[str] = None) -> np.ndarray:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        
        original_labels = labels.copy()
        n_samples = len(labels)
        self.noise_stats['total_samples'] = n_samples
        if self.noise_ratio == 0.0 or self.noise_type == 'none':
            self.noise_stats['noisy_samples'] = 0
            return original_labels
        n_noisy = int(n_samples * self.noise_ratio)
        corruption_indices = np.random.choice(n_samples, n_noisy, replace=False)
        self.noise_stats['corruption_indices'] = corruption_indices.tolist()
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        noise_matrix = np.zeros((n_classes, n_classes))
        if self.noise_type == 'symmetric':
            noisy_labels = self._add_symmetric_noise(
                original_labels, corruption_indices, noise_matrix
            )
        elif self.noise_type == 'asymmetric':
            noisy_labels = self._add_asymmetric_noise(
                original_labels, corruption_indices, noise_matrix, class_names
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        self.noise_stats['noisy_samples'] = n_noisy
        self.noise_stats['noise_matrix'] = noise_matrix
        if save_noise_map and save_path:
            self._save_noise_mapping(save_path, original_labels, noisy_labels)
        
        return noisy_labels
    
    def _add_symmetric_noise(self, 
                            labels: np.ndarray,
                            corruption_indices: np.ndarray,
                            noise_matrix: np.ndarray) -> np.ndarray:
        noisy_labels = labels.copy()
        unique_labels = np.unique(labels)
        
        for idx in corruption_indices:
            original_label = labels[idx]
            possible_targets = [label for label in unique_labels if label != original_label]
            new_label = np.random.choice(possible_targets)
            noisy_labels[idx] = new_label
            orig_idx = np.where(unique_labels == original_label)[0][0]
            new_idx = np.where(unique_labels == new_label)[0][0]
            noise_matrix[orig_idx, new_idx] += 1
        
        return noisy_labels
    
    def _add_asymmetric_noise(self,
                             labels: np.ndarray,
                             corruption_indices: np.ndarray,
                             noise_matrix: np.ndarray,
                             class_names: Optional[List[str]] = None) -> np.ndarray:
        noisy_labels = labels.copy()
        unique_labels = np.unique(labels)
        if self.dataset_type in self.asymmetric_patterns:
            pattern = self.asymmetric_patterns[self.dataset_type]
            if class_names and isinstance(list(pattern.keys())[0], str):
                name_to_idx = {name: idx for idx, name in enumerate(class_names)}
                index_pattern = {}
                for orig_name, targets in pattern.items():
                    if orig_name in name_to_idx:
                        orig_idx = name_to_idx[orig_name]
                        index_pattern[orig_idx] = {}
                        for target_name, prob in targets.items():
                            if target_name in name_to_idx:
                                target_idx = name_to_idx[target_name]
                                index_pattern[orig_idx][target_idx] = prob
                pattern = index_pattern
        else:
            return self._add_symmetric_noise(labels, corruption_indices, noise_matrix)
        
        for idx in corruption_indices:
            original_label = labels[idx]
            if original_label in pattern:
                targets = pattern[original_label]
                target_labels = list(targets.keys())
                probabilities = list(targets.values())
                new_label = np.random.choice(target_labels, p=probabilities)
            else:
                possible_targets = [label for label in unique_labels if label != original_label]
                new_label = np.random.choice(possible_targets)
            
            noisy_labels[idx] = new_label
            orig_idx = np.where(unique_labels == original_label)[0][0]
            new_idx = np.where(unique_labels == new_label)[0][0]
            noise_matrix[orig_idx, new_idx] += 1
        
        return noisy_labels
    
    def _save_noise_mapping(self, save_path: str, original_labels: np.ndarray, noisy_labels: np.ndarray):
        noise_info = {
            'noise_type': self.noise_type,
            'noise_ratio': self.noise_ratio,
            'noise_seed': self.noise_seed,
            'dataset_type': self.dataset_type,
            'corruption_indices': self.noise_stats['corruption_indices'],
            'original_labels': original_labels.tolist(),
            'noisy_labels': noisy_labels.tolist(),
            'noise_matrix': self.noise_stats['noise_matrix'].tolist() if self.noise_stats['noise_matrix'] is not None else None
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(noise_info, f, indent=2)
        
        print(f"Noise mapping saved to: {save_path}")
    
    def add_noise_multilabel(self, 
                            labels: Union[List, np.ndarray, torch.Tensor],
                            class_names: Optional[List[str]] = None,
                            save_noise_map: bool = False,
                            save_path: Optional[str] = None) -> np.ndarray:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        
        original_labels = labels.copy()
        n_samples, n_classes = labels.shape
        self.noise_stats['total_samples'] = n_samples
        self.noise_stats['n_classes'] = n_classes
        if self.noise_ratio == 0.0 or self.noise_type == 'none':
            self.noise_stats['noisy_samples'] = 0
            return original_labels
        total_label_assignments = np.sum(labels)
        n_corruptions = int(total_label_assignments * self.noise_ratio)
        positive_positions = np.where(labels == 1)
        corruption_candidates = list(zip(positive_positions[0], positive_positions[1]))
        
        if len(corruption_candidates) == 0:
            print("Warning: No positive labels found in multi-label data")
            return original_labels
        corruption_indices = np.random.choice(len(corruption_candidates), 
                                           min(n_corruptions, len(corruption_candidates)), 
                                           replace=False)
        
        corrupted_positions = [corruption_candidates[i] for i in corruption_indices]
        self.noise_stats['corruption_indices'] = corrupted_positions
        noisy_labels = original_labels.copy()
        noise_matrix = np.zeros((n_classes, n_classes))
        
        for sample_idx, class_idx in corrupted_positions:
            if self.noise_type == 'symmetric':
                noisy_labels[sample_idx, class_idx] = 1 - noisy_labels[sample_idx, class_idx]
            elif self.noise_type == 'asymmetric':
                if class_names and len(class_names) > 1:
                    other_classes = [i for i in range(n_classes) if i != class_idx]
                    if other_classes:
                        target_class = np.random.choice(other_classes)
                        noisy_labels[sample_idx, class_idx] = 0
                        noisy_labels[sample_idx, target_class] = 1
                        noise_matrix[class_idx, target_class] += 1
                else:
                    noisy_labels[sample_idx, class_idx] = 1 - noisy_labels[sample_idx, class_idx]
            if self.noise_type == 'symmetric':
                noise_matrix[class_idx, 1-class_idx] += 1
        
        self.noise_stats['noisy_samples'] = len(corrupted_positions)
        self.noise_stats['noise_matrix'] = noise_matrix
        if save_noise_map:
            if save_path is None:
                save_path = f"./noise_maps/{self.dataset_type}_multilabel_noise_seed{self.noise_seed}.json"
            self._save_noise_mapping_multilabel(save_path, original_labels, noisy_labels)
        if self.debug_noise:
            self._print_debug_info_multilabel(original_labels, noisy_labels)
        
        return noisy_labels
    
    def load_noise_mapping(self, load_path: str) -> Dict:
        with open(load_path, 'r') as f:
            noise_info = json.load(f)
        assert noise_info['noise_type'] == self.noise_type
        assert noise_info['noise_ratio'] == self.noise_ratio
        assert noise_info['noise_seed'] == self.noise_seed
        assert noise_info['dataset_type'] == self.dataset_type
        
        return noise_info
    
    def get_noise_stats(self) -> Dict:
        return self.noise_stats.copy()
    
    def _save_noise_mapping_multilabel(self, save_path: str, original_labels: np.ndarray, noisy_labels: np.ndarray):
        noise_info = {
            'noise_type': self.noise_type,
            'noise_ratio': self.noise_ratio,
            'noise_seed': self.noise_seed,
            'dataset_type': self.dataset_type,
            'corruption_indices': self.noise_stats['corruption_indices'],
            'original_labels': original_labels.tolist(),
            'noisy_labels': noisy_labels.tolist(),
            'noise_matrix': self.noise_stats['noise_matrix'].tolist() if self.noise_stats['noise_matrix'] is not None else None,
            'multi_label': True
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(noise_info, f, indent=2)
        
        print(f"Multi-label noise mapping saved to: {save_path}")
    
    def _print_debug_info_multilabel(self, original_labels: np.ndarray, noisy_labels: np.ndarray):
        print("\n=== Multi-Label Noise Debug Info ===")
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Noise Type: {self.noise_type}")
        print(f"Noise Ratio: {self.noise_ratio:.2%}")
        print(f"Random Seed: {self.noise_seed}")
        print(f"Total Samples: {self.noise_stats['total_samples']}")
        print(f"Number of Classes: {self.noise_stats['n_classes']}")
        print(f"Corrupted Label Assignments: {self.noise_stats['noisy_samples']}")
        
        if self.noise_stats['noise_matrix'] is not None:
            print("Noise Matrix:")
            print(self.noise_stats['noise_matrix'])
        
        print("=" * 40)
    
    def print_noise_summary(self):
        stats = self.noise_stats
        print(f"\n=== Label Noise Summary ===")
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Noise Type: {self.noise_type}")
        print(f"Noise Ratio: {self.noise_ratio:.2%}")
        print(f"Random Seed: {self.noise_seed}")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Noisy Samples: {stats['noisy_samples']} ({stats['noisy_samples']/stats['total_samples']:.2%})")
        
        if stats['noise_matrix'] is not None:
            print(f"Noise Matrix:")
            print(stats['noise_matrix'])
        print("=" * 30)


def create_noise_generator_from_args(args) -> LabelNoiseGenerator:
    if hasattr(args, 'label_noise') and args.label_noise:
        noise_type = getattr(args, 'noise_type', 'symmetric')
        noise_ratio = getattr(args, 'noise_ratio', 0.2)
        noise_seed = getattr(args, 'noise_seed', args.seed)
        dataset_type = getattr(args, 'dataset', 'CREMAD')
        
        return LabelNoiseGenerator(
            noise_type=noise_type,
            noise_ratio=noise_ratio,
            noise_seed=noise_seed,
            dataset_type=dataset_type
        )
    else:
        return LabelNoiseGenerator(noise_ratio=0.0)


if __name__ == "__main__":
    print("Testing symmetric noise...")
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    generator = LabelNoiseGenerator('symmetric', 0.3, 42, 'CREMAD')
    noisy_labels = generator.add_noise(labels)
    generator.print_noise_summary()
    print("\nTesting asymmetric noise...")
    class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    generator = LabelNoiseGenerator('asymmetric', 0.3, 42, 'CREMAD')
    noisy_labels = generator.add_noise(labels, class_names)
    generator.print_noise_summary()