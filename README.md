# MASAM: Modality-Adaptive Sharpness-Aware Minimization for Multimodal Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥1.10-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-≥2.0-792ee5.svg?logo=lightning)](https://www.lightning.ai/)

Official implementation of **MASAM** (Modality-Adaptive Sharpness-Aware Minimization), a novel optimization framework for multimodal learning that adaptively balances gradient sharpness across different modalities.



## 📊 Supported Datasets

| Dataset | Modality | Task | Classes |
|---------|----------|------|---------|
| **MIMIC** | EHR + CXR | Mortality/Phenotype Prediction | 2/25 |
| **CREMAD** | Audio + Video | Emotion Recognition | 6 |
| **KINETICS** | Audio + Video | Action Recognition | 31 |
| **FOOD101** | Text + Image | Food Classification | 101 |
| **URFUNNY** | Audio + Video + Text | Humor Detection | 2 |
| **ADNI** | Clinical + MRI | Disease Classification | 3 |

## 📚 Data Processing References

We process datasets following these established methods:

- **CREMA-D & Kinetics-Sounds**: Processed following [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main) (CVPR 2022 Oral)
- **Food101**: Processed following [CGGM](https://github.com/zrguo/CGGM)(NeurIPS 2024)
- **MIMIC**: Processed following [DrFuse](https://github.com/dorothy-yao/drfuse)(AAAI 2024)
- **UR-Funny**: Processed following [AGM](https://github.com/lihongcs/AGM) (ICCV 2023)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/MASAM.git
cd MASAM

# Create conda environment (recommended)
conda create -n masam python=3.9.19
conda activate masam

# Install dependencies from requirement file
pip install -r requirement.yaml
```


See [`requirement.yaml`](requirement.yaml) for the complete list of dependencies.

### Basic Training

Train MASAM on MIMIC-IV mortality prediction:
```bash
python main.py --model masam --mode train --dataset MIMIC --task mortality --sam_decomp --uniloss --patience 10 --seed 42
```

Train on other datasets:
```bash
# CREMAD (Audio-Visual Emotion Recognition)
python main.py --model masam --mode train --dataset CREMAD --sam_decomp --uniloss --patience 10 --seed 42

# FOOD101 (Text-Image Food Classification)
python main.py --model masam --mode train --dataset FOOD101 --sam_decomp --uniloss --patience 10 --seed 42

# KINETICS (Audio-Visual Action Recognition)
python main.py --model masam --mode train --dataset KINETICS --sam_decomp --uniloss --patience 10 --seed 42

# URFUNNY (Multimodal Humor Detection)
python main.py --model masam --mode train --dataset URFUNNY --sam_decomp --uniloss --patience 10 --seed 42

# ADNI (Clinical Disease Classification)
python main.py --model masam --mode train --dataset ADNI --sam_decomp --uniloss --patience 10 --seed 42
```

**Important:** The `--sam_decomp` flag is required to activate the MASAM method. Without it, the model will use standard training.

### Fast Debug Mode

Use `--dev_run` for quick testing (runs only 20 steps):
```bash
python main.py --model masam --mode train --dataset CREMAD --sam_decomp --uniloss --dev_run
```


## 📁 Project Structure

```
MASAM/
├── main.py                 # Main training script
├── arguments.py            # Argument parser
├── models/
│   ├── masam.py           # MASAM model implementation
│   ├── sam_closure.py     # SAM optimizer
│   ├── sagm_closure.py    # SAGM optimizer
│   └── ...                # Other models
├── datasets/
│   ├── dataset_mimic.py   # MIMIC dataset
│   ├── dataset_cremad.py  # CREMAD dataset
│   ├── dataset_ks.py      # KINETICS dataset
│   ├── dataset_food101.py # FOOD101 dataset
│   ├── dataset_urfunny.py # URFUNNY dataset
│   └── ...
├── configs/
│   ├── masam_mimic.yaml   # MIMIC config
│   ├── masam_cremad.yaml  # CREMAD config
│   ├── masam_ks.yaml      # KINETICS config
│   ├── masam_food101.yaml # FOOD101 config
│   └── masam_urfunny.yaml # URFUNNY config
└── README.md              # This file
```


## 📊 Monitoring Training

Training logs and checkpoints are saved in:
```
./experiments/for{model}/{dataset}/
├── {MODEL_NAME}-{params}-seed{seed}/
│   ├── checkpoints/          # Model checkpoints
│   ├── tb_logger/            # TensorBoard logs
│   └── test_set_results.yaml # Test results
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir ./experiments/for{model}/{dataset}/
```

## 🧪 Testing a Trained Model

```bash
python main.py \
  --model masam \
  --mode test \
  --dataset CREMAD \
  --best_model_path ./experiments/formasam/cremad/MASAM-.../checkpoints/epoch=xx-overall_Accuracy=0.xx.ckpt
```

## 📝 Configuration Files

Configuration files are located in `configs/` directory with naming format `{model_name}_{dataset}.yaml`.

Example configuration (`configs/masam_cremad.yaml`):
```yaml
hparams:
  fusion_method: 'concate'
  hidden_size: 512
  num_classes: 6
  batch_size: 16
  learning_rate: 1e-4
  seed: 42
  dataset: 'CREMAD'
  rho: 0.3
  score_weight: 0.8
  momentum: 0.91
  loss_multi: 1.0
  loss_ehr: 1.0
  loss_cxr: 1.0
```

Override config parameters via command line:
```bash
python main.py --model masam --mode train --dataset CREMAD --sam_decomp --uniloss --rho 0.5 --score_weight 0.8
```




## 🤝 Citing This Work

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{chenmasam,
  title={MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion},
  author={Chen, Zijie and Yin, Kejing and Yao, Wenfang and Cheung, William K and Qin, Jing},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Last Updated:** February 25, 2026
