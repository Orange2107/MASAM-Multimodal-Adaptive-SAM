import os
import pickle
import yaml
from copy import deepcopy
import torch.nn.init as init
from datetime import datetime
from argparse import Namespace
import numpy as np
from torch import nn
import random
import torch
import lightning as L

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from arguments import get_arg_parser

from datasets.dataset_mimic import create_data_loaders
from datasets.dataset_cremad import create_cremad_data_loaders
from datasets.dataset_ks import create_kinetics_data_loaders
from datasets.dataset_food101 import create_food101_data_loaders
from datasets.dataset_urfunny import create_urfunny_data_loaders

from models.masam import MASAM

def get_model_specific_params(args):
    """获取 MASAM 模型的参数字符串"""
    base_params = f"bs{args.batch_size}-lr{args.learning_rate}-fusion{args.fusion_method}-hs{args.hidden_size}"

    has_noise = hasattr(args, 'noise_ratio') and args.noise_ratio is not None and args.noise_ratio > 0
    noise_suffix = f"-noise{args.noise_ratio}-{getattr(args, 'noise_type', 'symmetric')}" if has_noise else ""

    if args.dataset in ['CREMAD', 'KINETICS']:
        model_specific = f"{base_params}-opt{getattr(args, 'optimizer', 'adam')}-rho{getattr(args, 'rho', 0.1)}-wd{getattr(args, 'wd', 0.0)}-lm{getattr(args, 'loss_multi', 1.0)}-le{getattr(args, 'loss_ehr', 1.0)}-lc{getattr(args, 'loss_cxr', 1.0)}-dynamic{getattr(args, 'dynamic_mode', False)}-sw{getattr(args, 'score_weight', 0.5)}-m{getattr(args, 'momentum', 0.9)}{noise_suffix}"
    elif args.dataset == 'FOOD101':
        model_specific = f"{base_params}-rho{getattr(args, 'rho', 0.1)}-wd{getattr(args, 'wd', 0.0)}-lm{getattr(args, 'loss_multi', 1.0)}-lt{getattr(args, 'loss_text', 1.0)}-li{getattr(args, 'loss_image', 1.0)}-dynamic{getattr(args, 'dynamic_mode', False)}-sw{getattr(args, 'score_weight', 0.5)}-m{getattr(args, 'momentum', 0.9)}{noise_suffix}"
    elif args.dataset == 'URFUNNY':
        model_specific = f"{base_params}-rho{getattr(args, 'rho', 0.1)}-wd{getattr(args, 'wd', 0.0)}-lm{getattr(args, 'loss_multi', 1.0)}-la{getattr(args, 'loss_audio', 1.0)}-lv{getattr(args, 'loss_video', 1.0)}-lt{getattr(args, 'loss_text', 1.0)}-dynamic{getattr(args, 'dynamic_mode', False)}-sw{getattr(args, 'score_weight', 0.5)}-m{getattr(args, 'momentum', 0.9)}{noise_suffix}"
    elif args.dataset == 'ADNI':
        model_specific = f"{base_params}-rho{getattr(args, 'rho', 0.1)}-wd{getattr(args, 'wd', 0.0)}-lm{getattr(args, 'loss_multi', 1.0)}-le{getattr(args, 'loss_ehr', 1.0)}-lmri{getattr(args, 'loss_mri', 1.0)}-dynamic{getattr(args, 'dynamic_mode', False)}-sw{getattr(args, 'score_weight', 0.5)}-m{getattr(args, 'momentum', 0.9)}{noise_suffix}"
    elif args.dataset == 'MIMIC':
        model_specific = f"{base_params}-eh{args.ehr_n_head}-el{args.ehr_n_layers}-ed{args.ehr_dropout}-rho{getattr(args, 'rho', 0.1)}-wd{getattr(args, 'wd', 0.0)}-lm{getattr(args, 'loss_multi', 1.0)}-le{getattr(args, 'loss_ehr', 1.0)}-lc{getattr(args, 'loss_cxr', 1.0)}-dynamic{getattr(args, 'dynamic_mode', False)}-sw{getattr(args, 'score_weight', 0.5)}-m{getattr(args, 'momentum', 0.9)}{noise_suffix}"
    else:
        model_specific = base_params + noise_suffix

    return model_specific

def get_log_info(args):
    """获取日志目录和版本名称"""
    has_noise = hasattr(args, 'noise_ratio') and args.noise_ratio is not None and args.noise_ratio > 0

    if args.dataset == 'CREMAD':
        base_dir = f"./experiments/for{args.model}/cremad"
        if has_noise:
            base_dir += "_noisy"
    elif args.dataset == 'KINETICS':
        base_dir = f"./experiments/for{args.model}/kinetics"
        if has_noise:
            base_dir += "_noisy"
    elif args.dataset == 'FOOD101':
        base_dir = f"./experiments/for{args.model}/food101"
        if has_noise:
            base_dir += "_noisy"
    elif args.dataset == 'URFUNNY':
        base_dir = f"./experiments/for{args.model}/urfunny"
        if has_noise:
            base_dir += "_noisy"
    elif args.dataset == 'ADNI':
        base_dir = f"./experiments/for{args.model}/adni"
        if has_noise:
            base_dir += "_noisy"
    elif args.dataset == 'MIMIC':
        base_dir = f"./experiments/for{args.model}/{args.task}"
        if has_noise:
            base_dir += "_noisy"
    else:
        base_dir = f"./experiments/for{args.model}/default"

    specific_params = get_model_specific_params(args)

    prefix = f"{args.model.upper()}"
    ver_name = f"{prefix}-{specific_params}-seed{args.seed}"

    return base_dir, ver_name

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_config(model_name, args):
    """加载模型配置并用命令行参数覆盖"""
    config_map = {
        'masam': f'masam_{args.dataset.lower()}.yaml',
        'sam': f'sam_{args.dataset.lower()}.yaml',
        'latefusion': f'latefusion_{args.dataset.lower()}.yaml',
    }

    config_file = config_map.get(model_name, f'{model_name}_{args.dataset.lower()}.yaml')
    config_path = f'./configs/{config_file}'

    if not os.path.exists(config_path):
        config_path = f'./configs/{model_name}.yaml'

    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using default config")
        model_params = {
            'fusion_method': 'concate',
            'hidden_size': 512,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'dataset': args.dataset,
            'num_classes': 2,
        }
    else:
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
        model_params = config['hparams']

    for key, value in vars(args).items():
        if key in model_params:
            model_params[key] = value

    model_params['dataset'] = args.dataset

    return model_params

def create_data_loaders_by_dataset(args, seed):
    """根据数据集类型创建数据加载器"""
    if args.dataset == 'CREMAD':
        return create_cremad_data_loaders(
            args, args.batch_size, args.num_workers,
            seed=seed
        )
    elif args.dataset == 'KINETICS':
        return create_kinetics_data_loaders(
            args, args.batch_size, args.num_workers,
            seed=seed
        )
    elif args.dataset == 'FOOD101':
        return create_food101_data_loaders(
            args, args.batch_size, args.num_workers,
            seed=seed
        )
    elif args.dataset == 'URFUNNY':
        return create_urfunny_data_loaders(
            args, args.batch_size, args.num_workers,
            seed=seed
        )
    elif args.dataset == 'MIMIC':
        return create_data_loaders(
            args.ehr_root, args.cxr_root, args.task,
            args.fold, args.batch_size, args.num_workers,
            matched_subset=args.matched,
            index=args.index,
            seed=seed,
            one_hot=args.mortality2 if hasattr(args, 'mortality2') else None,
            resized_base_path=args.resized_cxr_root
        )
    else:
        raise NotImplementedError(f'Dataset `{args.dataset}` is not supported.')

def get_model_class(args):
    """返回 MASAM 模型类"""
    return MASAM

def run_model(args):

    if isinstance(args, dict):
        args = Namespace(**args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(5)
    L.seed_everything(seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Now matched_subset is {args.matched}")
    print(f"Using dataset: {args.dataset}")

    train_loader, val_loader, test_loader = create_data_loaders_by_dataset(args, seed)
    print(f"train_loader: {len(train_loader)}")

    model_class = get_model_class(args)
    train_data_num = len(train_loader.dataset)
    model_params = load_model_config(args.model, args)

    model_params.update({
        'new_data': args.new_data,
        'bs': args.batch_size,
        'matched': args.matched,
        'learning_rate': args.learning_rate,
        'uniloss': getattr(args, 'uniloss', False),
        'gpu': args.gpu,
        'model_name': getattr(args, 'model_name', ''),
        'task': args.task,
        'seed': args.seed,
        'save': args.save,
        'dataset': args.dataset,
        'hidden_size': args.hidden_size,
        'hidden_size_cxr': getattr(args, 'hidden_size_cxr', args.hidden_size),
        'ehr_n_head': args.ehr_n_head,
        'ehr_n_layers': args.ehr_n_layers,
        'ehr_n_layers_distinct': getattr(args, 'ehr_n_layers_distinct', args.ehr_n_layers),
        'ehr_dropout': args.ehr_dropout,
        'cxr_n_layers': getattr(args, 'cxr_n_layers', 1),
        'cxr_dropout': getattr(args, 'cxr_dropout', 0.2),
        'class_names': train_loader.dataset.CLASSES,
        'train_data_num': train_data_num,
        'sam': getattr(args, 'sam', False),
        'sagm': getattr(args, 'sagm', False),
        'sam_decomp': getattr(args, 'sam_decomp', False),
        'rho': getattr(args, 'rho', 0.1),
        'weight_decay': getattr(args, 'wd', 0.0),
        'score_weight': getattr(args, 'score_weight', 0.5),
        'momentum': getattr(args, 'momentum', 0.9),
        'dynamic_mode': getattr(args, 'dynamic_mode', False),
        'loss_multi': getattr(args, 'loss_multi', 1.0),
        'loss_ehr': getattr(args, 'loss_ehr', 1.0),
        'loss_cxr': getattr(args, 'loss_cxr', 1.0),
        'fusion_method': args.fusion_method,
    })

    model = model_class(model_params)

    if args.dataset in ['CREMAD', 'KINETICS', 'FOOD101', 'URFUNNY', 'ADNI']:
        callback_metric = 'overall/Accuracy'
    elif args.task == 'mortality2':
        callback_metric = 'class-wise_PRAUC/Mortality'
    else:
        callback_metric = 'overall/PRAUC'

    early_stop_callback = EarlyStopping(
        monitor=callback_metric,
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=callback_metric,
        mode='max',
        save_top_k=1,
        verbose=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        filename='{epoch:02d}-{overall/Accuracy:.2f}' if (args.dataset in ['CREMAD', 'KINETICS', 'FOOD101', 'URFUNNY', 'ADNI']) else ('{epoch:02d}-{class-wise_PRAUC/Mortality:.2f}' if args.task == 'mortality2' else '{epoch:02d}-{overall/PRAUC:.2f}')
    )

    log_dir, ver_name = get_log_info(args)
    print(f"log_dir: {log_dir}")
    print(f"ver_name: {ver_name}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, version=ver_name)

    trainer = L.Trainer(
        enable_checkpointing=args.save_checkpoint if hasattr(args, 'save_checkpoint') else True,
        accelerator='gpu',
        devices=[args.gpu],
        fast_dev_run=20 if args.dev_run else False,
        logger=tb_logger,
        num_sanity_val_steps=0,
        max_epochs=200,
        log_every_n_steps=1,
        min_epochs=4,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    if args.mode == 'train':
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if args.dev_run:
            print("Fast dev run completed successfully!")
            return None
        else:
            print("Test model")
            best_model_path = checkpoint_callback.best_model_path
            print(f"best_model_path: {best_model_path}")

            if best_model_path and os.path.exists(best_model_path):
                best_model = model_class.load_from_checkpoint(best_model_path, strict=False)

                if hasattr(model, 'tree_info') and model.tree_info is not None:
                    with open(os.path.join(tb_logger.log_dir, 'tree_info.pkl'), 'wb') as f:
                        pickle.dump(model.tree_info, f)

                trainer.test(model=best_model, dataloaders=test_loader)
                with open(os.path.join(tb_logger.log_dir, 'test_set_results.yaml'), 'w') as f:
                    yaml.dump(best_model.test_results, f)

                print("save success!")
                print(best_model.test_results)
                return best_model.test_results
            else:
                print("No checkpoint saved, skipping test")
                return None

    elif args.mode == 'test':
        print("Test model by dynamic")

        if not hasattr(args, 'best_model_path') or not args.best_model_path:
            print("Error: best_model_path is required for test mode")
            return None

        best_model = model_class.load_from_checkpoint(args.best_model_path, strict=False)
        best_model.hparams.update(model_params)
        best_model.eval()

        if hasattr(model, 'tree_info') and model.tree_info is not None:
            with open(os.path.join(tb_logger.log_dir, 'tree_info.pkl'), 'wb') as f:
                pickle.dump(model.tree_info, f)

        if not args.dev_run:
            trainer.test(model=best_model, dataloaders=test_loader)
            if args.dynamic if hasattr(args, 'dynamic') else False:
                dynamic_path = f"./logs/MLA_Files/dynamic_result/index-{args.index}"
                if not os.path.exists(dynamic_path):
                    os.makedirs(dynamic_path)
                with open(os.path.join(dynamic_path, f'index{args.index}-seed{args.seed}.yaml'), 'w') as f:
                    yaml.dump(best_model.test_results, f)
            else:
                with open(os.path.join(tb_logger.log_dir, 'test_set_results.yaml'), 'w') as f:
                    yaml.dump(best_model.test_results, f)

        return best_model.test_results

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    test_results = run_model(args)

