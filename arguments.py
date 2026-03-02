import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--model', type=str, default='masam', help='The model to run.')
    parser.add_argument('--mode', type=str, default='train',help='Train, Both Fix, Single Fix')
    parser.add_argument('--gpu', type=int, default=0, help='GPU card to use')

    parser.add_argument('--dataset', type=str, default='MIMIC',
                        choices=['MIMIC', 'CREMAD', 'KINETICS', 'FOOD101', 'URFUNNY', 'ADNI'],
                        help='Dataset to use')

    parser.add_argument('--new_data', action='store_true' ,help='use new data')
    parser.add_argument('--task', type=str, default='mortality',help='phenotype or mortality')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--ehr_root', type=str, help='Path to the data dir',
                        default='/research/miccai/datasets/mimiciv_multimodal_full/data')
    parser.add_argument('--resized_cxr_root', type=str, help='Path to the cxr data',
                        default='/research/mimic_cxr_resized')
    parser.add_argument('--cxr_root', type=str, help='Path to the cxr data',
                        default='/hdd/datasets/mimic-cxr-jpg/2.0.0/files')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--best_model_path', type=str)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--hidden_size_cxr', type=int, default=256)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_layers_distinct', type=int, default=1)
    parser.add_argument('--ehr_dropout', type=float, default=0.2)
    parser.add_argument('--cxr_n_layers', type=int, default=1)
    parser.add_argument('--cxr_dropout', type=float, default=0.2)

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--matched',action='store_true')
    parser.add_argument('--index',type=int, default=None)
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

    parser.add_argument('--fusion_method',type=str, default='concate')
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help='number of epoch to wait for best')
    parser.add_argument('--dynamic',action='store_true')

    parser.add_argument('--rho',type=float,default=0.3)
    parser.add_argument('--scale_alpha',type=float,default=0.01)
    parser.add_argument('--uniloss',action = 'store_true',help='uniloss')

    parser.add_argument('--no_checkpoint', action='store_false', dest='save_checkpoint')
    parser.add_argument('--save_checkpoint', action='store_true', default=True, help='Save model checkpoint')
    parser.add_argument('--dev_run', action='store_true')

    parser.add_argument('--save', action='store_true', default=False, help='Save representation')
    parser.add_argument('--sam_decomp', action='store_true', default=False, help='SAM-Decomp')
    parser.add_argument('--score_weight', type=float, default=0.5, help='Score weight')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--dynamic_mode', action='store_true', help='dynamic mode')
    parser.add_argument('--loss_multi', type=float, default=1.0, help='Loss multi')
    parser.add_argument('--loss_ehr', type=float, default=1.0, help='Loss ehr')
    parser.add_argument('--loss_cxr', type=float, default=1.0, help='Loss cxr')
    return parser
