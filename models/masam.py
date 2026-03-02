
from .sam_decomp_closure import SAMDecompClosure
import math
import torch
import os
from copy import deepcopy
from torch import nn
import numpy  as  np
import torch.nn.init as init
from functools import partial
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import vit_l_32, ViT_L_32_Weights
from .fusion_method import SumFusion, ConcatFusion, FiLM
from .ehr_transformer import DisentangledEHRTransformer
from .base_fusion import BaseFuseTrainer
from .backbone import resnet18
from .FNNEncoder import FFNEncoder
import torch.nn.functional as F
from .Urfunny_Encoder import _URTransformer

class MASAM(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.class_names = self.hparams['class_names']
        self.is_urfunny = False

        if self.hparams['dataset'] == 'CREMAD':
            print(f"CREMAD dataset")
            self.num_classes = 6
            self.criterion = nn.CrossEntropyLoss()
            self.is_cremad = True
            self.is_food101 = False
            self.is_adni = False
        elif self.hparams['dataset'] == 'KINETICS':
            print(f"KINETICS dataset")
            self.num_classes = 31
            self.criterion = nn.CrossEntropyLoss()
            self.is_cremad = True
            self.is_food101 = False
            self.is_adni = False
        elif self.hparams['dataset'] == 'FOOD101':
            print(f"FOOD101 dataset")
            self.num_classes = 101
            self.criterion = nn.CrossEntropyLoss()
            self.is_cremad = False
            self.is_food101 = True
            self.is_adni = False
        elif self.hparams['dataset'] == 'URFUNNY':
            print(f"URFUNNY dataset")
            self.num_classes = 2
            self.criterion = nn.CrossEntropyLoss()
            self.is_cremad = False
            self.is_food101 = False
            self.is_adni = False
            self.is_urfunny = True
        elif self.hparams['dataset'] == 'ADNI':
            print(f"ADNI dataset")
            self.num_classes = 3
            self.criterion = nn.CrossEntropyLoss()
            self.is_cremad = False
            self.is_food101 = False
            self.is_adni = True
        else:
            self.pred_criterion = nn.BCELoss(reduction='none')
            self.num_classes = 1 if self.hparams.task == 'mortality' else 25
            self.is_cremad = False
            self.is_food101 = False
            self.is_adni = False

        self._step_counter = 0
        self.loss_history = {name: [] for name in self.class_names}
        self.loss_history_ehr = {name: [] for name in self.class_names}
        self.loss_history_cxr = {name: [] for name in self.class_names}
        self.losses_per_label = {name: [] for name in self.class_names}
        self.losses_per_label_ehr = {name: [] for name in self.class_names}
        self.losses_per_label_cxr = {name: [] for name in self.class_names}

        if self.hparams['sam_decomp']:
            self.automatic_optimization = False
            print("Automatic optimization is set to False")

        fusion_input_dim = self.hparams.hidden_size * 2
        if self.hparams['fusion_method'] == 'sum':
            self.fusion_module = SumFusion(input_dim=self.hparams.hidden_size, output_dim=self.num_classes)
        elif self.hparams['fusion_method'] == 'concate' or self.hparams['fusion_method'] == 'single':
            self.fusion_module = ConcatFusion(input_dim=fusion_input_dim, output_dim=self.num_classes)
        elif self.hparams['fusion_method'] == 'film':
            self.fusion_module = FiLM(input_dim=self.hparams.hidden_size, dim=self.hparams.hidden_size, output_dim=self.num_classes, x_film=True)

        if self.is_cremad:
            self.ehr_model = resnet18(modality='audio')
            self.cxr_model_spec = resnet18(modality='visual')
        elif self.is_urfunny:
            embed_dim = self.hparams.hidden_size
            self.ur_audio_encoder = _URTransformer(n_features=81, dim=embed_dim, n_head=8, n_layers=4)
            self.ur_visual_encoder = _URTransformer(n_features=371, dim=embed_dim, n_head=8, n_layers=4)
            self.ur_text_encoder = _URTransformer(n_features=300, dim=embed_dim, n_head=8, n_layers=4)
            self.ur_head = nn.Linear(embed_dim * 3, self.num_classes)
            self.audio_head = nn.Linear(embed_dim, self.num_classes)
            self.visual_head = nn.Linear(embed_dim, self.num_classes)
            self.text_head = nn.Linear(embed_dim, self.num_classes)
        elif self.is_food101:
            from .ehr_transformer import TransformerEncoder
            self.ehr_model = TransformerEncoder(
                input_size=768, num_classes=self.num_classes,
                d_model=self.hparams.hidden_size,
                n_head=self.hparams.get('ehr_n_head', 4),
                n_layers=self.hparams.get('ehr_n_layers', 1),
                dropout=self.hparams.get('ehr_dropout', 0.2)
            )
            if self.hparams.cxr_model == 'resnet50':
                self.cxr_model_spec = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.cxr_model_spec.fc = nn.Linear(in_features=2048, out_features=self.hparams.hidden_size)
            elif self.hparams.cxr_model == 'swin_s':
                self.cxr_model_spec = swin_s(weights=Swin_S_Weights.DEFAULT)
                self.cxr_model_spec.head = nn.Linear(in_features=self.cxr_model_spec.head.in_features,
                                                    out_features=self.hparams.hidden_size)
            elif self.hparams.cxr_model == 'swin_t':
                self.cxr_model_spec = swin_t(weights=Swin_T_Weights.DEFAULT)
                self.cxr_model_spec.head = nn.Linear(in_features=self.cxr_model_spec.head.in_features,
                                                    out_features=self.hparams.hidden_size)
            else:
                self.cxr_model_spec = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.cxr_model_spec.fc = nn.Linear(in_features=2048, out_features=self.hparams.hidden_size)
        elif self.is_adni:
            self.ehr_model = FFNEncoder(input_dim=43, hidden_dim=512, output_dim=self.hparams.hidden_size, num_layers=4, dropout_prob=0.2)

            if self.hparams.get('mri_model', 'mlp') == 'resnet50':
                self.cxr_model_spec = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.cxr_model_spec.fc = nn.Linear(in_features=2048, out_features=self.hparams.hidden_size)
            elif self.hparams.get('mri_model', 'mlp') == 'resnet18':
                self.cxr_model_spec = resnet18(modality='visual')
                self.cxr_model_spec.fc = nn.Linear(in_features=512, out_features=self.hparams.hidden_size)
            else:
                self.cxr_model_spec = FFNEncoder(input_dim=185, hidden_dim=self.hparams.hidden_size_cxr, output_dim=self.hparams.hidden_size, num_layers=self.hparams.cxr_n_layers, dropout_prob=self.hparams.cxr_dropout)
        else:
            ehr_input_size = self.hparams.get('ehr_input_size', 24)
            self.ehr_model = DisentangledEHRTransformer(input_size=ehr_input_size, num_classes=self.num_classes,
                                        d_model=self.hparams.hidden_size, n_head=self.hparams.ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=1,
                                        n_layers_distinct=self.hparams.ehr_n_layers_distinct,
                                        dropout=self.hparams.ehr_dropout,simple=True)
            self.cxr_model_spec = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.cxr_model_spec.fc = nn.Linear(self.cxr_model_spec.fc.in_features, self.hparams.hidden_size)

        if self.hparams['uniloss']:
            print(f"inter method uniloss")
            self.ehr_model_linear = nn.Linear(in_features=self.hparams.hidden_size, out_features=self.num_classes)
            self.cxr_model_linear = nn.Linear(in_features=self.hparams.hidden_size, out_features=self.num_classes)

        self.score_ehr = 0
        self.score_cxr = 0
        self.save_batch = []
        self.trace_ehr = []
        self.trace_cxr = []
        self.feat_ehr_encoder_list = []
        self.feat_ehr_distinct_list = []
        self.feat_ehr_shared_list = []
        self.feat_cxr_distinct_list = []
        self.feat_cxr_shared_list =[]
        self.eigenvalue_ehr = []
        self.eigenvalue_cxr = []

        if self.hparams['sam_decomp']:
            self.ma_loss = {
                'ehr':None,
                'cxr':None
            }

            self.last_loss = {
                'ehr':None,
                'cxr':None
            }

            self.loss_decay = {
                'ehr':None,
                'cxr':None
            }

            self.grad_similarity = {
                'ehr':None,
                'cxr':None
            }

            self.score_weight = self.hparams.get('score_weight',0.5)
            self.momentum = self.hparams.get('momentum',0.9)

            if self.is_urfunny:
                if 'text' not in self.ma_loss:
                    self.ma_loss['text'] = None
                if 'text' not in self.last_loss:
                    self.last_loss['text'] = None
                if 'text' not in self.loss_decay:
                    self.loss_decay['text'] = None
                if 'text' not in self.grad_similarity:
                    self.grad_similarity['text'] = None

    def calculate_score(self,labels,outputs,pairs=None):

        if self.hparams.task == 'mortality':
            labels = labels.float()
            pred_ehr = outputs['pred_ehr']
            pred_cxr = outputs['pred_cxr']

            match_score_ehr = labels * pred_ehr + (1 - labels) * (1 - pred_ehr)
            match_score_cxr = labels * pred_cxr + (1 - labels) * (1 - pred_cxr)

            score_ehr = match_score_ehr.sum()
            score_cxr = match_score_cxr.sum()

            return score_ehr,score_cxr
        elif self.hparams.task == 'phenotype':
            labels = labels.float()
            pred_ehr = outputs['pred_ehr']
            pred_cxr = outputs['pred_cxr']

            match_score_ehr = labels * pred_ehr + (1 - labels) * (1 - pred_ehr)
            match_score_cxr = labels * pred_cxr + (1 - labels) * (1 - pred_cxr)

            match_score_ehr = match_score_ehr.sum(dim=1)
            match_score_cxr = match_score_cxr.sum(dim=1)

            score_ehr = match_score_ehr.sum()
            score_cxr = match_score_cxr.sum()

            return score_ehr, score_cxr

    def training_step(self,batch,batch_idx):
        current_step = self._step_counter
        if self.hparams['sam_decomp']:
            current_step = self._step_counter

            def loss_fn(predictions, targets):
                pairs = torch.ones_like(predictions['predictions'][:, 0])
                loss_pred_multi = self._compute_masked_pred_loss(predictions['predictions'], targets, pairs)
                if self.is_urfunny:
                    loss_pred_ehr = self._compute_masked_pred_loss(predictions['pred_ehr'], targets, pairs)
                    loss_pred_cxr = self._compute_masked_pred_loss(predictions['pred_cxr'], targets, pairs)
                    loss_pred_text = self._compute_masked_pred_loss(predictions['pred_text'], targets, pairs)
                    return (
                        self.hparams.loss_multi * loss_pred_multi,
                        self.hparams.loss_ehr * loss_pred_ehr,
                        self.hparams.loss_cxr * loss_pred_cxr,
                        self.hparams.get('loss_text', 1.0) * loss_pred_text,
                    )
                else:
                    loss_pred_ehr = self._compute_masked_pred_loss(predictions['pred_ehr'], targets, pairs)
                    loss_pred_cxr = self._compute_masked_pred_loss(predictions['pred_cxr'], targets, pairs)
                    return (
                        self.hparams.loss_multi * loss_pred_multi,
                        self.hparams.loss_ehr * loss_pred_ehr,
                        self.hparams.loss_cxr * loss_pred_cxr,
                    )

            self.SAM_decomp_optimizer.set_closure(loss_fn, batch, batch['labels'])

            if self.hparams.dynamic_mode:
                vals = self.SAM_decomp_optimizer.forward_backward_func()
                if self.is_urfunny:
                    loss_multi, loss_ehr, loss_cxr, loss_text = vals
                    modal_list = ['ehr', 'cxr', 'text']
                    modal_losses = {'ehr': loss_ehr, 'cxr': loss_cxr, 'text': loss_text}
                else:
                    loss_multi, loss_ehr, loss_cxr = vals
                    modal_list = ['ehr', 'cxr']
                    modal_losses = {'ehr': loss_ehr, 'cxr': loss_cxr}

                for modality in modal_list:
                    loss_value = modal_losses[modality]
                    if self.ma_loss[modality] is None:
                        self.ma_loss[modality] = loss_value
                        self.loss_decay[modality] = 0.0
                    else:
                        previous_loss = self.ma_loss[modality]
                        self.ma_loss[modality] = self.momentum * previous_loss + (1 - self.momentum) * loss_value
                        self.loss_decay[modality] = max(0.0, self.last_loss[modality] - self.ma_loss[modality])
                    self.last_loss[modality] = loss_value

                grad_similarity = self.SAM_decomp_optimizer._compute_gradient_similarity()
                self.grad_similarity.update(grad_similarity)

                scores = {}
                for modality in modal_list:
                    sim = self.grad_similarity.get(modality, 0.0) if isinstance(self.grad_similarity, dict) else 0.0
                    scores[modality] = self.score_weight * self.loss_decay[modality] + (1 - self.score_weight) * sim

                select_modality = max(scores, key=lambda k: scores[k])

                self.SAM_decomp_optimizer.set_perturb_mode(select_modality)

            self._step_counter += 1
            loss_multi = self.SAM_decomp_optimizer.step()

            if self.hparams.dynamic_mode:
                if self.is_urfunny:
                    epoch_log = {
                        'loss/train': loss_multi,
                        'epoch_num': float(self.current_epoch),
                        'similarity_ehr': grad_similarity.get('ehr', 0.0),
                        'similarity_cxr': grad_similarity.get('cxr', 0.0),
                        'similarity_text': grad_similarity.get('text', 0.0),
                        'score_ehr': scores.get('ehr', 0.0),
                        'score_cxr': scores.get('cxr', 0.0),
                        'score_text': scores.get('text', 0.0),
                    }
                else:
                    epoch_log = {
                        'loss/train': loss_multi,
                        'epoch_num': float(self.current_epoch),
                        'similarity_ehr': grad_similarity['ehr'],
                        'similarity_cxr': grad_similarity['cxr'],
                        'score_ehr': scores['ehr'],
                        'score_cxr': scores['cxr'],
                    }
                self.log_dict(epoch_log, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0])
            else:
                epoch_log = {
                    'loss/train': loss_multi,
                    'epoch_num': float(self.current_epoch),
                }
                self.log_dict(epoch_log, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0])
            return None

        elif self.hparams['draw_sam_iamge']:
            optimizer = self.optimizers()
            self._step_counter += 1
            print(f"cureent steo is {current_step}")

            state_dict = {k: v.clone().detach() for k, v in self.state_dict().items()}
            optimize_dict = deepcopy(optimizer.state_dict())

            out = self._shared_step(batch)

            pairs = torch.ones_like(out['predictions'][:, 0])
            loss_total_original = self._compute_and_log_loss(out, batch['labels'], pairs, mode='train', where='training_step', name="original", step=current_step)

            self.eval()
            optimizer.set_perturb_mode('ehr')

            out = self._shared_step(batch)
            pairs = torch.ones_like(out['predictions'][:, 0])
            loss_total =  self._compute_and_log_loss(out, batch['labels'], pairs,mode='train', where='training_step',name="ehr_original", step=current_step)
            loss_total.backward()

            optimizer.first_step(zero_grad=True)

            perturbed_out = self._shared_step(batch)
            loss_perturbed =  self._compute_and_log_loss(perturbed_out, batch['labels'], pairs,mode='train', where='training_step', name="ehr_perturb", step=current_step)
            loss_perturbed.backward()
            optimizer.second_step(zero_grad=True)
            grad_similarities_ehr = optimizer.get_similarity(current_step)[0]

            optimizer.set_perturb_mode('cxr')

            out = self._shared_step(batch)
            pairs = torch.ones_like(out['predictions'][:, 0])
            loss_total =  self._compute_and_log_loss(out, batch['labels'], pairs,mode='train', where='training_step',name="cxr_original", step=current_step)
            loss_total.backward()
            optimizer.first_step(zero_grad=True)
            perturbed_out = self._shared_step(batch)
            loss_perturbed =  self._compute_and_log_loss(perturbed_out, batch['labels'], pairs,mode='train', where='training_step', name="cxr_perturb", step=current_step)
            loss_perturbed.backward()
            optimizer.second_step(zero_grad=True)
            grad_similarities_cxr = optimizer.get_similarity(current_step)[1]

            self.train()
            self.load_state_dict(state_dict)
            optimizer.load_state_dict(optimize_dict)
            self.zero_grad()
            enable_mode_specific_behavior(self)

            out = self._shared_step(batch)
            pairs = torch.ones_like(out['predictions'][:, 0])
            loss_total_original =  self._compute_and_log_loss(out, batch['labels'], pairs,mode='train', where='training_step',name="original", step=current_step)

            loss_total_original.backward()
            optimizer.normal_step(zero_grad=True)

            epoch_log = {}
            epoch_log.update({
                'loss/train': loss_total.detach(),
                'step': float(self.current_epoch),
                'grad/ehr_fusion_sim': grad_similarities_ehr,
                'grad/cxr_fusion_sim': grad_similarities_cxr,
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log,on_epoch=True, on_step=False,batch_size=batch['labels'].shape[0])

            return None

        else:
            current_step = self.global_step

            out = self._shared_step(batch)

            pairs = torch.ones_like(out['predictions'][:, 0])
            pairs = torch.ones_like(out['predictions'][:, 0])
            loss_total =  self._compute_and_log_loss(out, batch['labels'], pairs,mode='train', where='training_step', name="original", step=current_step)

            epoch_log = {}
            epoch_log.update({
                'loss/train': loss_total.detach(),
                'epoch_num': float(self.current_epoch),
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False , batch_size=batch['labels'].shape[0])
            return loss_total

    def validation_step(self, batch, batch_idx):
        self.eval()
        device = next(self.parameters()).device
        batch['labels'] = batch['labels'].to(device)

        out = self._val_test_shared_step(batch, self.val_info)

        pairs = torch.ones_like(out['feat_ehr_distinct'][:, 0], device=device)
        loss_pred_multi = self._compute_masked_pred_loss(out['predictions'], batch['labels'], pairs)
        loss_pred_ehr = self._compute_masked_pred_loss(out['pred_ehr'], batch['labels'], pairs)
        loss_pred_cxr = self._compute_masked_pred_loss(out['pred_cxr'], batch['labels'], pairs)
        loss_total = loss_pred_multi + loss_pred_ehr + loss_pred_cxr

        epoch_log = {
            'loss/validation': loss_total.detach(),
            'loss/valid_multi': loss_pred_multi.detach(),
            'loss/valid_ehr': loss_pred_ehr.detach(),
            'loss/valid_cxr': loss_pred_cxr.detach(),
            'step': float(self.current_epoch),
        }
        self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=batch['labels'].shape[0])

        return loss_total

    def test_step(self, batch, batch_idx):

        out = self._val_test_shared_step(batch, self.test_info)

    def _compute_masked_pred_loss(self, input, target, mask):
        if self.is_cremad or self.is_food101 or self.is_adni or self.is_urfunny:
            return self.criterion(input, target)
        else:
            return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train', where="None input", name="default", step=0):
        if self.is_cremad or self.is_food101 or self.is_adni or self.is_urfunny:
            loss_pred_multi = self.criterion(model_output['predictions'], y_gt)
            loss_pred_ehr = self.criterion(model_output['pred_ehr'], y_gt)
            loss_pred_cxr = self.criterion(model_output['pred_cxr'], y_gt)
            loss_pred_text = None
            if self.is_urfunny and ('pred_text' in model_output):
                loss_pred_text = self.criterion(model_output['pred_text'], y_gt)
        else:
            ehr_mask = torch.ones_like(model_output['predictions'][:, 0])
            loss_pred_multi = self._compute_masked_pred_loss(model_output['predictions'], y_gt, ehr_mask)
            loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
            loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)

        if self.hparams['uniloss']:
            if self.is_urfunny and (loss_pred_text is not None):
                w_text = self.hparams.get('loss_text', 1.0)
                loss_pred_final = loss_pred_multi + loss_pred_ehr + loss_pred_cxr + w_text * loss_pred_text
            else:
                loss_pred_final = loss_pred_multi + loss_pred_ehr + loss_pred_cxr
        else:
            loss_pred_final = loss_pred_multi

        if log:
            items = {
                f'{mode}_loss/pred_final': loss_pred_multi.detach(),
                f'{mode}_loss/pred_ehr': loss_pred_ehr.detach(),
                f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
                'step': float(self.current_epoch)
            }
            if self.is_urfunny and (loss_pred_text is not None):
                items[f'{mode}_loss/pred_text'] = loss_pred_text.detach()
            self.log_dict(items, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])
        return loss_pred_final

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train', where="None Input", name="default", step=0):
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode, where, name, step)
        return prediction_losses

    def forward(self, data_dict):
        if self.is_urfunny:
            vision = data_dict['vision']
            audio = data_dict['audio']
            text = data_dict['text']

            feat_audio = self.ur_audio_encoder(audio)
            feat_visual = self.ur_visual_encoder(vision)
            feat_text = self.ur_text_encoder(text)

            pred_final = self.ur_head(torch.cat([feat_audio, feat_visual, feat_text], dim=1))
            pred_audio = self.audio_head(feat_audio)
            pred_visual = self.visual_head(feat_visual)
            pred_text = self.text_head(feat_text)

            feat_ehr_distinct = feat_audio
            feat_cxr_distinct = feat_visual
            pred_ehr = pred_audio
            pred_cxr = pred_visual

            outputs = {
                'predictions': pred_final,
                'feat_ehr_distinct': feat_ehr_distinct,
                'feat_cxr_distinct': feat_cxr_distinct,
                'pred_ehr': pred_ehr,
                'pred_cxr': pred_cxr,
                'pred_text': pred_text,
                'feat_text': feat_text,
            }
            return outputs
        if self.is_cremad:
            audio = data_dict['audio']
            vedio = data_dict['vedio']

            a = self.ehr_model(audio)
            v = self.cxr_model_spec(vedio)
            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            feat_ehr_distinct = torch.flatten(a, 1)
            feat_cxr_distinct = torch.flatten(v, 1)

            _, _, pred_final = self.fusion_module(feat_ehr_distinct, feat_cxr_distinct)
            weight_size = self.fusion_module.fc_out.weight.size(1)
            pred_cxr_inner = (torch.mm(feat_cxr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)
            pred_ehr_inner = (torch.mm(feat_ehr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)

            if self.hparams['uniloss']:
                feat_ehr_distinct_pred = self.ehr_model_linear(feat_ehr_distinct)
                feat_cxr_distinct_pred = self.cxr_model_linear(feat_cxr_distinct)
                pred_ehr = feat_ehr_distinct_pred
                pred_cxr = feat_cxr_distinct_pred
            else:
                pred_ehr = pred_ehr_inner
                pred_cxr = pred_cxr_inner

            pred_final = pred_final

        elif self.is_food101:
            bert_features = data_dict['bert_features']
            images = data_dict['images']

            batch_size, seq_len, feature_dim = bert_features.shape
            seq_lengths = [seq_len] * batch_size
            feat_ehr_distinct = self.ehr_model(bert_features, seq_lengths)

            feat_cxr_distinct = self.cxr_model_spec(images)

            _, _, pred_final = self.fusion_module(feat_ehr_distinct, feat_cxr_distinct)
            weight_size = self.fusion_module.fc_out.weight.size(1)
            pred_cxr_inner = (torch.mm(feat_cxr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)
            pred_ehr_inner = (torch.mm(feat_ehr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)

            if self.hparams['uniloss']:
                feat_ehr_distinct_pred = self.ehr_model_linear(feat_ehr_distinct)
                feat_cxr_distinct_pred = self.cxr_model_linear(feat_cxr_distinct)
                pred_ehr = feat_ehr_distinct_pred
                pred_cxr = feat_cxr_distinct_pred
            else:
                pred_ehr = pred_ehr_inner
                pred_cxr = pred_cxr_inner

            pred_final = pred_final

        elif self.is_adni:
            ehr_features = data_dict['ehr_features']

            feat_ehr_distinct = self.ehr_model(ehr_features)

            if self.hparams.get('mri_model', 'mlp') in ['resnet50', 'resnet18']:
                mri_images = data_dict['mri_images']
                feat_cxr_distinct = self.cxr_model_spec(mri_images)
            else:
                mri_features = data_dict['mri_values']
                feat_cxr_distinct = self.cxr_model_spec(mri_features)

            _, _, pred_final = self.fusion_module(feat_ehr_distinct, feat_cxr_distinct)
            weight_size = self.fusion_module.fc_out.weight.size(1)
            pred_cxr_inner = (torch.mm(feat_cxr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)
            pred_ehr_inner = (torch.mm(feat_ehr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)

            if self.hparams['uniloss']:
                print("into uniloss")
                feat_ehr_distinct_pred = self.ehr_model_linear(feat_ehr_distinct)
                feat_cxr_distinct_pred = self.cxr_model_linear(feat_cxr_distinct)
                pred_ehr = feat_ehr_distinct_pred
                pred_cxr = feat_cxr_distinct_pred
            else:
                pred_ehr = pred_ehr_inner
                pred_cxr = pred_cxr_inner

            pred_final = pred_final

        else:
            x = data_dict['ehr_ts']
            img = data_dict['cxr_imgs']

            seq_lengths = data_dict['seq_len']
            pairs = data_dict['has_cxr']

            feat_ehr_distinct,_ = self.ehr_model(x, seq_lengths)
            feat_cxr_distinct = self.cxr_model_spec(img)

            _, _, pred_final = self.fusion_module(feat_ehr_distinct, feat_cxr_distinct)
            weight_size = self.fusion_module.fc_out.weight.size(1)
            pred_cxr_inner = (torch.mm(feat_cxr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)
            pred_ehr_inner = (torch.mm(feat_ehr_distinct, torch.transpose(self.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) +
                            self.fusion_module.fc_out.bias / 2)

            if self.hparams['uniloss']:
                feat_ehr_distinct_pred = self.ehr_model_linear(feat_ehr_distinct)
                feat_cxr_distinct_pred = self.cxr_model_linear(feat_cxr_distinct)
                pred_ehr = feat_ehr_distinct_pred.sigmoid()
                pred_cxr = feat_cxr_distinct_pred.sigmoid()
            else:
                pred_ehr = pred_ehr_inner.sigmoid()
                pred_cxr = pred_cxr_inner.sigmoid()

            pred_final = pred_final.sigmoid()

        outputs = {
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'predictions': pred_final,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
        }

        return outputs

    def on_train_epoch_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        scores_ehr,scores_cxr = self._get_ehr_cxr_scores(self.val_info,clear_cache=False)
        scores = self._val_test_epoch_end(self.val_info,clear_cache=True)
        scores_ehr_prefixed = {f"ehr_{k}": v for k, v in scores_ehr.items()}
        scores_cxr_prefixed = {f"cxr_{k}": v for k, v in scores_cxr.items()}
        combined_scores = {**scores, **scores_ehr_prefixed, **scores_cxr_prefixed}
        combined_scores['step'] = float(self.current_epoch)
        self.log_dict({k: v for k, v in combined_scores.items() if not isinstance(v, list)}, on_epoch=True, on_step=False)

        return scores

    def on_test_epoch_end(self):
        scores = self._val_test_epoch_end(self.test_info,clear_cache=False)
        scores_ehr,scores_cxr = self._get_ehr_cxr_scores(self.test_info,clear_cache=True)
        scores_ehr_prefixed = {f"ehr_{k}": v for k, v in scores_ehr.items()}
        scores_cxr_prefixed = {f"cxr_{k}": v for k, v in scores_cxr.items()}
        combined_scores = {**scores, **scores_ehr_prefixed, **scores_cxr_prefixed}
        self.test_results = {x: combined_scores[x] for x in combined_scores}

    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)
        cache['predictions'].append(out['predictions'].detach())
        cache['pred_ehr'].append(out['pred_ehr'].detach())
        cache['pred_cxr'].append(out['pred_cxr'].detach())
        cache['labels'].append(batch['labels'].detach())
        return out

    def configure_optimizers(self):
        if self.hparams['sam_decomp']:
            print("into sam-decomp")

            optimizer_choice = str(self.hparams.get('optimizer', 'adamw')).lower()
            use_sgd_optimizer = optimizer_choice == 'sgd'

            if use_sgd_optimizer:
                print("into sgd optimizer")
                print(f"learning_rate is {self.hparams.learning_rate}, weight_decay is {self.hparams.weight_decay}")
                base_optimizer = partial(
                    torch.optim.SGD,
                    lr=self.hparams.learning_rate,
                    momentum=0.9,
                    weight_decay=self.hparams.weight_decay,
                )
                sam_kwargs = {
                    "alpha": self.hparams.get('scale_alpha', 0.01),
                }
            else:
                base_optimizer = torch.optim.AdamW
                print("into adamw optimizer")
                print(f"learning_rate is {self.hparams.learning_rate}, weight_decay is {self.hparams.weight_decay}")
                sam_kwargs = {
                    "lr": self.hparams["learning_rate"],
                    "weight_decay": self.hparams.weight_decay,
                    "alpha": self.hparams.get('scale_alpha', 0.01),
                }

            model_params = list(self.parameters())
            if self.is_urfunny:
                ehr_params = list(self.ur_audio_encoder.parameters())
                cxr_params = list(self.ur_visual_encoder.parameters())
                text_params = list(self.ur_text_encoder.parameters())
                used = set(list(ehr_params) + list(cxr_params) + list(text_params))
                other_params = [p for p in model_params if p not in used]

                param_groups = [
                    {"params": ehr_params, "name": "ehr", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                    {"params": cxr_params, "name": "cxr", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                    {"params": text_params, "name": "text", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                    {"params": other_params, "name": "other", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                ]
            else:
                ehr_params = list(self.ehr_model.parameters())
                cxr_params = list(self.cxr_model_spec.parameters())
                other_params = [p for p in model_params if p not in set(ehr_params) and p not in set(cxr_params)]

                param_groups = [
                    {"params": ehr_params, "name": "ehr", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                    {"params": cxr_params, "name": "cxr", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                    {"params": other_params, "name": "other", "adaptive": False, "rho": self.hparams.get('rho', 0.1), "apply_sam": False, "sagm_alpha": self.hparams.get('sagm_alpha', 0.5)},
                ]

            self.SAM_decomp_optimizer = SAMDecompClosure(
                params=param_groups,
                base_optimizer=base_optimizer,
                model=self,
                pareto=False,
                dynamic=self.hparams.dynamic_mode,
                **sam_kwargs
            )

            if use_sgd_optimizer:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.SAM_decomp_optimizer,
                    step_size=70,
                    gamma=0.1
                )
                return [self.SAM_decomp_optimizer], [scheduler]

            return self.SAM_decomp_optimizer

        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0)
            return optimizer

