import torch
from torch import optim
import pandas as pd
import os
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

import lightning as L

class BaseFuseTrainer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_info = {}
        self.checkpoint_path = ""
        self.max_prauc = -1
        self.val_info = {'stay_ids': [], 'groups': [], 'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [],'meta_attrs': []}
        self.test_info = {'stay_ids': [], 'groups': [], 'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [], 'meta_attrs': []}

    def __get_batch_data(self, batch):

        for x in ['ehr_ts', 'cxr_imgs', 'labels']:
            batch[x] = batch[x].to(self.device)
        return batch

    def forward(self, data_dict):
        raise NotImplementedError('The `forward` method is not implemented.')

    def _shared_step(self, batch):
        batch = self.__get_batch_data(batch)
        out = self(batch)
        return out

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({'loss/train': out['loss'].detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])
        return out['loss']

    def _val_test_shared_step(self, batch, cache):

        out = self._shared_step(batch)
        cache['predictions'].append(out['predictions'])
        cache['labels'].append(batch['labels'])

        return out

    def _val_test_epoch_end(self, cache, clear_cache=True):

        scores = self.evaluate_performance(torch.cat(cache['predictions']),torch.cat(cache['labels']))

        if clear_cache:
            for x in cache:
                cache[x].clear()
        return scores

    def _get_ehr_cxr_scores(self, cache, clear_cache=True):
        scores_ehr = self.evaluate_performance(torch.cat(cache['pred_ehr']),torch.cat(cache['labels']))
        scores_cxr = self.evaluate_performance(torch.cat(cache['pred_cxr']),torch.cat(cache['labels']))
        if clear_cache:
            for x in cache:
                cache[x].clear()
        return scores_ehr,scores_cxr

    def validation_step(self, batch, batch_idx):
        out = self._val_test_shared_step(batch, self.val_info)
        self.log_dict({'loss/validation': out['loss'].detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])
        return out['loss']

    def test_step(self, batch, batch_idx):
        self._val_test_shared_step(batch, self.test_info)

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

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0)

        return optimizer

    def evaluate_performance(self, preds,labels):

        prauc = average_precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average=None)
        auroc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy(), average=None)

        if labels.shape[1] == 1:
            return {'overall/PRAUC': float(prauc), 'overall/AUROC': float(auroc)}

        scores = {'overall/PRAUC': float(prauc.mean()), 'overall/AUROC': float(auroc.mean())}
        if hasattr(self, 'class_names') and self.class_names is not None:
            class_names = self.class_names
        else:
            class_names = [f'Class_{i}' for i in range(labels.shape[1])]
        for i, name in enumerate(class_names):
            scores[f'class-wise_PRAUC/{name}'] = float(prauc[i])
            scores[f'class-wise_AUROC/{name}'] = float(auroc[i])
        return scores

    def evaluate_fairness(self, stay_ids, groups, predictions, labels):
        groups = pd.DataFrame(groups).loc[stay_ids]
        variables = groups.columns
        groups = groups.reset_index()
        scores = {}
        for var in variables:
            for value in groups[var].unique():
                indices = groups[groups[var]==value].index.values
                preds_, targets_ = predictions[indices], labels[indices]
                prauc_score = average_precision_score(targets_.cpu().numpy(), preds_.cpu().numpy(), average=None)
                auroc_score = roc_auc_score(targets_.cpu().numpy(), preds_.cpu().numpy(), average=None)
                scores['Group-PRAUC/'+var+'='+value] = prauc_score.tolist()
                scores['Group-AUROC/'+var+'='+value] = auroc_score.tolist()

                if labels.shape[-1] > 1:
                    scores['Group-classaverage/'+'PRAUC_'+var+'='+value] = float(prauc_score.mean())
                    scores['Group-classaverage/'+'AUROC_'+var+'='+value] = float(auroc_score.mean())
        return scores
