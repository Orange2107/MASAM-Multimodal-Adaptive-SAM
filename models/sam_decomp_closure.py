from copy import deepcopy
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from .util import enable_running_stats, disable_running_stats

class  SAMDecompClosure(Optimizer):
    def __init__(self, params, base_optimizer, model, rho=0.05, alpha=0.0, adaptive=True, perturb_eps=1e-12, rho_scheduler = None, pareto=False, dynamic=False,**kwargs):

        for group in params:
            group.setdefault("rho", rho)
            group.setdefault("apply_sagm", True)
            group.setdefault("name", "other")
            group.setdefault("sagm_alpha", alpha)

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAMDecompClosure, self).__init__(params, defaults)
        self.rho_scheduler = rho_scheduler
        self.model = model
        self.perturb_mode = "all"
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.original_params = {}
        self.original_grads = {}
        self.multi_gradients = {}
        self.uni_gradients = {}
        self.pareto = pareto
        self.e_w = []
        self.defaults.update(self.base_optimizer.defaults)
        self.alpha = alpha
        self.similarity_ehr = 0.0
        self.similarity_cxr = 0.0
        self.perturb_eps = perturb_eps
        self.forward_backward_func = None
        self.perturb_weak_mode = "all"
        self.dynamic = dynamic

    @torch.no_grad()
    def set_perturb_mode(self, perturb_mode):
        self.perturb_mode = perturb_mode
        self.perturb_weak_mode = "cxr" if perturb_mode == "ehr" else "ehr"

    @torch.no_grad()
    def store_module_gradients(self, module_name):
        gradients = {}
        for group in self.param_groups:
            if group['name'] == module_name:
                for p in group['params']:
                    if p.grad is not None:
                        gradients[p] = p.grad.clone().detach()
        return gradients

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets):

        self.multi_gradients = {}
        self.uni_gradients = {}
        def get_grad(only_multi = False):
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss_multi, loss_ehr, loss_cxr = loss_fn(outputs, targets)

                if not only_multi:
                    loss_multi.backward(retain_graph=True)
                    self.multi_gradients['ehr'] = self.store_module_gradients('ehr')
                    self.multi_gradients['cxr'] = self.store_module_gradients('cxr')
                    self.base_optimizer.zero_grad()

                    loss_ehr.backward(retain_graph=True)
                    self.uni_gradients['ehr'] = self.store_module_gradients('ehr')
                    self.base_optimizer.zero_grad()

                    loss_cxr.backward(retain_graph=True)
                    self.uni_gradients['cxr'] = self.store_module_gradients('cxr')
                    self.base_optimizer.zero_grad()

                total_loss = loss_multi + loss_ehr + loss_cxr
                total_loss.backward()

            return total_loss.item(), loss_ehr.item(), loss_cxr.item()

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            name = group['name']
            if name not in self.original_params:
                self.original_params[name] = {}
                self.original_grads[name] = {}
            for p in group['params']:
                if p.grad is not None:
                    self.original_params[name][p] = p.data.clone().detach()
                    self.original_grads[name][p] = p.grad.data.clone().detach()

        if self.dynamic:
            if self.perturb_mode == "all":
                self._perturb_all_params()
            elif self.perturb_mode == "ehr":
                self._perturb_specific_params("ehr")
            elif self.perturb_mode == "cxr":
                self._perturb_specific_params("cxr")
            elif self.perturb_mode == "other":
                self._perturb_specific_params("other")

        else:
            self._perturb_specific_params("cxr")
            self._perturb_specific_params("ehr")

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def _perturb_all_params(self, total_norm):

        print(f"total_norm: {total_norm}")
        for group in self.param_groups:

            scale = group["rho"] / (total_norm + self.perturb_eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    e_w *= torch.pow(p, 2)
                p.data.add_(e_w)

    @torch.no_grad()
    def _perturb_specific_params(self, group_name):
        total_cosine_similarity = 0.0
        total_angle = 0.0
        param_count = 0

        for group in self.param_groups:
            if group['name'] != group_name:
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if group_name == "other":
                    pass
                else:
                    g_u = self.uni_gradients[group_name][p]
                    g_m = self.multi_gradients[group_name][p]
                    factor = F.cosine_similarity(g_u.view(1, -1), g_m.view(1, -1)).item()
                    decomposed_grads = self._get_decomposed_gradients(g_u, g_m)
                    p.grad = decomposed_grads['uni_parallel_multi'].clone()

        specific_norm = self._grad_specific_norm(group_name)

        for group in self.param_groups:
            if group['name'] != group_name:
                continue

            scale = group["rho"] / (specific_norm + self.perturb_eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p) * factor if factor > 0 else p.grad * scale.to(p)
                if group["adaptive"]:
                    p.grad *= torch.pow(p, 2)
                p.data.add_(e_w)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            name = group['name']
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.original_params[name][p].to(p.device))

                if name == "other":
                    p.grad.data.copy_(self.original_grads[name][p])

        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        if not self.dynamic:
            loss_multi, loss_ehr, loss_cxr = get_grad()

        self.first_step(zero_grad=True)

        disable_running_stats(self.model)
        p_loss_multi, p_loss_ehr, p_loss_cxr = get_grad(only_multi=True)
        enable_running_stats(self.model)

        self.second_step(zero_grad=True)

        return p_loss_multi

    @torch.no_grad()
    def step_similarity(self,closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        loss_multi, loss_ehr, loss_cxr = get_grad()

        similarity_state = self._compute_gradient_similarity()

        self.base_optimizer.step()

        return similarity_state, loss_multi

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        ehr_norm = 0.0
        cxr_norm = 0.0
        other_norm = 0.0

        for group in self.param_groups:
            weighted_norm = torch.norm(
                    torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

            if group["name"] == "ehr":
                ehr_norm += weighted_norm
            elif group["name"] == "cxr":
                cxr_norm += weighted_norm
            elif group["name"] == "other":
                other_norm += weighted_norm
        total_norm = torch.sqrt(ehr_norm**2 + cxr_norm**2 + other_norm**2)
        return ehr_norm, cxr_norm, other_norm, total_norm

    @torch.no_grad()
    def _grad_specific_norm(self, group_name):
        shared_device = self.param_groups[0]["params"][0].device
        weighted_norm = 0.0
        for group in self.param_groups:
            if group["name"] == group_name:
                weighted_norm = torch.norm(
                        torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
                )

        return weighted_norm

    def _get_decomposed_gradients(self, g_u, g_m):
        dot_product = torch.dot(g_u.view(-1), g_m.view(-1))
        norm_u_squared = torch.norm(g_u) ** 2
        norm_m_squared = torch.norm(g_m) ** 2
        if dot_product <0:
            g_u_parallel_m = g_m.clone()
        else:
            g_u_parallel_m = (dot_product / norm_m_squared) * g_m

        g_u_vertical_m = g_u - g_u_parallel_m

        g_m_parallel_u = (dot_product / norm_u_squared) * g_u

        g_m_vertical_u = g_m - g_m_parallel_u

        sagm_like= ((g_u + g_m)*0.5)

        return {
            'uni_parallel_multi': g_u_parallel_m,
            'uni_vertical_multi': g_u_vertical_m,
            'multi_parallel_uni': g_m_parallel_u,
            'multi_vertical_uni': g_m_vertical_u,
            'sagm_like': sagm_like
        }

    @torch.no_grad()
    def _compute_gradient_similarity(self):

        similarity_state = {}
        for modality in ['ehr', 'cxr']:
            total_cosine_similarity = 0.0
            params_count = 0

            for p in self.uni_gradients[modality].keys():
                g_u = self.uni_gradients[modality][p]
                g_m = self.multi_gradients[modality][p]
                cosine_similarity = F.cosine_similarity(g_u.view(1, -1), g_m.view(1, -1))
                total_cosine_similarity += cosine_similarity.item()
                params_count += 1

            avg_cosine_similarity = total_cosine_similarity / params_count
            similarity_state[modality] = avg_cosine_similarity

        return similarity_state

