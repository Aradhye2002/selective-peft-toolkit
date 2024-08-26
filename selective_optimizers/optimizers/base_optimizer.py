import copy
import torch
from ..load_store import create_summary_from_param_groups

def get_base_optimizer(optimizer):
    
    class BaseOptimizer(optimizer):
        
        def __init__(self, *args, **kwargs):
            self.verify = kwargs.pop('verify', None)
            self.budget = kwargs.pop('budget', None)
            super().__init__(*args, **kwargs)
            self.init_chosen()
            self.pre_train_work()
            
        def pre_train_work(self):
            # Called automatically on init
            if self.verify:
                # store a copy of the initial param
                for param_group in self.param_groups:
                    param_group["initial_params"] = copy.deepcopy(param_group["params"])

        def get_budget_used(self):
            budget_used = 0
            for param_group in self.param_groups:
                if ("chosen_masks" in param_group):
                    for chosen in param_group["chosen_masks"]:
                        budget_used += chosen.count_nonzero()
                else:
                    for param in param_group["params"]:
                        budget_used += param.numel()
                        
            return budget_used
        
        def post_train_work(self):
            # Needs to be called by the user post-training
            if self.verify:
                budget_used = self.get_budget_used()
                if (budget_used > self.budget):
                    print("Budget exceeded! Budget allocated:{}, budget used: {}".format(self.budget, budget_used))
                for i, param_group in enumerate(self.param_groups):
                    num_params = len(param_group["params"])
                    for j in range(num_params):
                        param = param_group["params"][j]
                        chosen_mask =  param_group["chosen_masks"][j]
                        not_chosen_mask = torch.logical_not(chosen_mask)
                        initial_param =  param_group["initial_params"][j]
                        not_chosen_initial = initial_param.data[not_chosen_mask]
                        not_chosen = param.data[not_chosen_mask]
                        if (not_chosen_initial == not_chosen).all():
                            pass
                        else:
                            print("Verification failed for param {} in param_group {}".format(j, i))
        
        def mask_gradients(self):
            for param_group in self.param_groups:
                num_params = len(param_group["params"])
                for j in range(num_params):
                    param = param_group["params"][j]
                    grad = param.grad
                    chosen_mask =  param_group["chosen_masks"][j]
                    not_chosen_mask = torch.logical_not(chosen_mask)
                    grad[not_chosen_mask] = 0

        def step(self, *args, **kwargs):
            self.update_chosen()
            self.mask_gradients()
            super().step(*args, **kwargs)
            
        def init_chosen(self):
            # Must be necessarily overridden
            pass
        
        def update_chosen(self):
            # Must be necessarily overriden
            pass
        
        def get_summary(self, model):
            modules = model.modules()
            summary = create_summary_from_param_groups(self.param_groups, modules)
            return summary
        
    return BaseOptimizer