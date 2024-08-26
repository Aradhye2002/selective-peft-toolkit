import torch
from ..utils import set_chosen

def get_pafi(base_optimizer):
    
    class Pafi(base_optimizer):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        @torch.no_grad()
        def init_chosen(self):
            all_params = []
            for param_group in self.param_groups:
                all_params.extend(param_group["params"])
            all_params = [param.data.flatten() for param in all_params]
            all_params = torch.cat(all_params)
            all_params = all_params.abs()
            cutoff = torch.topk(all_params, self.budget, largest=False).values[-1]
            total_chosen = 0
            for param_group in self.param_groups:
                params = param_group["params"]
                num_params = len(params)
                chosen_masks = []
                for i in range(num_params):
                    param = params[i]
                    data = param.data
                    choose = (data.abs() <= cutoff)
                    choose_indices = choose.nonzero()
                    num_chosen = len(choose_indices)
                    if (total_chosen + num_chosen <= self.budget):
                        total_chosen += num_chosen
                    else:
                        choose_indices = choose_indices[:self.budget-total_chosen]
                        total_chosen = self.budget
                    mask = torch.zeros_like(param, device=param.device, dtype=torch.bool)
                    set_chosen(mask, choose_indices, 1)
                    chosen_masks.append(mask)
                param_group["chosen_masks"] = chosen_masks
                
        @torch.no_grad()
        def update_chosen(self):
            pass     
        
    return Pafi  
        