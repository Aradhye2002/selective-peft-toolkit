import torch

def get_bitfit(base_optimizer):
    
    class Bitfit(base_optimizer):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        @torch.no_grad()
        def init_chosen(self):
            total_chosen = 0
            for param_group in self.param_groups:
                params = param_group["params"]
                num_params = len(params)
                chosen_masks = []
                for i in range(num_params):
                    param = params[i]
                    ndim = param.ndim
                    if ndim == 1:
                        mask = torch.ones_like(param, device=param.device, dtype=torch.bool)
                        # parameter is a bias term
                        num_chosen = mask.numel()
                        if (total_chosen + num_chosen <= self.budget):
                            total_chosen += num_chosen
                        else:
                            mask[-(total_chosen+num_chosen-self.budget):] = 0
                            total_chosen = self.budget
                    elif ndim == 0:
                        if (total_chosen+1 <= self.budget):
                            mask = torch.ones_like(param, device=param.device, dtype=torch.bool)
                        else:
                            mask = torch.zeros_like(param, device=param.device, dtype=torch.bool)
                    else:
                        mask = torch.zeros_like(param, device=param.device, dtype=torch.bool)
                    chosen_masks.append(mask)
                param_group["chosen_masks"] = chosen_masks
        
        @torch.no_grad()
        def update_chosen(self):
            pass
        
    return Bitfit  
        