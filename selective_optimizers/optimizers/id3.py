import torch
from ..utils import get_not_chosen, set_chosen

def get_id3(base_optimizer):
    
    class Id3(base_optimizer):
        
        def __init__(self, *args, **kwargs):
            self.max_steps = kwargs.pop("max_steps", None)
            self.eps = kwargs.pop("eps", None)
            self.exp = kwargs.pop("exp", None)
            self.global_step = 0
            super().__init__(*args, **kwargs)
        
        def get_num_unmask(self):
            base_unmask = self.budget//self.max_steps
            rem_unmask = self.budget - base_unmask*self.max_steps
            # distribute rem_unmask uniformly
            if (rem_unmask):
                interval = rem_unmask/self.max_steps
                num_unmask = base_unmask+1 if self.global_step/self.max_steps < interval else base_unmask
            else:
                num_unmask = base_unmask
            return num_unmask
        
        @torch.no_grad()
        def init_chosen(self):
            for param_group in self.param_groups:
                params = param_group["params"]
                num_params = len(params)
                chosen_masks = []
                for i in range(num_params):
                    param = params[i]
                    mask = torch.zeros_like(param, device=param.device, dtype=torch.bool)
                    chosen_masks.append(mask)
                param_group["chosen_masks"] = chosen_masks

        @torch.no_grad()
        def compute_d3_metric(self, data, grad):
            metric = grad.abs()/(self.eps+data.abs())**self.exp
            return metric
        
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            self.global_step += 1
        
        @torch.no_grad()
        def update_chosen(self):
            num_unmask = self.get_num_unmask()
            metrics = []
            for param_group in self.param_groups:
                params = param_group["params"]
                chosen_masks = param_group["chosen_masks"]
                num_params = len(params)
                for i in range(num_params):
                    param = params[i]
                    chosen_mask = chosen_masks[i]
                    metric = self.compute_d3_metric(param.data, param.grad)
                    metric = get_not_chosen(metric, chosen_mask)
                    metrics.append(metric.flatten())
            metrics = torch.cat(metrics)
            cutoff = torch.topk(metrics, k=num_unmask, largest=True).values[-1]
            curr_chosen = 0
            for param_group in self.param_groups:
                params = param_group["params"]
                chosen_masks = param_group["chosen_masks"]
                num_params = len(params)
                for i in range(num_params):
                    param = params[i]
                    chosen_mask = chosen_masks[i]
                    metric = self.compute_d3_metric(param.data, param.grad)
                    mask = (metric>=cutoff)
                    earlier = chosen_mask.count_nonzero()
                    chosen_mask[mask] = 1
                    new = chosen_mask.count_nonzero()
                    added = new-earlier
                    if (added+curr_chosen > num_unmask):
                        num_mask = added+curr_chosen-num_unmask
                        # mask num_mask 1's in chosen_mask
                        indices = chosen_mask.nonzero()[:num_mask]
                        set_chosen(chosen_mask, indices, 0)
    return Id3
        