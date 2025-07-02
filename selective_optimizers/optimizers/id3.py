import torch
from ..utils import get_not_chosen, set_chosen

def get_id3(base_optimizer):
    """
    Factory function to create the Id3 optimizer class, inheriting from base_optimizer.
    The Id3 optimizer selectively updates parameters based on the D3 metric.
    """

    class Id3(base_optimizer):
        """
        Id3 optimizer class that iteratively unfreezes parameters during optimization based on a custom metric.
        """

        def __init__(self, *args, **kwargs):
            """
            Initialize the Id3 optimizer.

            Args:
                *args: Positional arguments for the base optimizer.
                **kwargs: Keyword arguments for the base optimizer, including:
                    - max_steps (int): Maximum number of optimization steps.
                    - eps (float): Small epsilon value to avoid division by zero.
                    - exp (float): Exponent value used in the D3 metric computation.
            """
            # Extract custom arguments
            self.max_steps = kwargs.pop("max_steps", None)
            self.eps = kwargs.pop("eps", 1e-8)
            self.exp = kwargs.pop("exp", 1.0)

            # Initialize global step counter
            self.global_step = 0

            # Call the base optimizer's initializer
            super().__init__(*args, **kwargs)

            
        def get_num_unmask(self):
            """
            Calculate the number of parameters to unmask at the current optimization step.

            Returns:
                int: Number of parameters to unmask.
            """
            base_unmask = self.effective_budget // self.max_steps
            assert base_unmask > 0, "base_unmask should be non-zero 0"
            rem_unmask = self.effective_budget % self.max_steps  # Remaining parameters to unmask
            if self.global_step < rem_unmask:
                # For the first 'rem_unmask' steps, unmask one extra parameter
                num_unmask = base_unmask + 1
            else:
                num_unmask = base_unmask
            return num_unmask

        @torch.no_grad()
        def init_chosen(self):
            """
            Initialize the chosen_masks for each parameter group.
            """
            used_budget = 0
            for param_group in self.param_groups:
                if "choose_all" in param_group and param_group["choose_all"]:
                    mode = 0
                else:
                    mode = 2
                chosen_masks = []
                for param in param_group["params"]:
                    if param.requires_grad:
                        if mode == 0:
                            # Create a mask of ones (True), of the same shape as the parameter
                            mask = torch.ones_like(param.data, dtype=torch.bool, device=param.device)
                            used_budget += param.numel()
                        else:
                            # Create a mask of zeros (False), of the same shape as the parameter
                            mask = torch.zeros_like(param.data, dtype=torch.bool, device=param.device)
                    else:
                        mask = None
                    chosen_masks.append(mask)
                param_group["chosen_masks"] = chosen_masks
                
            if self.budget <= used_budget:
                raise Exception(f"Used initial budget: {used_budget} has exceeded total budget: {self.budget}")
            
            self.effective_budget = self.budget - used_budget
            
        @torch.no_grad()
        def compute_d3_metric(self, data, grad):
            """
            Compute the D3 metric for parameter selection.

            Args:
                data (torch.Tensor): The parameter data.
                grad (torch.Tensor): The gradient of the parameter.

            Returns:
                torch.Tensor: The computed metric.
            """
            metric = grad.abs() / (self.eps + data.abs()).pow(self.exp)
            return metric

        @torch.no_grad()
        def update_chosen(self):
            """
            Update the chosen masks based on the computed metrics.
            """
            # num_unmask needs to be non-zero to ensure k is not 0 in topk computation
            num_unmask = self.get_num_unmask()
            assert num_unmask > 0, "num_unmask should be non-zero"
            filtered_metrics = []
            device = None
            # Collect metrics from all parameters that are not yet chosen
            for param_group in self.param_groups:
                if "choose_all" in param_group and param_group["choose_all"]:
                    continue
                else:
                    pass
                for param, chosen_mask in zip(param_group["params"], param_group["chosen_masks"]):
                    if device is None:
                        device = param.device
                    if not param.requires_grad:
                        continue  # Skip parameters without gradients
                    metric = self.compute_d3_metric(param.data, param.grad)
                    # Only consider parameters that are not already chosen
                    metric = get_not_chosen(metric, chosen_mask)
                    cutoff = min(len(metric), num_unmask)
                    filtered_metric = torch.topk(metric, k=cutoff, largest=True, sorted=False).values.to(device)
                    filtered_metrics.append(filtered_metric)

            # Concatenate filtered metrics and find the cutoff value for unmasking
            filtered_metrics = torch.cat(filtered_metrics)
            if filtered_metrics.numel() < num_unmask:
                num_unmask = filtered_metrics.numel()
            cutoff_value = torch.topk(filtered_metrics, k=num_unmask, largest=True).values[-1].item()

            curr_chosen = 0  # Counter for currently chosen parameters
            stop = False
            # Update chosen masks based on the cutoff value
            for param_group in self.param_groups:
                if stop:
                    break
                if "choose_all" in param_group and param_group["choose_all"]:
                    continue
                else:
                    for param, chosen_mask in zip(param_group["params"], param_group["chosen_masks"]):
                        if chosen_mask is None:
                            continue  # Skip parameters without gradients
                        metric = self.compute_d3_metric(param.data, param.grad)
                        new_mask = (metric >= cutoff_value) & (~chosen_mask)
                        added = new_mask.sum().item()
                        chosen_mask[new_mask] = True
                        curr_chosen += added

                        # If we've chosen more parameters than allowed, un-choose some
                        if curr_chosen > num_unmask:
                            excess = curr_chosen - num_unmask
                            # Find indices of chosen parameters to unmask
                            indices_to_unmask = new_mask.nonzero()[:excess]
                            set_chosen(chosen_mask, indices_to_unmask, False)
                            curr_chosen -= excess
                            if curr_chosen == num_unmask:
                                stop = True
                                break  # Exit early if we've reached the budget

    return Id3
