import torch

def get_bitfit(base_optimizer):
    """
    Factory function to create the BitFit optimizer class, inheriting from base_optimizer.
    The BitFit optimizer updates only the bias terms in the model.
    """

    class Bitfit(base_optimizer):
        """
        BitFit optimizer class that fine-tunes only the bias parameters in the model.
        """

        def __init__(self, *args, **kwargs):
            """
            Initialize the BitFit optimizer.

            Args:
                *args: Positional arguments for the base optimizer.
                **kwargs: Keyword arguments for the base optimizer.
            """
            super().__init__(*args, **kwargs)

        @torch.no_grad()
        def init_chosen(self):
            """
            Initialize the chosen masks for each parameter, selecting only bias terms.
            """
            total_chosen = 0
            for param_group in self.param_groups:
                params = param_group["params"]
                chosen_masks = []
                for param in params:
                    if param.requires_grad:
                        if param_group.get("choose_all", False):
                            mask = torch.ones_like(param.data, dtype=torch.bool, device=param.device)
                            total_chosen += mask.numel()
                        else:
                            param_size = param.numel()

                            if param.ndim == 1:
                                # Assume 1D parameters are bias terms
                                num_chosen = param_size
                                if total_chosen + num_chosen <= self.budget:
                                    mask = torch.ones_like(param, dtype=torch.bool, device=param.device)
                                    total_chosen += num_chosen
                                else:
                                    # Select as many as possible to stay within the budget
                                    remaining = self.budget - total_chosen
                                    mask = torch.zeros_like(param, dtype=torch.bool, device=param.device)
                                    mask[:remaining] = True
                                    total_chosen = self.budget
                            elif param.ndim == 0:
                                # Scalar parameters
                                if total_chosen < self.budget:
                                    mask = torch.ones_like(param, dtype=torch.bool, device=param.device)
                                    total_chosen += 1
                                else:
                                    mask = None
                                    param.requires_grad = False
                            else:
                                # Non-bias parameters
                                mask = None
                                param.requires_grad = False
                    else:
                        mask = None

                    chosen_masks.append(mask)
                param_group["chosen_masks"] = chosen_masks
            
            if total_chosen > self.budget:
                raise Exception(f"Used initial budget: {total_chosen} has exceeded total budget: {self.budget}")


        @torch.no_grad()
        def update_chosen(self):
            """
            BitFit does not update chosen masks after initialization.
            """
            pass

    return Bitfit
