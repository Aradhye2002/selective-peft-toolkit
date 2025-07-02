import torch
from ..utils import set_chosen

def get_pafi(base_optimizer):
    """
    Factory function to create the Pafi optimizer class, inheriting from base_optimizer.
    The Pafi optimizer selects parameters to fine-tune based on their magnitude.
    """

    class Pafi(base_optimizer):
        """
        Pafi optimizer class that selects parameters with small magnitudes to fine-tune.
        """

        @torch.no_grad()
        def init_chosen(self):
            """
            Initialize the chosen masks by selecting parameters with the smallest magnitudes,
            respecting the global budget.
            """
            total_chosen = 0
            mode2_params = []  # will hold (param, group_idx) for reconstruction

            # --------------------------------------------------
            # 1) Figure out how many parameters come from choose_all
            #    and also collect mode=2 params for later
            # --------------------------------------------------
            for group_idx, param_group in enumerate(self.param_groups):
                # Check group mode
                if param_group.get("choose_all", False):
                    # Add all parameters in this group to total_chosen
                    for param in param_group["params"]:
                        if param.requires_grad:
                            total_chosen += param.numel()
                else:
                    # Mode=2 group
                    for param in param_group["params"]:
                        if param.requires_grad:
                            mode2_params.append((param, group_idx))

            # --------------------------------------------------
            # 2) Check if we already exceeded the budget
            # --------------------------------------------------
            if total_chosen > self.budget:
                raise ValueError(
                    f"[Pafi:init_chosen] 'choose_all' groups alone exceed the budget "
                    f"(got {total_chosen} > budget {self.budget})."
                )

            # --------------------------------------------------
            # 3) Flatten all mode=2 param magnitudes into one big vector
            # --------------------------------------------------
            # We'll store them in 'all_abs' for top-k. Also track their shapes for reconstruction.
            device = self.param_groups[0]["params"][0].device
            all_abs = []
            param_shapes = []
            for (p, _) in mode2_params:
                all_abs.append(p.abs().view(-1).to(device))
                param_shapes.append(p.shape)

            if len(all_abs) > 0:
                all_abs = torch.cat(all_abs, dim=0)
            else:
                # If no mode=2 params, we can just assign masks for choose_all below
                all_abs = torch.tensor([], device=device)

            # --------------------------------------------------
            # 4) Choose the smallest K = budget - total_chosen from mode=2
            # --------------------------------------------------
            k = self.budget - total_chosen

            # If k == 0, no param from mode=2 is allowed
            if k > 0 and len(all_abs) > 0:
                # Get the indices of the k smallest elements
                sorted_indices = torch.argsort(all_abs)
                chosen_flat_indices = sorted_indices[:k]
            else:
                # No budget left or no mode=2 params
                chosen_flat_indices = torch.tensor([], dtype=torch.long, device=all_abs.device)

            # --------------------------------------------------
            # 5) Build a boolean mask *per parameter* in mode=2
            # --------------------------------------------------
            # We know how many elements each param had. We'll iterate in the same order.
            chosen_masks_mode2 = []
            offset = 0
            for (param, group_idx) in mode2_params:
                size = param.numel()
                # Make a zero mask first
                mask = torch.zeros(size, dtype=torch.bool, device=param.device)

                # Find the subset of chosen_flat_indices that belong to [offset, offset+size)
                # We'll do a quick check via:
                start, end = offset, offset + size
                # These bools tell us which chosen indices fall in that param's range
                in_range = (chosen_flat_indices >= start) & (chosen_flat_indices < end)
                if in_range.any():
                    # Those chosen indices that fall in range, we shift by -offset so they
                    # index into [0..size).
                    relevant_indices = chosen_flat_indices[in_range] - start
                    mask[relevant_indices] = True

                # Reshape back to param's shape
                mask = mask.view(param.shape)
                chosen_masks_mode2.append((mask, group_idx))
                offset += size

            # --------------------------------------------------
            # 6) Now assign chosen_masks to each param group
            # --------------------------------------------------
            # We'll keep track of an index into chosen_masks_mode2 for each parameter in mode=2
            mode2_counter = 0
            for group_idx, param_group in enumerate(self.param_groups):
                if param_group.get("choose_all", False):
                    # All params are chosen
                    chosen_masks = []
                    for param in param_group["params"]:
                        if param.requires_grad:
                            # set mask=1 everywhere
                            chosen_masks.append(torch.ones_like(param.data, dtype=torch.bool))
                        else:
                            chosen_masks.append(None)
                    param_group["chosen_masks"] = chosen_masks
                else:
                    # mode=2 group
                    chosen_masks = []
                    for param in param_group["params"]:
                        if param.requires_grad:
                            # retrieve the precomputed mask
                            mask, stored_group_idx = chosen_masks_mode2[mode2_counter]
                            if stored_group_idx != group_idx:
                                raise RuntimeError("Internal mismatch of group indices.")
                            chosen_masks.append(mask)
                            mode2_counter += 1
                        else:
                            chosen_masks.append(torch.zeros_like(param.data, dtype=torch.bool))
                    param_group["chosen_masks"] = chosen_masks

            # --------------------------------------------------
            # 7) Optional: check final usage
            # --------------------------------------------------
            final_used = 0
            for param_group in self.param_groups:
                for mask in param_group["chosen_masks"]:
                    final_used += mask.sum().item()

            if final_used > self.budget:
                raise RuntimeError(
                    f"[Pafi:init_chosen] final_used={final_used} exceeds budget={self.budget}."
                )

        @torch.no_grad()
        def update_chosen(self):
            """
            Pafi does not update chosen masks after initialization.
            """
            pass

    return Pafi
