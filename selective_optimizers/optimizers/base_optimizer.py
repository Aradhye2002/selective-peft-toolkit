import torch
from ..load_store import create_summary_from_param_groups
import uuid
import os

def get_base_optimizer(optimizer):
    """
    Factory function to create a BaseOptimizer class that extends the given optimizer.
    """

    class BaseOptimizer(optimizer):
        """
        BaseOptimizer class that provides the foundation for selective parameter updates.
        """

        def __init__(self, *args, **kwargs):
            """
            Initialize the BaseOptimizer.

            Args:
                *args: Positional arguments for the base optimizer.
                **kwargs: Keyword arguments for the base optimizer, including:
                    - verify (bool): Whether to perform verification after training.
                    - budget (int): The total budget of parameters allowed to update.
                    - initial_params_path (str): File path to save initial parameters for verification.
            """
            self.verify = kwargs.pop('verify', True)
            self.budget = kwargs.pop('budget', None)
            
            unique_id = str(uuid.uuid4())
            self.initial_params_path = kwargs.pop('initial_params_path', f"/tmp/initial_params_{unique_id}.pt")
            
            # Ensure weight decay is zero since non-chosen parameters should not decay on their own
            weight_decay = kwargs.get("weight_decay", 0.0)
            if weight_decay != 0.0:
                print(f"Detected non-zero weight_decay: {weight_decay}, forcefully setting to 0.0")
            kwargs["weight_decay"] = 0.0
            
            super().__init__(*args, **kwargs)
            self.init_chosen()
            self.pre_train_work()
            
            self.global_step = 0

        def pre_train_work(self):
            """
            Pre-training work to save initial parameters if verification is enabled.
            """
            if self.verify:
                # Save initial parameters to disk for later verification
                initial_params = []
                for param_group in self.param_groups:
                    params = [param.detach().cpu().clone() for param in param_group["params"]]
                    initial_params.append(params)
                torch.save(initial_params, self.initial_params_path)

        def get_budget_used(self):
            """
            Calculate the total number of parameters currently chosen for updating.

            Returns:
                int: The number of parameters that are being updated.
            """
            budget_used = 0
            for param_group in self.param_groups:
                for chosen_mask in param_group["chosen_masks"]:
                    if chosen_mask is not None:
                        budget_used += chosen_mask.sum().item()
            return budget_used

        def post_train_work(self):
            """
            Post-training work to verify that the budget has not been exceeded and that
            non-chosen parameters have not changed.
            """            
            if self.verify:
                # Load initial parameters from disk
                initial_params = torch.load(self.initial_params_path, weights_only=True)
                budget_used = self.get_budget_used()
                if budget_used > self.budget:
                    print(f"Budget exceeded! Budget allocated: {self.budget}, budget used: {budget_used}")
                else:
                    print(f"Budget allocated: {self.budget}, budget used: {budget_used}")
                for i, param_group in enumerate(self.param_groups):
                    params = param_group["params"]
                    chosen_masks = param_group["chosen_masks"]
                    for j, (param, chosen_mask) in enumerate(zip(params, chosen_masks)):
                        initial_param = initial_params[i][j].to(param.device)
                        if chosen_mask is not None:
                            not_chosen_mask = ~chosen_mask
                            # Move initial parameters to the current device
                            # Check if non-chosen parameters have remained unchanged
                            if not torch.allclose(param.data[not_chosen_mask], initial_param.data[not_chosen_mask]):
                                print(f"Verification failed for param {j} in param_group {i}")
                                return  # Early exit on verification failure
                        else:
                            if not torch.allclose(param.data, initial_param.data):
                                print(f"Verification failed for param {j} in param_group {i}")
                                return  # Early exit on verification failure
                            
                print("Verification successful: Non-chosen parameters have not changed.")
                os.remove(self.initial_params_path)

        def mask_gradients(self):
            """
            Mask gradients of non-chosen parameters to zero before the optimizer step.
            """
            for param_group in self.param_groups:
                params = param_group["params"]
                chosen_masks = param_group.get("chosen_masks")
                for param, chosen_mask in zip(params, chosen_masks):
                    if chosen_mask is not None:
                        not_chosen_mask = ~chosen_mask
                        param.grad.data[not_chosen_mask] = 0.0

        @torch.no_grad()
        def move_chosen_to_param_device(self):
            for param_group in self.param_groups:
                for i, (param, chosen_mask) in enumerate(zip(param_group["params"], param_group["chosen_masks"])):
                    if chosen_mask is not None:
                        param_group["chosen_masks"][i] = param_group["chosen_masks"][i].to(param.device)

        def step(self, *args, **kwargs):
            """
            Perform a single optimization step after updating chosen masks and masking gradients.
            """
            if self.global_step == 0:
                self.move_chosen_to_param_device()
            self.update_chosen()
            self.mask_gradients()
            super().step(*args, **kwargs)

        def init_chosen(self):
            """
            Initialize the chosen masks for each parameter group.
            Must be overridden by subclasses.
            """
            raise NotImplementedError("init_chosen method must be overridden in the subclass.")

        def update_chosen(self):
            """
            Update the chosen masks based on the custom selection logic.
            Must be overridden by subclasses.
            """
            raise NotImplementedError("update_chosen method must be overridden in the subclass.")

        def get_summary(self, model):
            """
            Create a summary of the model's parameters and batch norm statistics.

            Args:
                model (torch.nn.Module): The model to summarize.

            Returns:
                SelectivePeftSummary: A dataclass containing the summary.
            """
            # modules is needed in order to obtain batch statistics for batchnorm which is not tracked by the optimizer
            summary = create_summary_from_param_groups(self.param_groups, model)
            return summary

    return BaseOptimizer
