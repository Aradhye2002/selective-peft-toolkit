from dataclasses import dataclass
from typing import List, Dict
import torch
from .utils import set_chosen
import torch.nn as nn

@dataclass
class SelectivePeftSummary:
    """
    Dataclass to store the summary of selective PEFT.
    """
    values: List[torch.Tensor]
    pointers: List[torch.Tensor]
    shapes: List[torch.Size]
    bn_metadata: List[Dict[str, torch.Tensor]]
    budget_used: int

def create_summary_from_param_groups(param_groups, model):
    """
    Create a summary from the parameter groups and model modules.

    Args:
        param_groups (List[Dict]): The parameter groups from the optimizer.
        model (nn.Module): The model.

    Returns:
        SelectivePeftSummary: A dataclass containing the summary.
    """
    values = []
    pointers = []
    shapes = []
    bn_metadata = []
    budget_used = 0

    param2info = {}
    for param_group in param_groups:
        params = param_group["params"]
        chosen_masks = param_group["chosen_masks"]
        for param, chosen_mask in zip(params, chosen_masks):
            if chosen_mask is None:
                chosen_values = None
                chosen_indices = None
            else:
                budget_used += chosen_mask.sum().item()
                chosen_values = param.data[chosen_mask].cpu()
                chosen_indices = chosen_mask.nonzero().cpu()
            info = (chosen_values, chosen_indices, param.shape)
            param2info[id(param)] = info
    
    for param in model.parameters():
        if id(param) in param2info:
            chosen_values, chosen_indices, shape = param2info[id(param)]
        else:
            chosen_values, chosen_indices, shape = None, None, None
        values.append(chosen_values)
        pointers.append(chosen_indices)
        shapes.append(shape)
    # Collect batch norm metadata
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            metadata = {
                "running_mean": module.running_mean.cpu(),
                "running_var": module.running_var.cpu(),
                "num_batches_tracked": module.num_batches_tracked.cpu()
            }
            bn_metadata.append(metadata)

    summary = SelectivePeftSummary(values, pointers, shapes, bn_metadata, budget_used)
    return summary

def load_weights_from_summary(model: nn.Module, summary: SelectivePeftSummary):
    """
    Load the weights from the summary back into the model.

    Args:
        model (nn.Module): The model to load weights into.
        summary (SelectivePeftSummary): The summary containing the weights.
    """
    values = summary.values
    pointers = summary.pointers
    shapes = summary.shapes
    bn_metadata = summary.bn_metadata

    # Load sparse parameter weights
    idx = 0
    for param in model.parameters():
        if values[idx] is not None:
            device = param.device
            chosen_values = values[idx].to(device)
            chosen_indices = pointers[idx].to(device)
            shape = shapes[idx]
            assert param.shape == shape, "Mismatch in parameter shapes between summary and model."
            set_chosen(param.data, chosen_indices, chosen_values)
        idx += 1

    assert idx == len(values), "Mismatch in number of parameters between summary and model."

    # Load batch norm statistics
    bn_idx = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            device = module.running_mean.device
            metadata = bn_metadata[bn_idx]
            module.running_mean.data.copy_(metadata["running_mean"].to(device))
            module.running_var.data.copy_(metadata["running_var"].to(device))
            module.num_batches_tracked.data.copy_(metadata["num_batches_tracked"].to(device))
            bn_idx += 1

    assert bn_idx == len(bn_metadata), "Mismatch in number of batch norm layers between summary and model."

def write_summary_to_disk(path: str, summary: SelectivePeftSummary):
    """
    Write the summary to disk.

    Args:
        path (str): The file path to save the summary.
        summary (SelectivePeftSummary): The summary to save.
    """
    try:
        torch.save(summary, path)
        return True
    except Exception as e:
        print(f"Error saving summary to disk: {e}")
        return False

def load_summary_from_disk(path: str) -> SelectivePeftSummary:
    """
    Load the summary from disk.

    Args:
        path (str): The file path to load the summary from.

    Returns:
        SelectivePeftSummary: The loaded summary.
    """
    summary = torch.load(path)
    return summary
