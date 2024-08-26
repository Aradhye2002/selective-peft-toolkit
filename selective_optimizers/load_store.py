from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from .utils import set_chosen
import torch.nn as nn

@dataclass
class SelectivePeftSummary:
    # List of relevant info of all parameters with requires_grad = True
    values: List[torch.Tensor] = None
    pointers: List[torch.Tensor] = None
    shapes: List[torch.Size] = None
    bn_metadata: List[Dict[str, torch.Size]] = None
    budget_used : int = 0
    
def create_summary_from_param_groups(param_groups, modules):
    values = []
    pointers = []
    shapes = []
    bn_metadata = []
    budget_used = 0
    for param_group in param_groups:
        num_params = len(param_group["params"])
        for i in range(num_params):
            param = param_group["params"][i]
            chosen_mask = param_group["chosen_masks"][i]
            budget_used += chosen_mask.count_nonzero()
            chosen_values = param.data[chosen_mask].cpu()
            chosen_tensor = chosen_mask.nonzero().cpu()
            shape = param.shape
            values.append(chosen_values)
            pointers.append(chosen_tensor)
            shapes.append(shape)
    for m in modules:
        if (isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
            metadata = {"running_mean" : m.running_mean.cpu(), "running_var" : m.running_var.cpu(), "num_batches_tracked" : m.num_batches_tracked.cpu()}
            bn_metadata.append(metadata)
    summary = SelectivePeftSummary(values, pointers, bn_metadata, shapes, budget_used)
    return summary

def load_weights_from_summary(model : nn.Module, summary : SelectivePeftSummary):
    values = summary.values
    pointers = summary.pointers
    shapes = summary.shapes
    bn_metadata = summary.bn_metadata
    
    # Load sparse parameter weights
    num_params = len(values)
    curr_idx = 0
    for param in model.parameters():
        device = param.device
        if param.requires_grad:
            chosen_values = values[curr_idx].to(device)
            chosen_tensor = pointers[curr_idx].to(device)
            shape = shapes[curr_idx]
            assert param.shape == shape, "Mismatch in shape of parameters between the summary and the model!"
            set_chosen(param.data, chosen_tensor, chosen_values)
            curr_idx += 1
    assert curr_idx == num_params, "Mismatch in number of parameters between the summary and the model!"
    
    # Load full batch statistics for all BN layers
    num_bn = len(bn_metadata)
    curr_idx = 0
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
            device = m.running_mean.device
            metadata = bn_metadata[curr_idx]
            running_mean = metadata["running_mean"].to(device)
            running_var = metadata["running_var"].to(device)
            num_batches_tracked = metadata["num_batches_tracked"].to(device)
            m.running_mean = running_mean
            m.running_var = running_var
            m.num_batches_tracked = num_batches_tracked
            curr_idx +=1
    assert curr_idx == num_bn, "Mismatch in number of batchnorm layers between the summary and the model!"
    
    
def write_summary_to_disk(path : str, summary : SelectivePeftSummary):
    try:
        values = summary.values
        pointers = summary.pointers
        bn_metadata = summary.bn_metadata
        shapes = summary.shapes
        budget_used = summary.budget_used
        obj = (values, pointers, bn_metadata, shapes, budget_used)
        torch.save(obj, path)
        return True
    except:
        return False
    
def load_summary_from_disk(path : str):
    obj = torch.load(path)
    values, pointers, shapes, bn_metadata, budget_used = obj
    summary = SelectivePeftSummary(values, pointers, shapes, bn_metadata, budget_used)
    return summary
    