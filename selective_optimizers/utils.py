import torch

def get_chosen(param, chosen_indices):
    """
    Retrieve the values of the chosen indices from the parameter tensor.

    Args:
        param (torch.Tensor): The parameter tensor.
        chosen_indices (torch.Tensor): Tensor of indices indicating the chosen elements.

    Returns:
        torch.Tensor: Flattened tensor of chosen parameter values.
    """
    if chosen_indices.numel() == 0:
        return torch.tensor([], device=param.device)
    else:
        return param[tuple(chosen_indices.t())]

def set_chosen(param, chosen_indices, values):
    """
    Set the values of the chosen indices in the parameter tensor.

    Args:
        param (torch.Tensor): The parameter tensor.
        chosen_indices (torch.Tensor): Tensor of indices indicating where to set values.
        values (torch.Tensor): Values to set at the chosen indices.
    """
    if chosen_indices.numel() == 0:
        return
    else:
        param[tuple(chosen_indices.t())] = values

def get_not_chosen(param, chosen_mask):
    """
    Retrieve the values of the not chosen elements from the parameter tensor.

    Args:
        param (torch.Tensor): The parameter tensor.
        chosen_mask (torch.Tensor): Boolean mask where True indicates chosen elements.

    Returns:
        torch.Tensor: Flattened tensor of not chosen parameter values.
    """
    not_chosen_values = param[~chosen_mask]
    return not_chosen_values
